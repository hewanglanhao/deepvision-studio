package com.deepvision.studio.training;

import com.deepvision.studio.training.TrainingDtos.HistogramBin;
import com.deepvision.studio.training.TrainingDtos.SplitRequest;
import com.deepvision.studio.training.TrainingDtos.StartTrainingRequest;
import com.deepvision.studio.training.TrainingDtos.TrainingConfigRequest;
import com.deepvision.studio.training.TrainingDtos.TrainingControlResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingDatasetDetail;
import com.deepvision.studio.training.TrainingDtos.TrainingMetricMessage;
import com.deepvision.studio.training.TrainingDtos.TrainingStartResponse;
import com.deepvision.studio.training.TrainingDtos.TrainingStatusResponse;
import com.deepvision.studio.training.TrainingDtos.WeightHistogramResponse;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

@Service
public class TrainingJobService {
  private static final DateTimeFormatter JOB_ID_TIME = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

  private final TrainingDatasetService datasetService;
  private final ObjectMapper objectMapper;
  private final ScheduledExecutorService executor = Executors.newScheduledThreadPool(2);
  private final Map<String, TrainingJob> jobs = new ConcurrentHashMap<>();
  private final Map<String, CopyOnWriteArraySet<WebSocketSession>> sessions = new ConcurrentHashMap<>();
  private final String streamBaseUrl;

  public TrainingJobService(
      TrainingDatasetService datasetService,
      ObjectMapper objectMapper,
      @Value("${deepvision.training.stream-base-url:ws://127.0.0.1:8080}") String streamBaseUrl
  ) {
    this.datasetService = datasetService;
    this.objectMapper = objectMapper;
    this.streamBaseUrl = trimTrailingSlash(streamBaseUrl);
  }

  public TrainingStartResponse start(StartTrainingRequest request) {
    TrainingDatasetDetail dataset = datasetService.getDetail(request.datasetId());
    validateSplit(request.split());
    if (!dataset.hasLabels()) {
      throw new IllegalArgumentException("Dataset has no labels and cannot be used for supervised training.");
    }

    TrainingConfigRequest config = request.config();
    int totalEpochs = valueOrDefault(config.totalEpochs(), 20);
    int batchSize = valueOrDefault(config.batchSize(), 32);
    double trainSamples = Math.max(1, dataset.sampleCount() * request.split().train());
    int totalBatches = Math.max(1, (int) Math.ceil(trainSamples / batchSize));

    String jobId = nextJobId();
    TrainingJob job = new TrainingJob(
        jobId,
        request.datasetId(),
        "running",
        totalEpochs,
        totalBatches,
        batchSize,
        config.learningRate() == null || config.learningRate() <= 0 ? 0.001 : config.learningRate(),
        config.lrDecay() == null || config.lrDecay() <= 0 ? 0.9 : config.lrDecay(),
        config.scheduler() == null ? "none" : config.scheduler(),
        request.split().val() > 0,
        Instant.now()
    );
    jobs.put(jobId, job);
    ScheduledFuture<?> future = executor.scheduleAtFixedRate(() -> tick(jobId), 400, 550, TimeUnit.MILLISECONDS);
    job.setFuture(future);
    return new TrainingStartResponse(
        jobId,
        job.status(),
        job.totalEpochs(),
        job.totalBatches(),
        streamBaseUrl + "/api/training/stream?jobId=" + jobId
    );
  }

  public TrainingStatusResponse status(String jobId) {
    return getJob(jobId).toStatus();
  }

  public WeightHistogramResponse histogram(String jobId) {
    TrainingJob job = getJob(jobId);
    TrainingMetricMessage metric = job.latestMetric();
    double mean = metric == null ? 0 : metric.weightMean();
    double std = metric == null ? 0.16 : Math.max(0.02, metric.weightStd());
    List<HistogramBin> bins = new ArrayList<>();
    for (int i = -5; i <= 5; i += 1) {
      double value = mean + i * std * 0.35;
      int count = (int) Math.round(80 * Math.exp(-0.5 * Math.pow(i / 2.1, 2))) + Math.max(0, job.epoch());
      bins.add(new HistogramBin(String.format(java.util.Locale.US, "%.2f", value), count));
    }
    return new WeightHistogramResponse(jobId, job.epoch(), bins);
  }

  public TrainingControlResponse pause(String jobId) {
    TrainingJob job = getJob(jobId);
    synchronized (job) {
      if ("running".equals(job.status())) {
        job.setStatus("paused");
      }
      return new TrainingControlResponse(jobId, job.status(), "Training paused.");
    }
  }

  public TrainingControlResponse resume(String jobId) {
    TrainingJob job = getJob(jobId);
    synchronized (job) {
      if ("paused".equals(job.status())) {
        job.setStatus("running");
      }
      return new TrainingControlResponse(jobId, job.status(), "Training resumed.");
    }
  }

  public TrainingControlResponse stop(String jobId) {
    TrainingJob job = getJob(jobId);
    synchronized (job) {
      job.cancelFuture();
      job.setStatus("stopped");
      return new TrainingControlResponse(jobId, job.status(), "Training stopped.");
    }
  }

  public TrainingControlResponse reset(String jobId) {
    TrainingJob job = getJob(jobId);
    synchronized (job) {
      job.cancelFuture();
      job.reset();
      ScheduledFuture<?> future = executor.scheduleAtFixedRate(() -> tick(jobId), 400, 550, TimeUnit.MILLISECONDS);
      job.setFuture(future);
      return new TrainingControlResponse(jobId, job.status(), "Training reset.");
    }
  }

  public TrainingControlResponse save(String jobId) {
    TrainingJob job = getJob(jobId);
    return new TrainingControlResponse(jobId, job.status(), "Training saved.");
  }

  public void addSession(String jobId, WebSocketSession session) {
    TrainingJob job = getJob(jobId);
    sessions.computeIfAbsent(jobId, ignored -> new CopyOnWriteArraySet<>()).add(session);
    TrainingMetricMessage latest = job.latestMetric();
    if (latest != null) {
      send(session, latest);
    }
  }

  public void removeSession(WebSocketSession session) {
    sessions.values().forEach(set -> set.remove(session));
  }

  @PreDestroy
  void shutdown() {
    jobs.values().forEach(TrainingJob::cancelFuture);
    executor.shutdownNow();
  }

  private void tick(String jobId) {
    TrainingJob job = jobs.get(jobId);
    if (job == null) {
      return;
    }
    TrainingMetricMessage metric;
    synchronized (job) {
      if (!"running".equals(job.status())) {
        return;
      }
      if (job.epoch() >= job.totalEpochs()) {
        job.setStatus("completed");
        job.cancelFuture();
        return;
      }
      metric = job.nextMetric();
      if (job.epoch() >= job.totalEpochs()) {
        job.setStatus("completed");
        job.cancelFuture();
      }
    }
    broadcast(jobId, metric);
  }

  private void broadcast(String jobId, TrainingMetricMessage metric) {
    CopyOnWriteArraySet<WebSocketSession> jobSessions = sessions.get(jobId);
    if (jobSessions == null || jobSessions.isEmpty()) {
      return;
    }
    String payload;
    try {
      payload = objectMapper.writeValueAsString(metric);
    } catch (JsonProcessingException ex) {
      return;
    }
    for (WebSocketSession session : jobSessions) {
      if (!session.isOpen()) {
        jobSessions.remove(session);
        continue;
      }
      try {
        session.sendMessage(new TextMessage(payload));
      } catch (IOException ex) {
        jobSessions.remove(session);
      }
    }
  }

  private void send(WebSocketSession session, TrainingMetricMessage metric) {
    if (!session.isOpen()) {
      return;
    }
    try {
      session.sendMessage(new TextMessage(objectMapper.writeValueAsString(metric)));
    } catch (IOException ex) {
      removeSession(session);
    }
  }

  private TrainingJob getJob(String jobId) {
    TrainingJob job = jobs.get(jobId);
    if (job == null) {
      throw new IllegalArgumentException("Training job not found.");
    }
    return job;
  }

  private void validateSplit(SplitRequest split) {
    if (split.train() <= 0) {
      throw new IllegalArgumentException("split.train must be greater than 0.");
    }
    if (split.val() < 0 || split.test() < 0) {
      throw new IllegalArgumentException("split ratios cannot be negative.");
    }
    double sum = split.train() + split.val() + split.test();
    if (Math.abs(sum - 1.0) > 0.0001) {
      throw new IllegalArgumentException("split.train + split.val + split.test must equal 1.0.");
    }
  }

  private int valueOrDefault(Integer value, int fallback) {
    return value == null || value <= 0 ? fallback : value;
  }

  private String nextJobId() {
    return "train-" + LocalDateTime.now().format(JOB_ID_TIME) + "-" + UUID.randomUUID().toString().substring(0, 8);
  }

  private String trimTrailingSlash(String value) {
    if (value == null || value.isBlank()) {
      return "ws://127.0.0.1:8080";
    }
    return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
  }

  static String jobIdFromSession(WebSocketSession session) {
    URI uri = session.getUri();
    if (uri == null || uri.getQuery() == null) {
      return null;
    }
    for (String part : uri.getQuery().split("&")) {
      String[] pair = part.split("=", 2);
      if (pair.length == 2 && Objects.equals(pair[0], "jobId")) {
        return pair[1];
      }
    }
    return null;
  }

  private static final class TrainingJob {
    private final String jobId;
    private final String datasetId;
    private String status;
    private final int totalEpochs;
    private final int totalBatches;
    private final int batchSize;
    private final double initialLearningRate;
    private final double lrDecay;
    private final String scheduler;
    private final boolean hasValidation;
    private Instant startedAt;
    private int epoch;
    private int batch;
    private TrainingMetricMessage latestMetric;
    private ScheduledFuture<?> future;

    private TrainingJob(
        String jobId,
        String datasetId,
        String status,
        int totalEpochs,
        int totalBatches,
        int batchSize,
        double initialLearningRate,
        double lrDecay,
        String scheduler,
        boolean hasValidation,
        Instant startedAt
    ) {
      this.jobId = jobId;
      this.datasetId = datasetId;
      this.status = status;
      this.totalEpochs = totalEpochs;
      this.totalBatches = totalBatches;
      this.batchSize = batchSize;
      this.initialLearningRate = initialLearningRate;
      this.lrDecay = lrDecay;
      this.scheduler = scheduler;
      this.hasValidation = hasValidation;
      this.startedAt = startedAt;
    }

    synchronized TrainingMetricMessage nextMetric() {
      epoch += 1;
      batch = totalBatches;
      double progress = epoch / (double) totalEpochs;
      double curve = 1 - Math.exp(-3.2 * progress);
      double oscillation = Math.sin(epoch * 0.9) * 0.015;
      double loss = Math.max(0.05, 1.75 * Math.exp(-2.7 * progress) + 0.05 + oscillation);
      double accuracy = Math.min(0.995, 0.18 + 0.78 * curve - Math.abs(oscillation));
      Double valLoss = hasValidation ? Math.max(0.06, loss + 0.05 + Math.sin(epoch * 0.55) * 0.018) : null;
      Double valAccuracy = hasValidation ? Math.max(0, Math.min(0.99, accuracy - 0.035 + Math.cos(epoch * 0.4) * 0.012)) : null;
      double lr = currentLearningRate(epoch);
      long elapsed = Duration.between(startedAt, Instant.now()).toSeconds();
      long eta = epoch == 0 ? 0 : Math.max(0, Math.round(elapsed * (totalEpochs - epoch) / (double) epoch));
      double gradientNorm = Math.max(0.015, 1.05 * Math.exp(-1.9 * progress) + Math.abs(Math.sin(epoch * 0.7)) * 0.08);
      String gradientStatus = gradientNorm < 0.04 ? "vanishing" : gradientNorm > 2.5 ? "exploding" : "stable";
      double weightMean = Math.sin(epoch * 0.35 + batchSize * 0.001) * 0.012;
      double weightStd = Math.max(0.035, 0.18 - progress * 0.07 + Math.cos(epoch * 0.3) * 0.008);
      latestMetric = new TrainingMetricMessage(
          "metric",
          jobId,
          epoch,
          epoch,
          batch,
          totalEpochs,
          totalBatches,
          round(loss),
          roundNullable(valLoss),
          round(accuracy),
          roundNullable(valAccuracy),
          lr,
          elapsed,
          eta,
          round(gradientNorm),
          round(weightMean),
          round(weightStd),
          gradientStatus
      );
      return latestMetric;
    }

    synchronized TrainingStatusResponse toStatus() {
      TrainingMetricMessage metric = latestMetric;
      long elapsed = Duration.between(startedAt, Instant.now()).toSeconds();
      long eta = metric == null ? 0 : metric.etaSeconds();
      return new TrainingStatusResponse(
          jobId,
          status,
          epoch,
          batch,
          totalEpochs,
          totalBatches,
          metric == null ? 1.7 : metric.loss(),
          metric == null ? (hasValidation ? 1.78 : null) : metric.valLoss(),
          metric == null ? 0.2 : metric.accuracy(),
          metric == null ? (hasValidation ? 0.18 : null) : metric.valAccuracy(),
          elapsed,
          eta
      );
    }

    synchronized void reset() {
      status = "running";
      startedAt = Instant.now();
      epoch = 0;
      batch = 0;
      latestMetric = null;
    }

    synchronized void setFuture(ScheduledFuture<?> future) {
      this.future = future;
    }

    synchronized void cancelFuture() {
      if (future != null) {
        future.cancel(false);
        future = null;
      }
    }

    synchronized String status() {
      return status;
    }

    synchronized void setStatus(String status) {
      this.status = status;
    }

    int totalEpochs() {
      return totalEpochs;
    }

    int totalBatches() {
      return totalBatches;
    }

    synchronized int epoch() {
      return epoch;
    }

    synchronized TrainingMetricMessage latestMetric() {
      return latestMetric;
    }

    @SuppressWarnings("unused")
    String datasetId() {
      return datasetId;
    }

    private double currentLearningRate(int currentEpoch) {
      if ("step".equalsIgnoreCase(scheduler)) {
        int steps = Math.max(0, currentEpoch / 5);
        return round(initialLearningRate * Math.pow(lrDecay, steps));
      }
      if ("exp".equalsIgnoreCase(scheduler) || "exponential".equalsIgnoreCase(scheduler)) {
        return round(initialLearningRate * Math.pow(lrDecay, Math.max(0, currentEpoch - 1)));
      }
      return round(initialLearningRate);
    }

    private double round(double value) {
      return Math.round(value * 10000.0) / 10000.0;
    }

    private Double roundNullable(Double value) {
      return value == null ? null : round(value);
    }
  }
}
