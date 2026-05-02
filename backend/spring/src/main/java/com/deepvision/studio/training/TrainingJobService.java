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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

@Service
public class TrainingJobService {
  private static final DateTimeFormatter JOB_ID_TIME = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

  private final TrainingDatasetService datasetService;
  private final ObjectMapper objectMapper;
  private final ExecutorService executor = Executors.newCachedThreadPool();
  private final Map<String, TrainingJob> jobs = new ConcurrentHashMap<>();
  private final Map<String, CopyOnWriteArraySet<WebSocketSession>> sessions = new ConcurrentHashMap<>();
  private final String streamBaseUrl;
  private final Path datasetRoot;
  private final Path jobRoot;
  private final Path workerScript;
  private final String pythonExecutable;

  public TrainingJobService(
      TrainingDatasetService datasetService,
      ObjectMapper objectMapper,
      @Value("${deepvision.training.stream-base-url:ws://127.0.0.1:8080}") String streamBaseUrl,
      @Value("${deepvision.datasets.root}") String datasetRoot,
      @Value("${deepvision.training.jobs-root:./training-jobs}") String jobRoot,
      @Value("${deepvision.training.python-executable:python}") String pythonExecutable,
      @Value("${deepvision.training.worker-script:../python-training/training_worker.py}") String workerScript
  ) {
    this.datasetService = datasetService;
    this.objectMapper = objectMapper;
    this.streamBaseUrl = trimTrailingSlash(streamBaseUrl);
    this.datasetRoot = Path.of(datasetRoot).toAbsolutePath().normalize();
    this.jobRoot = Path.of(jobRoot).toAbsolutePath().normalize();
    this.pythonExecutable = pythonExecutable;
    this.workerScript = Path.of(workerScript).toAbsolutePath().normalize();
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
    TrainingJob job = new TrainingJob(jobId, request, totalEpochs, totalBatches);
    jobs.put(jobId, job);
    startPythonWorker(job);
    return new TrainingStartResponse(
        jobId,
        job.status(),
        totalEpochs,
        totalBatches,
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
    writeControl(job, "paused");
    job.setStatus("paused");
    return new TrainingControlResponse(jobId, job.status(), "Training paused.");
  }

  public TrainingControlResponse resume(String jobId) {
    TrainingJob job = getJob(jobId);
    writeControl(job, "running");
    if ("paused".equals(job.status())) {
      job.setStatus("running");
    }
    return new TrainingControlResponse(jobId, job.status(), "Training resumed.");
  }

  public TrainingControlResponse stop(String jobId) {
    TrainingJob job = getJob(jobId);
    writeControl(job, "stopped");
    job.destroyProcess();
    job.setStatus("stopped");
    return new TrainingControlResponse(jobId, job.status(), "Training stopped.");
  }

  public TrainingControlResponse reset(String jobId) {
    TrainingJob job = getJob(jobId);
    writeControl(job, "stopped");
    job.destroyProcess();
    job.reset();
    startPythonWorker(job);
    return new TrainingControlResponse(jobId, job.status(), "Training reset.");
  }

  public TrainingControlResponse save(String jobId) {
    TrainingJob job = getJob(jobId);
    return new TrainingControlResponse(jobId, job.status(), "Training metadata saved. Model checkpoint export is not implemented yet.");
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
    jobs.values().forEach(TrainingJob::destroyProcess);
    executor.shutdownNow();
  }

  private void startPythonWorker(TrainingJob job) {
    try {
      Files.createDirectories(jobRoot);
      Files.createDirectories(job.directory());
      job.writeRequest();
      writeControl(job, "running");
      Process process = new ProcessBuilder(
          pythonExecutable,
          "-B",
          workerScript.toString(),
          "--request",
          job.requestFile().toString()
      )
          .directory(workerScript.getParent().toFile())
          .redirectErrorStream(true)
          .start();
      job.setProcess(process);
      job.setStatus("running");
      executor.submit(() -> readWorkerOutput(job));
      executor.submit(() -> waitForWorkerExit(job));
    } catch (IOException ex) {
      job.setStatus("stopped");
      throw new IllegalArgumentException("Failed to start Python training worker: " + ex.getMessage());
    }
  }

  private void readWorkerOutput(TrainingJob job) {
    Process process = job.process();
    if (process == null) {
      return;
    }
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
      String line;
      while ((line = reader.readLine()) != null) {
        handleWorkerLine(job, line.trim());
      }
    } catch (IOException ex) {
      if (!"stopped".equals(job.status())) {
        job.setStatus("stopped");
      }
    }
  }

  private void handleWorkerLine(TrainingJob job, String line) {
    if (line.isBlank()) {
      return;
    }
    JsonNode node;
    try {
      node = objectMapper.readTree(line);
    } catch (JsonProcessingException ex) {
      return;
    }
    String type = node.path("type").asText("");
    if ("metric".equals(type)) {
      try {
        TrainingMetricMessage metric = objectMapper.treeToValue(node, TrainingMetricMessage.class);
        job.setLatestMetric(metric);
        if (metric.epoch() >= metric.totalEpochs()) {
          job.setStatus("completed");
        } else if (!"paused".equals(job.status())) {
          job.setStatus("running");
        }
        broadcast(job.jobId(), metric);
      } catch (JsonProcessingException ignored) {
        // Ignore malformed metric rows from the worker.
      }
    } else if ("control".equals(type)) {
      String status = node.path("status").asText(job.status());
      job.setStatus(status);
      broadcastRaw(job.jobId(), line);
    } else if ("test_result".equals(type)) {
      broadcastRaw(job.jobId(), line);
    } else if ("error".equals(type)) {
      job.setStatus("stopped");
      broadcastRaw(job.jobId(), line);
    }
  }

  private void waitForWorkerExit(TrainingJob job) {
    Process process = job.process();
    if (process == null) {
      return;
    }
    try {
      int exitCode = process.waitFor();
      if (exitCode == 0 && "running".equals(job.status())) {
        job.setStatus("completed");
      } else if (exitCode != 0 && !"stopped".equals(job.status())) {
        job.setStatus("stopped");
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      if (!"stopped".equals(job.status())) {
        job.setStatus("stopped");
      }
    }
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

  private void broadcastRaw(String jobId, String payload) {
    CopyOnWriteArraySet<WebSocketSession> jobSessions = sessions.get(jobId);
    if (jobSessions == null || jobSessions.isEmpty()) {
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

  private void writeControl(TrainingJob job, String command) {
    try {
      Files.writeString(job.controlFile(), objectMapper.writeValueAsString(Map.of("command", command)), StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to update training control file.");
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

  private final class TrainingJob {
    private final String jobId;
    private final StartTrainingRequest request;
    private final int totalEpochs;
    private final int totalBatches;
    private final Path directory;
    private final Path requestFile;
    private final Path controlFile;
    private final Instant startedAt = Instant.now();
    private volatile String status = "running";
    private volatile TrainingMetricMessage latestMetric;
    private volatile Process process;

    private TrainingJob(String jobId, StartTrainingRequest request, int totalEpochs, int totalBatches) {
      this.jobId = jobId;
      this.request = request;
      this.totalEpochs = totalEpochs;
      this.totalBatches = totalBatches;
      this.directory = jobRoot.resolve(jobId).normalize();
      this.requestFile = directory.resolve("request.json");
      this.controlFile = directory.resolve("control.json");
    }

    private void writeRequest() throws IOException {
      Map<String, Object> payload = Map.of(
          "jobId", jobId,
          "datasetRoot", datasetRoot.toString(),
          "controlFile", controlFile.toString(),
          "datasetId", request.datasetId(),
          "split", request.split(),
          "layers", request.layers() == null ? List.of() : request.layers(),
          "connections", request.connections() == null ? List.of() : request.connections(),
          "config", request.config()
      );
      Files.writeString(requestFile, objectMapper.writeValueAsString(payload), StandardCharsets.UTF_8);
    }

    private TrainingStatusResponse toStatus() {
      TrainingMetricMessage metric = latestMetric;
      return new TrainingStatusResponse(
          jobId,
          status,
          metric == null ? 0 : metric.epoch(),
          metric == null ? 0 : metric.batch(),
          totalEpochs,
          metric == null ? totalBatches : metric.totalBatches(),
          metric == null ? 1.7 : metric.loss(),
          metric == null ? 1.78 : metric.valLoss(),
          metric == null ? 0.2 : metric.accuracy(),
          metric == null ? 0.18 : metric.valAccuracy(),
          metric == null ? Math.max(0, java.time.Duration.between(startedAt, Instant.now()).toSeconds()) : metric.elapsedSeconds(),
          metric == null ? 0 : metric.etaSeconds()
      );
    }

    private void reset() {
      latestMetric = null;
      status = "running";
    }

    private void destroyProcess() {
      Process current = process;
      if (current != null && current.isAlive()) {
        current.destroy();
        try {
          if (!current.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) {
            current.destroyForcibly();
          }
        } catch (InterruptedException ex) {
          Thread.currentThread().interrupt();
          current.destroyForcibly();
        }
      }
      process = null;
    }

    private String jobId() {
      return jobId;
    }

    private String status() {
      return status;
    }

    private void setStatus(String status) {
      this.status = status;
    }

    private int epoch() {
      return latestMetric == null ? 0 : latestMetric.epoch();
    }

    private TrainingMetricMessage latestMetric() {
      return latestMetric;
    }

    private void setLatestMetric(TrainingMetricMessage latestMetric) {
      this.latestMetric = latestMetric;
    }

    private Path directory() {
      return directory;
    }

    private Path requestFile() {
      return requestFile;
    }

    private Path controlFile() {
      return controlFile;
    }

    private Process process() {
      return process;
    }

    private void setProcess(Process process) {
      this.process = process;
    }
  }
}
