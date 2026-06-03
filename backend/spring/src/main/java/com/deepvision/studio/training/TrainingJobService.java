package com.deepvision.studio.training;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.training.TrainingDtos.CheckpointTestResult;
import com.deepvision.studio.training.TrainingDtos.HistogramBin;
import com.deepvision.studio.training.TrainingDtos.SplitRequest;
import com.deepvision.studio.training.TrainingDtos.StartTrainingRequest;
import com.deepvision.studio.training.TrainingDtos.TestCheckpointRequest;
import com.deepvision.studio.training.TrainingDtos.TrainingCheckpointSummary;
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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

@Service
public class TrainingJobService {
  private static final DateTimeFormatter JOB_ID_TIME = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");
  private static final int MAX_RECENT_TRAINING_EVENTS = 240;
  private static final int MAX_CHECKPOINT_METRIC_HISTORY = 2000;

  private final TrainingDatasetService datasetService;
  private final AppUserRepository users;
  private final TrainingCheckpointRepository checkpoints;
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
      AppUserRepository users,
      TrainingCheckpointRepository checkpoints,
      ObjectMapper objectMapper,
      @Value("${deepvision.training.stream-base-url:ws://127.0.0.1:8080}") String streamBaseUrl,
      @Value("${deepvision.datasets.root}") String datasetRoot,
      @Value("${deepvision.training.jobs-root:./training-jobs}") String jobRoot,
      @Value("${deepvision.training.python-executable:python}") String pythonExecutable,
      @Value("${deepvision.training.worker-script:../python-training/training_worker.py}") String workerScript
  ) {
    this.datasetService = datasetService;
    this.users = users;
    this.checkpoints = checkpoints;
    this.objectMapper = objectMapper;
    this.streamBaseUrl = trimTrailingSlash(streamBaseUrl);
    this.datasetRoot = Path.of(datasetRoot).toAbsolutePath().normalize();
    this.jobRoot = Path.of(jobRoot).toAbsolutePath().normalize();
    this.pythonExecutable = pythonExecutable;
    this.workerScript = Path.of(workerScript).toAbsolutePath().normalize();
  }

  public TrainingStartResponse start(StartTrainingRequest request, String username) {
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
    TrainingJob job = new TrainingJob(jobId, request, username, modelSignature(request), totalEpochs, totalBatches);
    jobs.put(jobId, job);
    startPythonWorker(job);
    return new TrainingStartResponse(
        jobId,
        job.status(),
        totalEpochs,
        totalBatches,
        streamUrl(jobId)
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
    if (job.username() == null || job.username().isBlank()) {
      throw new IllegalArgumentException("Please login before saving checkpoints.");
    }
    if (job.testResult() == null) {
      throw new IllegalArgumentException("Checkpoint can be saved after test set evaluation completes.");
    }
    TrainingCheckpoint checkpoint = saveCheckpoint(job, job.testResult());
    return new TrainingControlResponse(jobId, job.status(), "Checkpoint saved: " + checkpoint.getName());
  }

  public List<TrainingCheckpointSummary> listCheckpoints(String username, String datasetId) {
    requireUser(username);
    List<TrainingCheckpoint> rows = datasetId == null || datasetId.isBlank()
        ? checkpoints.findByUserUsernameOrderByCreatedAtDesc(username)
        : checkpoints.findByUserUsernameAndDatasetIdOrderByCreatedAtDesc(username, datasetId);
    return rows.stream()
        .map(this::toCheckpointSummary)
        .toList();
  }

  public CheckpointTestResult testCheckpoint(String username, Long checkpointId, TestCheckpointRequest request) {
    requireUser(username);
    TrainingCheckpoint checkpoint = checkpoints.findByIdAndUserUsername(checkpointId, username)
        .orElseThrow(() -> new IllegalArgumentException("Checkpoint not found."));
    return runCheckpointTest(checkpoint);
  }

  public void addSession(String jobId, WebSocketSession session) {
    TrainingJob job = getJob(jobId);
    sessions.computeIfAbsent(jobId, ignored -> new CopyOnWriteArraySet<>()).add(session);
    List<String> recentEvents = job.recentStreamEvents();
    if (!recentEvents.isEmpty()) {
      for (String event : recentEvents) {
        sendRaw(session, event);
      }
      return;
    }
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
        job.addMetricHistory(metric);
        job.addStreamEvent(line);
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
      job.addStreamEvent(line);
      broadcastRaw(job.jobId(), line);
    } else if ("test_result".equals(type)) {
      job.setTestResult(node);
      job.addStreamEvent(line);
      if (!job.checkpointSaved() && job.username() != null && !job.username().isBlank()) {
        try {
          saveCheckpoint(job, node);
          job.setCheckpointSaved(true);
        } catch (RuntimeException ignored) {
          // Keep streaming the test result even if persistence fails.
        }
      }
      broadcastRaw(job.jobId(), line);
    } else if ("backprop".equals(type)) {
      job.addStreamEvent(line);
      broadcastRaw(job.jobId(), line);
    } else if ("error".equals(type)) {
      job.setStatus("stopped");
      job.addStreamEvent(line);
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

  private void sendRaw(WebSocketSession session, String payload) {
    if (!session.isOpen()) {
      return;
    }
    try {
      session.sendMessage(new TextMessage(payload));
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

  private void requireUser(String username) {
    if (username == null || username.isBlank()) {
      throw new IllegalArgumentException("Please login first.");
    }
  }

  private TrainingCheckpoint saveCheckpoint(TrainingJob job, JsonNode testResult) {
    AppUser user = users.findByUsername(job.username())
        .orElseThrow(() -> new IllegalArgumentException("User not found."));
    if (!Files.exists(job.checkpointFile())) {
      throw new IllegalArgumentException("Checkpoint file is not available yet.");
    }
    TrainingDatasetDetail dataset = datasetService.getDetail(job.request().datasetId());
    try {
      String layersJson = objectMapper.writeValueAsString(job.request().layers() == null ? List.of() : job.request().layers());
      String configJson = objectMapper.writeValueAsString(job.request().config());
      String splitJson = objectMapper.writeValueAsString(job.request().split());
      String testResultJson = objectMapper.writeValueAsString(testResult);
      String metricHistoryJson = objectMapper.writeValueAsString(job.metricHistory());
      TrainingMetricMessage metric = job.latestMetric();
      TrainingCheckpoint checkpoint = new TrainingCheckpoint(
          user,
          dataset.name() + " · " + job.jobId(),
          job.jobId(),
          job.request().datasetId(),
          dataset.name(),
          job.modelSignature(),
          job.checkpointFile().toString(),
          layersJson,
          configJson,
          splitJson,
          testResultJson,
          metricHistoryJson,
          describeNetwork(job.request().layers()),
          job.status(),
          job.epoch(),
          job.totalEpochs(),
          metric == null ? null : metric.loss(),
          metric == null ? null : metric.accuracy(),
          metric == null ? null : metric.valLoss(),
          metric == null ? null : metric.valAccuracy(),
          testResult.path("testLoss").isNull() || testResult.path("testLoss").isMissingNode() ? null : testResult.path("testLoss").asDouble(),
          testResult.path("testAccuracy").isNull() || testResult.path("testAccuracy").isMissingNode() ? null : testResult.path("testAccuracy").asDouble(),
          testResult.path("sampleCount").asInt(0)
      );
      return checkpoints.save(checkpoint);
    } catch (JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to serialize checkpoint metadata.");
    }
  }

  private TrainingCheckpointSummary toCheckpointSummary(TrainingCheckpoint checkpoint) {
    JsonNode layers = readCheckpointJson(checkpoint.getLayersJson(), objectMapper.createArrayNode());
    JsonNode config = readCheckpointJson(checkpoint.getConfigJson(), objectMapper.createObjectNode());
    JsonNode split = readCheckpointJson(checkpoint.getSplitJson(), objectMapper.createObjectNode());
    JsonNode testResult = readCheckpointJson(checkpoint.getTestResultJson(), objectMapper.createObjectNode());
    JsonNode metricHistory = readCheckpointJson(checkpoint.getMetricHistoryJson(), objectMapper.createArrayNode());
    List<String> layerSummary = summarizeLayers(layers);
    String description = checkpoint.getNetworkDescription();
    if (description == null || description.isBlank()) {
      description = String.join(" -> ", layerSummary);
    }
    return new TrainingCheckpointSummary(
        checkpoint.getId(),
        checkpoint.getName(),
        checkpoint.getJobId(),
        checkpoint.getDatasetId(),
        checkpoint.getDatasetName(),
        checkpoint.getModelSignature(),
        description,
        layerSummary,
        layers,
        config,
        split,
        testResult,
        metricHistory,
        checkpoint.getStatus(),
        checkpoint.getEpoch(),
        checkpoint.getTotalEpochs(),
        checkpoint.getTrainLoss(),
        checkpoint.getTrainAccuracy(),
        checkpoint.getValLoss(),
        checkpoint.getValAccuracy(),
        checkpoint.getTestLoss(),
        checkpoint.getTestAccuracy(),
        checkpoint.getTestSampleCount(),
        checkpoint.getCreatedAt()
    );
  }

  private JsonNode readCheckpointJson(String raw, JsonNode fallback) {
    if (raw == null || raw.isBlank()) {
      return fallback;
    }
    try {
      return objectMapper.readTree(raw);
    } catch (JsonProcessingException ex) {
      return fallback;
    }
  }

  private String describeNetwork(List<JsonNode> layers) {
    List<String> parts = summarizeLayers(layers == null ? objectMapper.createArrayNode() : objectMapper.valueToTree(layers));
    return String.join(" -> ", parts);
  }

  private List<String> summarizeLayers(JsonNode layers) {
    List<String> summary = new ArrayList<>();
    if (layers == null || !layers.isArray()) {
      return summary;
    }
    for (JsonNode layer : layers) {
      if (layer.path("enabled").isBoolean() && !layer.path("enabled").asBoolean()) {
        continue;
      }
      summary.add(summarizeLayer(layer));
    }
    return summary;
  }

  private String summarizeLayer(JsonNode layer) {
    String type = layer.path("type").asText("layer");
    JsonNode params = layer.path("params");
    return switch (type) {
      case "input" -> {
        String inputKind = params.path("inputKind").asText("image");
        if ("table".equals(inputKind)) {
          yield "CSV输入(" + params.path("featureCount").asInt(0) + "维)";
        }
        yield "图像输入(" + params.path("width").asInt(0) + "x" + params.path("height").asInt(0) + "x" + params.path("channels").asInt(0) + ")";
      }
      case "conv2d" -> "卷积(" + params.path("outChannels").asInt(0) + "通道,k" + params.path("kernelSize").asInt(0) + ",s" + params.path("stride").asInt(1) + ")";
      case "residual" -> "残差块(" + params.path("outChannels").asInt(0) + "通道,k" + params.path("kernelSize").asInt(0) + ",投影" + (params.path("useProjection").asBoolean(false) ? "开" : "关") + ")";
      case "pool2d" -> "池化(" + params.path("mode").asText("max") + ",k" + params.path("kernelSize").asInt(0) + ")";
      case "flatten" -> "展平";
      case "dense" -> "全连接(" + params.path("units").asInt(0) + "," + params.path("activation").asText("none") + ")";
      case "activation" -> "激活(" + params.path("activationType").asText("none") + ")";
      case "dropout" -> "Dropout(" + String.format(java.util.Locale.US, "%.0f%%", params.path("rate").asDouble(0) * 100) + ")";
      case "output" -> "输出(" + params.path("units").asInt(0) + "类," + params.path("activation").asText("softmax") + ")";
      default -> layer.path("name").asText(type);
    };
  }

  private CheckpointTestResult runCheckpointTest(TrainingCheckpoint checkpoint) {
    String testId = "checkpoint-test-" + checkpoint.getId() + "-" + UUID.randomUUID();
    Path testDir = jobRoot.resolve("checkpoint-tests").resolve(testId).normalize();
    Path requestFile = testDir.resolve("request.json");
    try {
      Files.createDirectories(testDir);
      JsonNode splitNode = objectMapper.readTree(checkpoint.getSplitJson());
      JsonNode layersNode = objectMapper.readTree(checkpoint.getLayersJson());
      Map<String, Object> payload = Map.of(
          "action", "test_checkpoint",
          "jobId", testId,
          "datasetRoot", datasetRoot.toString(),
          "datasetId", checkpoint.getDatasetId(),
          "split", splitNode,
          "layers", layersNode,
          "checkpointFile", checkpoint.getCheckpointPath()
      );
      Files.writeString(requestFile, objectMapper.writeValueAsString(payload), StandardCharsets.UTF_8);
      Process process = new ProcessBuilder(pythonExecutable, workerScript.toString(), "--request", requestFile.toString())
          .redirectErrorStream(true)
          .directory(jobRoot.toFile())
          .start();
      JsonNode result = null;
      StringBuilder diagnostics = new StringBuilder();
      try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
          if (line.isBlank()) {
            continue;
          }
          if (!line.startsWith("{")) {
            if (diagnostics.length() < 2000) {
              diagnostics.append(line).append(System.lineSeparator());
            }
            continue;
          }
          JsonNode node = objectMapper.readTree(line);
          String type = node.path("type").asText("");
          if ("error".equals(type)) {
            throw new IllegalArgumentException(node.path("message").asText("Checkpoint test failed."));
          }
          if ("test_result".equals(type)) {
            result = node;
          }
        }
      }
      if (!process.waitFor(10, TimeUnit.MINUTES)) {
        process.destroyForcibly();
        throw new IllegalArgumentException("Checkpoint test timed out.");
      }
      if (process.exitValue() != 0) {
        String detail = diagnostics.isEmpty() ? "" : " " + diagnostics.toString().trim();
        throw new IllegalArgumentException("Checkpoint test process failed." + detail);
      }
      if (result == null) {
        String detail = diagnostics.isEmpty() ? "" : " " + diagnostics.toString().trim();
        throw new IllegalArgumentException("Checkpoint test returned no result." + detail);
      }
      return objectMapper.treeToValue(result, CheckpointTestResult.class);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to run checkpoint test: " + ex.getMessage());
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalArgumentException("Checkpoint test interrupted.");
    }
  }

  private String modelSignature(StartTrainingRequest request) {
    return modelSignature(request.datasetId(), request.layers());
  }

  private String modelSignature(String datasetId, List<JsonNode> layers) {
    ObjectNode root = objectMapper.createObjectNode();
    root.put("datasetId", datasetId);
    ArrayNode normalizedLayers = root.putArray("layers");
    for (JsonNode layer : layers == null ? List.<JsonNode>of() : layers) {
      ObjectNode normalized = objectMapper.createObjectNode();
      normalized.set("type", layer.path("type"));
      normalized.set("enabled", layer.has("enabled") ? layer.path("enabled") : objectMapper.getNodeFactory().booleanNode(true));
      normalized.set("params", layer.path("params"));
      normalizedLayers.add(normalized);
    }
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      byte[] hash = digest.digest(objectMapper.writeValueAsBytes(root));
      return HexFormat.of().formatHex(hash);
    } catch (NoSuchAlgorithmException | JsonProcessingException ex) {
      throw new IllegalArgumentException("Failed to compute model signature.");
    }
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
      return "";
    }
    return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
  }

  private String streamUrl(String jobId) {
    String path = "/api/training/stream?jobId=" + jobId;
    return streamBaseUrl.isBlank() ? path : streamBaseUrl + path;
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
    private final String username;
    private final String modelSignature;
    private final int totalEpochs;
    private final int totalBatches;
    private final Path directory;
    private final Path requestFile;
    private final Path controlFile;
    private final Path checkpointFile;
    private final Instant startedAt = Instant.now();
    private volatile String status = "running";
    private volatile TrainingMetricMessage latestMetric;
    private volatile JsonNode testResult;
    private volatile boolean checkpointSaved;
    private volatile Process process;
    private final ArrayDeque<String> recentStreamEvents = new ArrayDeque<>();
    private final ArrayDeque<TrainingMetricMessage> metricHistory = new ArrayDeque<>();

    private TrainingJob(String jobId, StartTrainingRequest request, String username, String modelSignature, int totalEpochs, int totalBatches) {
      this.jobId = jobId;
      this.request = request;
      this.username = username;
      this.modelSignature = modelSignature;
      this.totalEpochs = totalEpochs;
      this.totalBatches = totalBatches;
      this.directory = jobRoot.resolve(jobId).normalize();
      this.requestFile = directory.resolve("request.json");
      this.controlFile = directory.resolve("control.json");
      this.checkpointFile = directory.resolve("checkpoint.pt");
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
          "config", request.config(),
          "checkpointFile", checkpointFile.toString(),
          "modelSignature", modelSignature
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
      testResult = null;
      checkpointSaved = false;
      synchronized (recentStreamEvents) {
        recentStreamEvents.clear();
      }
      synchronized (metricHistory) {
        metricHistory.clear();
      }
      status = "running";
    }

    private void addStreamEvent(String payload) {
      synchronized (recentStreamEvents) {
        recentStreamEvents.addLast(payload);
        while (recentStreamEvents.size() > MAX_RECENT_TRAINING_EVENTS) {
          recentStreamEvents.removeFirst();
        }
      }
    }

    private List<String> recentStreamEvents() {
      synchronized (recentStreamEvents) {
        return new ArrayList<>(recentStreamEvents);
      }
    }

    private void addMetricHistory(TrainingMetricMessage metric) {
      synchronized (metricHistory) {
        metricHistory.addLast(metric);
        while (metricHistory.size() > MAX_CHECKPOINT_METRIC_HISTORY) {
          metricHistory.removeFirst();
        }
      }
    }

    private List<TrainingMetricMessage> metricHistory() {
      synchronized (metricHistory) {
        return new ArrayList<>(metricHistory);
      }
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

    private StartTrainingRequest request() {
      return request;
    }

    private String username() {
      return username;
    }

    private String modelSignature() {
      return modelSignature;
    }

    private int totalEpochs() {
      return totalEpochs;
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

    private JsonNode testResult() {
      return testResult;
    }

    private void setTestResult(JsonNode testResult) {
      this.testResult = testResult;
    }

    private boolean checkpointSaved() {
      return checkpointSaved;
    }

    private void setCheckpointSaved(boolean checkpointSaved) {
      this.checkpointSaved = checkpointSaved;
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

    private Path checkpointFile() {
      return checkpointFile;
    }

    private Process process() {
      return process;
    }

    private void setProcess(Process process) {
      this.process = process;
    }
  }
}
