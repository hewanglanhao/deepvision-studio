package com.deepvision.studio.training;

import com.fasterxml.jackson.databind.JsonNode;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import java.time.Instant;
import java.util.List;

public final class TrainingDtos {
  private TrainingDtos() {}

  @Schema(description = "Training dataset list item")
  public record TrainingDatasetOption(
      String id,
      String name,
      String source,
      String kind,
      String description,
      int sampleCount,
      int classCount,
      String inputShape,
      String recommendedSplit,
      List<String> labels
  ) {}

  @Schema(description = "Dataset label distribution item")
  public record LabelDistributionItem(String label, int count, String color) {}

  @Schema(description = "Image preview item for image datasets")
  public record ImagePreviewItem(String name, String label, String url) {}

  @Schema(description = "Table preview for tabular datasets")
  public record TablePreview(List<String> headers, List<List<String>> rows) {}

  @Schema(description = "2D point preview item")
  public record PointPreviewItem(double x, double y, String label, String color) {}

  @Schema(description = "Detailed training dataset metadata")
  public record TrainingDatasetDetail(
      String id,
      String name,
      String source,
      String kind,
      String description,
      int sampleCount,
      int classCount,
      String inputShape,
      String recommendedSplit,
      List<String> labels,
      boolean hasLabels,
      double trainRatio,
      double valRatio,
      double testRatio,
      List<LabelDistributionItem> labelDistribution,
      List<ImagePreviewItem> imagePreview,
      TablePreview tablePreview,
      List<PointPreviewItem> pointPreview,
      List<String> warnings
  ) {
    TrainingDatasetOption toOption() {
      return new TrainingDatasetOption(
          id, name, source, kind, description, sampleCount, classCount, inputShape, recommendedSplit, labels
      );
    }
  }

  @Schema(description = "Dataset import result")
  public record DatasetImportResponse(String datasetId, TrainingDatasetDetail detail) {}

  @Schema(description = "Dataset error response")
  public record DatasetErrorResponse(String error, String message) {}

  @Schema(description = "Train/validation/test split ratios")
  public record SplitRequest(double train, double val, double test) {}

  @Schema(description = "Training hyperparameter configuration")
  public record TrainingConfigRequest(
      @Schema(description = "Mini-batch size", example = "32")
      @Positive(message = "batchSize must be positive.")
      Integer batchSize,
      @Schema(description = "Total training epochs", example = "10")
      @Positive(message = "totalEpochs must be positive.")
      Integer totalEpochs,
      @Schema(description = "Learning rate", example = "0.001")
      Double learningRate,
      String optimizer,
      String scheduler,
      Double lrDecay,
      String lossFunction
  ) {}

  @Schema(description = "Request to start a training job")
  public record StartTrainingRequest(
      @Schema(description = "Dataset id", example = "builtin-moons")
      @NotBlank(message = "datasetId is required.")
      String datasetId,
      @NotNull(message = "split is required.")
      @Valid
      SplitRequest split,
      List<JsonNode> layers,
      List<JsonNode> connections,
      @NotNull(message = "config is required.")
      @Valid
      TrainingConfigRequest config
  ) {}

  @Schema(description = "Training job start response")
  public record TrainingStartResponse(
      String jobId,
      String status,
      int totalEpochs,
      int totalBatches,
      String streamUrl
  ) {}

  @Schema(description = "Training metric stream message")
  public record TrainingMetricMessage(
      String type,
      String jobId,
      int step,
      int epoch,
      int batch,
      int totalEpochs,
      int totalBatches,
      double loss,
      Double valLoss,
      double accuracy,
      Double valAccuracy,
      double lr,
      long elapsedSeconds,
      long etaSeconds,
      double gradientNorm,
      double weightMean,
      double weightStd,
      String gradientStatus
  ) {}

  @Schema(description = "Training job status response")
  public record TrainingStatusResponse(
      String jobId,
      String status,
      int epoch,
      int batch,
      int totalEpochs,
      int totalBatches,
      double latestLoss,
      Double latestValLoss,
      double latestAccuracy,
      Double latestValAccuracy,
      long elapsedSeconds,
      long etaSeconds
  ) {}

  @Schema(description = "Training job control response")
  public record TrainingControlResponse(String jobId, String status, String message) {}

  @Schema(description = "Weight histogram bin")
  public record HistogramBin(String label, int count) {}

  @Schema(description = "Weight histogram response")
  public record WeightHistogramResponse(String jobId, int epoch, List<HistogramBin> bins) {}

  @Schema(description = "Checkpoint test prediction sample")
  public record TrainingPredictionSample(
      int index,
      int trueIndex,
      int predictedIndex,
      String trueLabel,
      String predictedLabel,
      double confidence,
      boolean correct,
      String name,
      String imageUrl
  ) {}

  @Schema(description = "Checkpoint test evaluation result")
  public record CheckpointTestResult(
      String type,
      String jobId,
      Double testLoss,
      Double testAccuracy,
      int sampleCount,
      List<TrainingPredictionSample> samples
  ) {}

  @Schema(description = "Training collaboration room summary")
  public record CollaborationRoomSummary(
      String jobId,
      int onlineCount,
      Instant createdAt,
      List<String> users
  ) {}

  @Schema(description = "Saved training checkpoint summary")
  public record TrainingCheckpointSummary(
      Long id,
      String name,
      String jobId,
      String datasetId,
      String datasetName,
      String modelSignature,
      String networkDescription,
      List<String> layerSummary,
      JsonNode layers,
      JsonNode config,
      JsonNode split,
      JsonNode testResult,
      JsonNode metricHistory,
      String status,
      int epoch,
      int totalEpochs,
      Double trainLoss,
      Double trainAccuracy,
      Double valLoss,
      Double valAccuracy,
      Double testLoss,
      Double testAccuracy,
      int testSampleCount,
      Instant createdAt
  ) {}

  @Schema(description = "Request to test a checkpoint on a dataset")
  public record TestCheckpointRequest(
      @NotBlank(message = "datasetId is required.")
      String datasetId,
      List<JsonNode> layers
  ) {}

  @Schema(description = "Sample item available for single checkpoint inference")
  public record InferenceSampleItem(
      int index,
      int trueIndex,
      String trueLabel,
      String name,
      String imageUrl,
      List<Integer> shape,
      List<Double> featurePreview,
      List<String> featureNames,
      Integer featureCount,
      List<String> rawHeaders,
      List<String> rawValues,
      JsonNode rawPreview
  ) {}

  @Schema(description = "Samples available for checkpoint inference")
  public record InferenceSampleListResponse(
      String type,
      String datasetId,
      int sampleCount,
      List<InferenceSampleItem> samples
  ) {}

  @Schema(description = "Request to run single-sample checkpoint inference")
  public record SingleInferenceRequest(
      int sampleIndex
  ) {}

  @Schema(description = "Single-sample checkpoint inference result")
  public record SingleInferenceResult(
      String type,
      String jobId,
      String datasetId,
      JsonNode sample,
      JsonNode prediction,
      JsonNode activations
  ) {}
}
