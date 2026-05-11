package com.deepvision.studio.training;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import java.time.Instant;
import java.util.List;

public final class TrainingDtos {
  private TrainingDtos() {}

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

  public record LabelDistributionItem(String label, int count, String color) {}

  public record ImagePreviewItem(String name, String label, String url) {}

  public record TablePreview(List<String> headers, List<List<String>> rows) {}

  public record PointPreviewItem(double x, double y, String label, String color) {}

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

  public record DatasetImportResponse(String datasetId, TrainingDatasetDetail detail) {}

  public record DatasetErrorResponse(String error, String message) {}

  public record SplitRequest(double train, double val, double test) {}

  public record TrainingConfigRequest(
      @Positive(message = "batchSize must be positive.")
      Integer batchSize,
      @Positive(message = "totalEpochs must be positive.")
      Integer totalEpochs,
      Double learningRate,
      String optimizer,
      String scheduler,
      Double lrDecay,
      String lossFunction
  ) {}

  public record StartTrainingRequest(
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

  public record TrainingStartResponse(
      String jobId,
      String status,
      int totalEpochs,
      int totalBatches,
      String streamUrl
  ) {}

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

  public record TrainingControlResponse(String jobId, String status, String message) {}

  public record HistogramBin(String label, int count) {}

  public record WeightHistogramResponse(String jobId, int epoch, List<HistogramBin> bins) {}

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

  public record CheckpointTestResult(
      String type,
      String jobId,
      Double testLoss,
      Double testAccuracy,
      int sampleCount,
      List<TrainingPredictionSample> samples
  ) {}

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

  public record TestCheckpointRequest(
      @NotBlank(message = "datasetId is required.")
      String datasetId,
      List<JsonNode> layers
  ) {}
}
