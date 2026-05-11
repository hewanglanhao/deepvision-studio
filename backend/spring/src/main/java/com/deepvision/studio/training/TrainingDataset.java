package com.deepvision.studio.training;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(
    name = "training_datasets",
    indexes = {
        @Index(name = "idx_training_datasets_source", columnList = "source"),
        @Index(name = "idx_training_datasets_kind", columnList = "kind")
    }
)
public class TrainingDataset {
  @Id
  @Column(length = 120)
  private String id;

  @Column(nullable = false, length = 160)
  private String name;

  @Column(nullable = false, length = 32)
  private String source;

  @Column(nullable = false, length = 32)
  private String kind;

  @Column(nullable = false, length = 1000)
  private String description;

  @Column(nullable = false)
  private int sampleCount;

  @Column(nullable = false)
  private int classCount;

  @Column(nullable = false, length = 160)
  private String inputShape;

  @Column(nullable = false, length = 80)
  private String recommendedSplit;

  @Lob
  @Column(nullable = false)
  private String labelsJson;

  @Column(nullable = false)
  private boolean hasLabels;

  @Column(nullable = false)
  private double trainRatio;

  @Column(nullable = false)
  private double valRatio;

  @Column(nullable = false)
  private double testRatio;

  @Lob
  @Column(nullable = false)
  private String labelDistributionJson;

  @Lob
  @Column(nullable = false)
  private String imagePreviewJson;

  @Lob
  @Column(nullable = false)
  private String tablePreviewJson;

  @Lob
  @Column(nullable = false)
  private String pointPreviewJson;

  @Lob
  @Column(nullable = false)
  private String warningsJson;

  @Column(name = "created_at", nullable = false)
  private Instant createdAt = Instant.now();

  @Column(name = "updated_at", nullable = false)
  private Instant updatedAt = Instant.now();

  protected TrainingDataset() {}

  public TrainingDataset(
      String id,
      String name,
      String source,
      String kind,
      String description,
      int sampleCount,
      int classCount,
      String inputShape,
      String recommendedSplit,
      String labelsJson,
      boolean hasLabels,
      double trainRatio,
      double valRatio,
      double testRatio,
      String labelDistributionJson,
      String imagePreviewJson,
      String tablePreviewJson,
      String pointPreviewJson,
      String warningsJson
  ) {
    this.id = id;
    this.name = name;
    this.source = source;
    this.kind = kind;
    this.description = description;
    this.sampleCount = sampleCount;
    this.classCount = classCount;
    this.inputShape = inputShape;
    this.recommendedSplit = recommendedSplit;
    this.labelsJson = labelsJson;
    this.hasLabels = hasLabels;
    this.trainRatio = trainRatio;
    this.valRatio = valRatio;
    this.testRatio = testRatio;
    this.labelDistributionJson = labelDistributionJson;
    this.imagePreviewJson = imagePreviewJson;
    this.tablePreviewJson = tablePreviewJson;
    this.pointPreviewJson = pointPreviewJson;
    this.warningsJson = warningsJson;
  }

  public String getId() { return id; }
  public String getName() { return name; }
  public String getSource() { return source; }
  public String getKind() { return kind; }
  public String getDescription() { return description; }
  public int getSampleCount() { return sampleCount; }
  public int getClassCount() { return classCount; }
  public String getInputShape() { return inputShape; }
  public String getRecommendedSplit() { return recommendedSplit; }
  public String getLabelsJson() { return labelsJson; }
  public boolean isHasLabels() { return hasLabels; }
  public double getTrainRatio() { return trainRatio; }
  public double getValRatio() { return valRatio; }
  public double getTestRatio() { return testRatio; }
  public String getLabelDistributionJson() { return labelDistributionJson; }
  public String getImagePreviewJson() { return imagePreviewJson; }
  public String getTablePreviewJson() { return tablePreviewJson; }
  public String getPointPreviewJson() { return pointPreviewJson; }
  public String getWarningsJson() { return warningsJson; }
  public Instant getCreatedAt() { return createdAt; }
  public Instant getUpdatedAt() { return updatedAt; }
}
