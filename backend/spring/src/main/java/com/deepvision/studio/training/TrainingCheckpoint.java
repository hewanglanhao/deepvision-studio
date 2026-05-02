package com.deepvision.studio.training;

import com.deepvision.studio.auth.AppUser;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(
    name = "training_checkpoints",
    indexes = @Index(name = "idx_training_checkpoints_user_created", columnList = "user_id,created_at")
)
public class TrainingCheckpoint {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "user_id", nullable = false)
  private AppUser user;

  @Column(nullable = false, length = 160)
  private String name;

  @Column(nullable = false, length = 120)
  private String jobId;

  @Column(nullable = false, length = 120)
  private String datasetId;

  @Column(nullable = false, length = 160)
  private String datasetName;

  @Column(nullable = false, length = 80)
  private String modelSignature;

  @Column(nullable = false, length = 500)
  private String checkpointPath;

  @Lob
  @Column(nullable = false)
  private String layersJson;

  @Lob
  @Column(nullable = false)
  private String configJson;

  @Lob
  @Column(nullable = false)
  private String splitJson;

  @Lob
  @Column(nullable = false)
  private String testResultJson;

  @Column(nullable = false)
  private int epoch;

  @Column(nullable = false)
  private int totalEpochs;

  private Double testLoss;

  private Double testAccuracy;

  @Column(nullable = false)
  private int testSampleCount;

  @Column(name = "created_at", nullable = false)
  private Instant createdAt = Instant.now();

  protected TrainingCheckpoint() {}

  public TrainingCheckpoint(
      AppUser user,
      String name,
      String jobId,
      String datasetId,
      String datasetName,
      String modelSignature,
      String checkpointPath,
      String layersJson,
      String configJson,
      String splitJson,
      String testResultJson,
      int epoch,
      int totalEpochs,
      Double testLoss,
      Double testAccuracy,
      int testSampleCount
  ) {
    this.user = user;
    this.name = name;
    this.jobId = jobId;
    this.datasetId = datasetId;
    this.datasetName = datasetName;
    this.modelSignature = modelSignature;
    this.checkpointPath = checkpointPath;
    this.layersJson = layersJson;
    this.configJson = configJson;
    this.splitJson = splitJson;
    this.testResultJson = testResultJson;
    this.epoch = epoch;
    this.totalEpochs = totalEpochs;
    this.testLoss = testLoss;
    this.testAccuracy = testAccuracy;
    this.testSampleCount = testSampleCount;
  }

  public Long getId() { return id; }
  public AppUser getUser() { return user; }
  public String getName() { return name; }
  public String getJobId() { return jobId; }
  public String getDatasetId() { return datasetId; }
  public String getDatasetName() { return datasetName; }
  public String getModelSignature() { return modelSignature; }
  public String getCheckpointPath() { return checkpointPath; }
  public String getLayersJson() { return layersJson; }
  public String getConfigJson() { return configJson; }
  public String getSplitJson() { return splitJson; }
  public String getTestResultJson() { return testResultJson; }
  public int getEpoch() { return epoch; }
  public int getTotalEpochs() { return totalEpochs; }
  public Double getTestLoss() { return testLoss; }
  public Double getTestAccuracy() { return testAccuracy; }
  public int getTestSampleCount() { return testSampleCount; }
  public Instant getCreatedAt() { return createdAt; }
}
