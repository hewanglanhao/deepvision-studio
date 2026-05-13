package com.deepvision.studio.forward;

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
    name = "forward_records",
    indexes = @Index(name = "idx_forward_records_user_created", columnList = "user_id,created_at")
)
public class ForwardRecord {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "user_id", nullable = false)
  private AppUser user;

  @Column(nullable = false, length = 120)
  private String name;

  @Column(nullable = false, length = 80)
  private String templateId;

  @Column(nullable = false, length = 80)
  private String datasetName;

  @Column(nullable = false)
  private int layerCount;

  @Column(nullable = false)
  private long parameterCount;

  @Column(length = 300)
  private String imagePath;

  @Lob
  @Column(nullable = false)
  private String snapshotJson;

  @Column(name = "created_at", nullable = false)
  private Instant createdAt = Instant.now();

  protected ForwardRecord() {}

  public ForwardRecord(
      AppUser user,
      String name,
      String templateId,
      String datasetName,
      int layerCount,
      long parameterCount,
      String imagePath,
      String snapshotJson
  ) {
    this.user = user;
    this.name = name;
    this.templateId = templateId;
    this.datasetName = datasetName;
    this.layerCount = layerCount;
    this.parameterCount = parameterCount;
    this.imagePath = imagePath;
    this.snapshotJson = snapshotJson;
  }

  public Long getId() {
    return id;
  }

  public AppUser getUser() {
    return user;
  }

  public String getName() {
    return name;
  }

  public String getTemplateId() {
    return templateId;
  }

  public String getDatasetName() {
    return datasetName;
  }

  public int getLayerCount() {
    return layerCount;
  }

  public long getParameterCount() {
    return parameterCount;
  }

  public String getImagePath() {
    return imagePath;
  }

  public String getSnapshotJson() {
    return snapshotJson;
  }

  public Instant getCreatedAt() {
    return createdAt;
  }
}
