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

  /** JPA 反射构造器；业务代码创建记录时使用带参数构造器。 */
  protected ForwardRecord() {}

  /** 创建一条 A 模式记录实体，元数据用于列表展示，snapshotJson 用于完整恢复实验页面。 */
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

  /** 返回记录主键，前端用它请求详情、恢复或删除快照。 */
  public Long getId() {
    return id;
  }

  /** 返回所属用户，记录通过用户关联实现历史快照隔离。 */
  public AppUser getUser() {
    return user;
  }

  /** 返回用户给这次实验保存的名称。 */
  public String getName() {
    return name;
  }

  /** 返回保存时选中的网络模板，例如 cnn-classic 或 residual-cnn。 */
  public String getTemplateId() {
    return templateId;
  }

  /** 返回保存时使用的数据集或样本类别名称。 */
  public String getDatasetName() {
    return datasetName;
  }

  /** 返回保存时网络层数，列表中用于快速判断模型深度。 */
  public int getLayerCount() {
    return layerCount;
  }

  /** 返回保存时参数量，列表中用于快速判断模型容量。 */
  public long getParameterCount() {
    return parameterCount;
  }

  /** 返回输入预览图路径，前端恢复记录时可重新构建输入张量。 */
  public String getImagePath() {
    return imagePath;
  }

  /** 返回完整页面快照 JSON，包括 layers、connections、选中层和 forwardResult。 */
  public String getSnapshotJson() {
    return snapshotJson;
  }

  /** 返回记录创建时间，用于历史记录按时间倒序展示。 */
  public Instant getCreatedAt() {
    return createdAt;
  }
}
