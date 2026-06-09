package com.deepvision.studio.auth;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(name = "app_users")
public class AppUser {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @Column(nullable = false, unique = true, length = 64)
  private String username;

  @Column(nullable = false)
  private String passwordHash;

  @Column(nullable = false, length = 80)
  private String displayName;

  @Column(nullable = false)
  private Instant createdAt = Instant.now();

  /** JPA 反射创建实体时需要的空构造器，业务代码不直接使用。 */
  protected AppUser() {}

  /** 创建平台用户，密码只保存 BCrypt 哈希，后续 A 模式的推理记录会通过用户 id 归属到该账号。 */
  public AppUser(String username, String passwordHash, String displayName) {
    this.username = username;
    this.passwordHash = passwordHash;
    this.displayName = displayName;
  }

  /** 返回数据库主键，用于关联用户私有的 forward 推理记录。 */
  public Long getId() {
    return id;
  }

  /** 返回唯一用户名，JWT subject 和 Spring Security Principal 都使用它识别用户。 */
  public String getUsername() {
    return username;
  }

  /** 返回 BCrypt 密码哈希，认证时由 PasswordEncoder 与用户输入的明文密码比对。 */
  public String getPasswordHash() {
    return passwordHash;
  }

  /** 返回展示名，前端导航和记录列表用它显示当前实验操作者。 */
  public String getDisplayName() {
    return displayName;
  }

  /** 返回账号创建时间，接口摘要中可用于展示用户注册或审计信息。 */
  public Instant getCreatedAt() {
    return createdAt;
  }
}
