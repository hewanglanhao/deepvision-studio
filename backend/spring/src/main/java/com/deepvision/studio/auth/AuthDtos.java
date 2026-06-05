package com.deepvision.studio.auth;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.time.Instant;

public final class AuthDtos {
  /** DTO 容器类只承载认证接口的数据结构，不允许被实例化。 */
  private AuthDtos() {}

  @Schema(description = "Login or registration request")
  /** 前端提交的登录/注册表单，后端在这里约束用户名格式和密码长度。 */
  public record AuthRequest(
      @Schema(description = "Username, 3-32 letters, numbers, or underscores", example = "student_01")
      @NotBlank(message = "Username is required.")
      @Pattern(regexp = "^[A-Za-z0-9_]{3,32}$", message = "Username must be 3-32 letters, numbers, or underscores.")
      String username,

      @Schema(description = "Plain password submitted by the client", example = "password123")
      @NotBlank(message = "Password is required.")
      @Size(min = 6, max = 72, message = "Password must be 6-72 characters.")
      String password,

      @Schema(description = "Optional display name", example = "王同学")
      @Size(max = 80, message = "Display name must be at most 80 characters.")
      String displayName
  ) {}

  @Schema(description = "Authenticated user profile")
  /** 返回给前端的用户摘要，刻意不包含密码哈希，只保留界面和记录归属需要的信息。 */
  public record UserResponse(Long id, String username, String displayName, Instant createdAt) {
    /** 从数据库实体投影成安全响应对象，避免把 AppUser 的敏感字段直接序列化。 */
    static UserResponse from(AppUser user) {
      return new UserResponse(user.getId(), user.getUsername(), user.getDisplayName(), user.getCreatedAt());
    }
  }

  @Schema(description = "Authentication response containing JWT and user profile")
  /** 认证成功响应，JWT 用于后续接口鉴权，user 用于前端立即刷新登录态。 */
  public record AuthResponse(String token, UserResponse user) {}
}
