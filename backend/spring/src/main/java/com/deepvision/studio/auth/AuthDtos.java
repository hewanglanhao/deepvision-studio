package com.deepvision.studio.auth;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.time.Instant;

public final class AuthDtos {
  private AuthDtos() {}

  @Schema(description = "Login or registration request")
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
  public record UserResponse(Long id, String username, String displayName, Instant createdAt) {
    static UserResponse from(AppUser user) {
      return new UserResponse(user.getId(), user.getUsername(), user.getDisplayName(), user.getCreatedAt());
    }
  }

  @Schema(description = "Authentication response containing JWT and user profile")
  public record AuthResponse(String token, UserResponse user) {}
}
