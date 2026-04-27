package com.deepvision.studio.auth;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.time.Instant;

public final class AuthDtos {
  private AuthDtos() {}

  public record AuthRequest(
      @NotBlank(message = "Username is required.")
      @Pattern(regexp = "^[A-Za-z0-9_]{3,32}$", message = "Username must be 3-32 letters, numbers, or underscores.")
      String username,

      @NotBlank(message = "Password is required.")
      @Size(min = 6, max = 72, message = "Password must be 6-72 characters.")
      String password,

      @Size(max = 80, message = "Display name must be at most 80 characters.")
      String displayName
  ) {}

  public record UserResponse(Long id, String username, String displayName, Instant createdAt) {
    static UserResponse from(AppUser user) {
      return new UserResponse(user.getId(), user.getUsername(), user.getDisplayName(), user.getCreatedAt());
    }
  }

  public record AuthResponse(String token, UserResponse user) {}
}

