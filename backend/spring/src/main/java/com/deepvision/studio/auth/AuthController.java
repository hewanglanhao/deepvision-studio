package com.deepvision.studio.auth;

import com.deepvision.studio.auth.AuthDtos.AuthRequest;
import com.deepvision.studio.auth.AuthDtos.AuthResponse;
import com.deepvision.studio.auth.AuthDtos.UserResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import java.security.Principal;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
@Tag(name = "Auth", description = "User registration, login, and JWT session APIs")
public class AuthController {
  private final AppUserRepository users;
  private final PasswordEncoder passwordEncoder;
  private final JwtService jwtService;
  private final AuthenticationManager authenticationManager;

  /** 汇集用户仓库、密码加密器、JWT 服务和认证管理器，提供前端登录注册入口。 */
  public AuthController(
      AppUserRepository users,
      PasswordEncoder passwordEncoder,
      JwtService jwtService,
      AuthenticationManager authenticationManager
  ) {
    this.users = users;
    this.passwordEncoder = passwordEncoder;
    this.jwtService = jwtService;
    this.authenticationManager = authenticationManager;
  }

  @PostMapping("/register")
  @Operation(summary = "Register a new user")
  @ApiResponse(responseCode = "200", description = "User registered and JWT issued")
  @ApiResponse(responseCode = "400", description = "Invalid request or duplicated username")
  /** 创建新用户、加密密码并签发登录凭证。 */
  public AuthResponse register(@Valid @RequestBody AuthRequest request) {
    String username = request.username().trim();
    if (users.existsByUsername(username)) {
      throw new IllegalArgumentException("Username already exists.");
    }
    String displayName = request.displayName() == null || request.displayName().isBlank()
        ? username
        : request.displayName().trim();
    AppUser user = users.save(new AppUser(username, passwordEncoder.encode(request.password()), displayName));
    return new AuthResponse(jwtService.issue(user), UserResponse.from(user));
  }

  @PostMapping("/login")
  @Operation(summary = "Login with username and password")
  @ApiResponse(responseCode = "200", description = "Login succeeded and JWT issued")
  @ApiResponse(responseCode = "400", description = "Invalid username or password")
  /** 校验用户名密码并签发 JWT 登录凭证。 */
  public AuthResponse login(@Valid @RequestBody AuthRequest request) {
    authenticationManager.authenticate(
        new UsernamePasswordAuthenticationToken(request.username().trim(), request.password())
    );
    AppUser user = users.findByUsername(request.username().trim())
        .orElseThrow(() -> new IllegalArgumentException("Invalid username or password."));
    return new AuthResponse(jwtService.issue(user), UserResponse.from(user));
  }

  @GetMapping("/me")
  @Operation(summary = "Get current authenticated user")
  @ApiResponse(responseCode = "200", description = "Current user, or null when no principal is available")
  /** 返回当前 JWT 对应的用户信息。 */
  public UserResponse me(Principal principal) {
    if (principal == null) {
      return null;
    }
    AppUser user = users.findByUsername(principal.getName())
        .orElseThrow(() -> new IllegalArgumentException("User not found."));
    return UserResponse.from(user);
  }
}
