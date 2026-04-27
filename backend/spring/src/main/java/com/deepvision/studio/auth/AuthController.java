package com.deepvision.studio.auth;

import com.deepvision.studio.auth.AuthDtos.AuthRequest;
import com.deepvision.studio.auth.AuthDtos.AuthResponse;
import com.deepvision.studio.auth.AuthDtos.UserResponse;
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
public class AuthController {
  private final AppUserRepository users;
  private final PasswordEncoder passwordEncoder;
  private final JwtService jwtService;
  private final AuthenticationManager authenticationManager;

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
  public AuthResponse login(@Valid @RequestBody AuthRequest request) {
    authenticationManager.authenticate(
        new UsernamePasswordAuthenticationToken(request.username().trim(), request.password())
    );
    AppUser user = users.findByUsername(request.username().trim())
        .orElseThrow(() -> new IllegalArgumentException("Invalid username or password."));
    return new AuthResponse(jwtService.issue(user), UserResponse.from(user));
  }

  @GetMapping("/me")
  public UserResponse me(Principal principal) {
    if (principal == null) {
      return null;
    }
    AppUser user = users.findByUsername(principal.getName())
        .orElseThrow(() -> new IllegalArgumentException("User not found."));
    return UserResponse.from(user);
  }
}
