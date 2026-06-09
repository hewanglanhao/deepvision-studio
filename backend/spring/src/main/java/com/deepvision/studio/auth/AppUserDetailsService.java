package com.deepvision.studio.auth;

import java.util.List;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class AppUserDetailsService implements UserDetailsService {
  private final AppUserRepository users;

  /** 注入用户仓库，Spring Security 登录流程会通过该服务读取用户名和密码哈希。 */
  public AppUserDetailsService(AppUserRepository users) {
    this.users = users;
  }

  @Override
  /** 把业务用户转换成 Spring Security 的 UserDetails，供密码校验和 JWT 过滤器复用。 */
  public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
    AppUser user = users.findByUsername(username)
        .orElseThrow(() -> new UsernameNotFoundException("User not found."));
    return new User(user.getUsername(), user.getPasswordHash(), List.of());
  }
}
