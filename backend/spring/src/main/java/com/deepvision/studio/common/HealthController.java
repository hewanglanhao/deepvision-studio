package com.deepvision.studio.common;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Tag(name = "Health", description = "Service health check")
public class HealthController {
  @GetMapping("/api/health")
  @Operation(summary = "Check Spring backend health")
  /** 返回服务健康状态，供前端或部署环境探活。 */
  public Map<String, Object> health() {
    return Map.of("ok", true, "service", "spring-backend");
  }
}
