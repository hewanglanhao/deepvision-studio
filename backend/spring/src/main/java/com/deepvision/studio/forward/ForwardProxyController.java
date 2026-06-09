package com.deepvision.studio.forward;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
@Tag(name = "Mode A Forward", description = "Proxy APIs for Python forward-pass execution")
public class ForwardProxyController {
  private static final Logger log = LoggerFactory.getLogger(ForwardProxyController.class);

  private final RestTemplate restTemplate;
  private final String forwardBaseUrl;

  /** 注入 HTTP 客户端和 Python forward 地址；Spring 只做代理，真实张量计算交给 Python/NumPy。 */
  public ForwardProxyController(
      RestTemplateBuilder restTemplateBuilder,
      @Value("${deepvision.forward.base-url}") String forwardBaseUrl,
      @Value("${deepvision.forward.connect-timeout-seconds}") long connectTimeoutSeconds,
      @Value("${deepvision.forward.read-timeout-seconds}") long readTimeoutSeconds
  ) {
    this.restTemplate = restTemplateBuilder
        .setConnectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
        .setReadTimeout(Duration.ofSeconds(readTimeoutSeconds))
        .build();
    this.forwardBaseUrl = trimTrailingSlash(forwardBaseUrl);
  }

  @GetMapping("/forward/health")
  @Operation(summary = "Check Python forward service health")
  @ApiResponse(responseCode = "200", description = "Python service responded")
  @ApiResponse(responseCode = "503", description = "Python service is unavailable")
  /** 代理 Python forward 健康检查，部署时用它确认计算服务是否在线。 */
  ResponseEntity<String> health() {
    return proxyGet("/api/health");
  }

  @PostMapping(value = "/forward", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  @Operation(summary = "Execute a forward pass through the Python runtime")
  @ApiResponse(responseCode = "200", description = "Forward pass completed")
  @ApiResponse(responseCode = "400", description = "Invalid graph or payload")
  @ApiResponse(responseCode = "503", description = "Python service is unavailable")
  /** 接收 A 模式网络结构和输入张量，并转发给 Python 执行真实前向传播。 */
  ResponseEntity<String> forward(@RequestBody String payload) {
    return proxyPost("/api/forward", payload);
  }

  /** 代理 GET 请求到 Python 服务，保留 Python 返回的状态码和 JSON 内容。 */
  private ResponseEntity<String> proxyGet(String path) {
    try {
      ResponseEntity<String> response = restTemplate.getForEntity(forwardBaseUrl + path, String.class);
      return jsonResponse(response);
    } catch (HttpStatusCodeException ex) {
      return ResponseEntity.status(ex.getStatusCode()).contentType(MediaType.APPLICATION_JSON).body(ex.getResponseBodyAsString());
    } catch (ResourceAccessException ex) {
      throw new IllegalStateException("Python forward service is unavailable.");
    }
  }

  /** 代理 POST 请求到 Python forward；payload 中包含 layers、connections 和 inputTensor。 */
  private ResponseEntity<String> proxyPost(String path, String payload) {
    try {
      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.APPLICATION_JSON);
      ResponseEntity<String> response = restTemplate.postForEntity(
          forwardBaseUrl + path,
          new HttpEntity<>(payload, headers),
          String.class
      );
      int bytes = response.getBody() == null ? 0 : response.getBody().length();
      log.info("Forward proxy completed with status {}, response chars={}", response.getStatusCode(), bytes);
      return jsonResponse(response);
    } catch (HttpStatusCodeException ex) {
      log.warn("Forward proxy returned status {}", ex.getStatusCode());
      return ResponseEntity.status(ex.getStatusCode()).contentType(MediaType.APPLICATION_JSON).body(ex.getResponseBodyAsString());
    } catch (ResourceAccessException ex) {
      log.warn("Forward proxy cannot reach Python service: {}", ex.getMessage());
      throw new IllegalStateException("Python forward service is unavailable.");
    }
  }

  /** 将 Python 服务响应包装成 JSON 响应，前端可以直接解析 ForwardPassResult。 */
  private static ResponseEntity<String> jsonResponse(ResponseEntity<String> response) {
    return ResponseEntity
        .status(response.getStatusCode())
        .contentType(MediaType.APPLICATION_JSON)
        .body(response.getBody());
  }

  /** 统一 forward 服务地址格式，避免拼接 /api/forward 时出现双斜杠。 */
  private static String trimTrailingSlash(String value) {
    if (value == null || value.isBlank()) {
      return "http://127.0.0.1:5000";
    }
    return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
  }
}
