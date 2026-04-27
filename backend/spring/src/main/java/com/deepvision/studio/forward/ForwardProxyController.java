package com.deepvision.studio.forward;

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
public class ForwardProxyController {
  private static final Logger log = LoggerFactory.getLogger(ForwardProxyController.class);

  private final RestTemplate restTemplate;
  private final String forwardBaseUrl;

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
  ResponseEntity<String> health() {
    return proxyGet("/api/health");
  }

  @PostMapping(value = "/forward", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  ResponseEntity<String> forward(@RequestBody String payload) {
    return proxyPost("/api/forward", payload);
  }

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

  private static ResponseEntity<String> jsonResponse(ResponseEntity<String> response) {
    return ResponseEntity
        .status(response.getStatusCode())
        .contentType(MediaType.APPLICATION_JSON)
        .body(response.getBody());
  }

  private static String trimTrailingSlash(String value) {
    if (value == null || value.isBlank()) {
      return "http://127.0.0.1:5000";
    }
    return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
  }
}
