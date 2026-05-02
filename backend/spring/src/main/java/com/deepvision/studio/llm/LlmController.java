package com.deepvision.studio.llm;

import com.deepvision.studio.llm.LlmDtos.ChatRequest;
import com.deepvision.studio.llm.LlmDtos.ChatResponse;
import jakarta.validation.Valid;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api/llm")
public class LlmController {
  private final RestTemplate restTemplate;
  private final String arkBaseUrl;
  private final String arkApiKey;
  private final String defaultModel;

  public LlmController(
      RestTemplateBuilder restTemplateBuilder,
      @Value("${deepvision.llm.ark-base-url}") String arkBaseUrl,
      @Value("${deepvision.llm.ark-api-key}") String arkApiKey,
      @Value("${deepvision.llm.default-model}") String defaultModel,
      @Value("${deepvision.llm.connect-timeout-seconds}") long connectTimeoutSeconds,
      @Value("${deepvision.llm.read-timeout-seconds}") long readTimeoutSeconds
  ) {
    this.restTemplate = restTemplateBuilder
        .setConnectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
        .setReadTimeout(Duration.ofSeconds(readTimeoutSeconds))
        .build();
    this.arkBaseUrl = trimTrailingSlash(arkBaseUrl);
    this.arkApiKey = arkApiKey == null ? "" : arkApiKey.trim();
    this.defaultModel = defaultModel;
  }

  @PostMapping(value = "/chat", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  ResponseEntity<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {
    if (arkApiKey.isBlank()) {
      throw new IllegalStateException("ARK_API_KEY is not configured.");
    }

    try {
      HttpHeaders headers = new HttpHeaders();
      headers.setContentType(MediaType.APPLICATION_JSON);
      headers.setBearerAuth(arkApiKey);

      @SuppressWarnings("unchecked")
      Map<String, Object> response = restTemplate.postForObject(
          arkBaseUrl + "/chat/completions",
          new HttpEntity<>(toArkPayload(request), headers),
          Map.class
      );
      return ResponseEntity.ok(parseArkResponse(response));
    } catch (HttpStatusCodeException ex) {
      String body = ex.getResponseBodyAsString();
      throw new IllegalArgumentException(body.isBlank() ? "LLM provider rejected the request." : body);
    } catch (ResourceAccessException ex) {
      throw new IllegalStateException("LLM provider is unavailable.");
    }
  }

  private Map<String, Object> toArkPayload(ChatRequest request) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("model", blankToDefault(request.model(), defaultModel));
    payload.put("reasoning_effort", blankToDefault(request.reasoningEffort(), "medium"));

    List<Map<String, Object>> messages = new ArrayList<>();
    if (request.systemPrompt() != null && !request.systemPrompt().isBlank()) {
      messages.add(Map.of("role", "system", "content", request.systemPrompt()));
    }

    for (var message : request.messages()) {
      List<Map<String, Object>> parts = new ArrayList<>();
      for (var part : message.content()) {
        if ("image_url".equals(part.type())) {
          if (part.imageUrl() == null || part.imageUrl().isBlank()) continue;
          parts.add(Map.of(
              "type", "image_url",
              "image_url", Map.of("url", part.imageUrl())
          ));
        } else {
          parts.add(Map.of("type", "text", "text", blankToDefault(part.text(), "")));
        }
      }
      if (!parts.isEmpty()) {
        messages.add(Map.of("role", message.role(), "content", parts));
      }
    }

    payload.put("messages", messages);
    return payload;
  }

  private ChatResponse parseArkResponse(Map<String, Object> response) {
    if (response == null) {
      return new ChatResponse("", defaultModel, "");
    }
    Object choicesObj = response.get("choices");
    String content = "";
    if (choicesObj instanceof List<?> choices && !choices.isEmpty() && choices.get(0) instanceof Map<?, ?> choice) {
      Object messageObj = choice.get("message");
      if (messageObj instanceof Map<?, ?> message) {
        Object contentObj = message.get("content");
        content = contentObj == null ? "" : String.valueOf(contentObj);
      }
    }
    return new ChatResponse(
        content,
        String.valueOf(response.getOrDefault("model", defaultModel)),
        String.valueOf(response.getOrDefault("id", ""))
    );
  }

  private static String blankToDefault(String value, String fallback) {
    return value == null || value.isBlank() ? fallback : value;
  }

  private static String trimTrailingSlash(String value) {
    if (value == null || value.isBlank()) {
      return "https://ark.cn-beijing.volces.com/api/v3";
    }
    return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
  }
}
