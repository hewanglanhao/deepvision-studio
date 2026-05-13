package com.deepvision.studio.llm;

import com.deepvision.studio.llm.LlmDtos.ChatRequest;
import com.deepvision.studio.llm.LlmDtos.ChatResponse;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

@Component
public class LlmChatClient {
  private final RestTemplate restTemplate;
  private final ObjectMapper objectMapper;
  private final String arkBaseUrl;
  private final String arkApiKey;
  private final String defaultModel;

  public LlmChatClient(
      RestTemplateBuilder restTemplateBuilder,
      ObjectMapper objectMapper,
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
    this.objectMapper = objectMapper;
    this.arkBaseUrl = trimTrailingSlash(arkBaseUrl);
    this.arkApiKey = arkApiKey == null ? "" : arkApiKey.trim();
    this.defaultModel = defaultModel;
  }

  public ChatResponse chat(ChatRequest request) {
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
          new org.springframework.http.HttpEntity<>(toArkPayload(request, false), headers),
          Map.class
      );
      return parseArkResponse(response);
    } catch (HttpStatusCodeException ex) {
      String body = ex.getResponseBodyAsString();
      throw new IllegalArgumentException(body.isBlank() ? "LLM provider rejected the request." : body);
    } catch (ResourceAccessException ex) {
      throw new IllegalStateException("LLM provider is unavailable.");
    }
  }

  public ChatResponse stream(ChatRequest request, Consumer<String> onDelta) {
    if (arkApiKey.isBlank()) {
      throw new IllegalStateException("ARK_API_KEY is not configured.");
    }
    StringBuilder content = new StringBuilder();
    try {
      restTemplate.execute(
          arkBaseUrl + "/chat/completions",
          HttpMethod.POST,
          req -> {
            req.getHeaders().setContentType(MediaType.APPLICATION_JSON);
            req.getHeaders().setAccept(List.of(MediaType.TEXT_EVENT_STREAM, MediaType.APPLICATION_JSON));
            req.getHeaders().setBearerAuth(arkApiKey);
            objectMapper.writeValue(req.getBody(), toArkPayload(request, true));
          },
          response -> {
            if (!response.getStatusCode().is2xxSuccessful()) {
              String body = new String(response.getBody().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
              throw new IllegalArgumentException(body.isBlank() ? "LLM provider rejected the request." : body);
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.getBody(), java.nio.charset.StandardCharsets.UTF_8))) {
              String line;
              while ((line = reader.readLine()) != null) {
                String data = sseData(line);
                if (data == null || data.isBlank()) {
                  continue;
                }
                if ("[DONE]".equals(data)) {
                  break;
                }
                String delta = streamDelta(data);
                if (!delta.isBlank()) {
                  content.append(delta);
                  onDelta.accept(delta);
                }
              }
            }
            return null;
          }
      );
      return new ChatResponse(content.toString(), blankToDefault(request.model(), defaultModel), "");
    } catch (ResourceAccessException ex) {
      throw new IllegalStateException("LLM provider is unavailable.");
    }
  }

  private Map<String, Object> toArkPayload(ChatRequest request, boolean stream) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("model", blankToDefault(request.model(), defaultModel));
    payload.put("reasoning_effort", blankToDefault(request.reasoningEffort(), "medium"));
    if (stream) {
      payload.put("stream", true);
    }

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

  private String sseData(String line) {
    String trimmed = line.trim();
    if (!trimmed.startsWith("data:")) {
      return null;
    }
    return trimmed.substring(5).trim();
  }

  private String streamDelta(String data) {
    try {
      JsonNode root = objectMapper.readTree(data);
      JsonNode choices = root.path("choices");
      if (!choices.isArray() || choices.isEmpty()) {
        return "";
      }
      JsonNode choice = choices.get(0);
      JsonNode delta = choice.path("delta");
      JsonNode content = delta.path("content");
      if (content.isTextual()) {
        return content.asText();
      }
      JsonNode reasoning = delta.path("reasoning_content");
      if (reasoning.isTextual()) {
        return reasoning.asText();
      }
      JsonNode messageContent = choice.path("message").path("content");
      return messageContent.isTextual() ? messageContent.asText() : "";
    } catch (JsonProcessingException ex) {
      return "";
    }
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
