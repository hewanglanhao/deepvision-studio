package com.deepvision.studio.llm;

import com.deepvision.studio.llm.LlmDtos.ChatRequest;
import com.deepvision.studio.llm.LlmDtos.ChatResponse;
import jakarta.annotation.PreDestroy;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/llm")
public class LlmController {
  private final LlmChatClient llmChatClient;
  private final ExecutorService streamExecutor = Executors.newCachedThreadPool();

  public LlmController(LlmChatClient llmChatClient) {
    this.llmChatClient = llmChatClient;
  }

  @PostMapping(value = "/chat", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  ResponseEntity<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {
    return ResponseEntity.ok(llmChatClient.chat(request));
  }

  @PostMapping(value = "/chat/stream", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  SseEmitter streamChat(@Valid @RequestBody ChatRequest request) {
    SseEmitter emitter = new SseEmitter(180_000L);
    streamExecutor.submit(() -> {
      try {
        ChatResponse response = llmChatClient.stream(request, delta -> sendEvent(emitter, "delta", Map.of("text", delta)));
        sendEvent(emitter, "done", response);
        emitter.complete();
      } catch (RuntimeException ex) {
        sendEvent(emitter, "error", Map.of("message", ex.getMessage() == null ? "LLM stream failed." : ex.getMessage()));
        emitter.complete();
      }
    });
    return emitter;
  }

  @PreDestroy
  void shutdown() {
    streamExecutor.shutdownNow();
  }

  private void sendEvent(SseEmitter emitter, String event, Object data) {
    try {
      emitter.send(SseEmitter.event().name(event).data(data));
    } catch (IOException ignored) {
      // The client disconnected; the emitter will be completed by the caller.
    }
  }
}
