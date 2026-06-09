package com.deepvision.studio.llm;

import com.deepvision.studio.llm.LlmDtos.ChatRequest;
import com.deepvision.studio.llm.LlmDtos.ChatResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
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
@Tag(name = "LLM", description = "Assistant chat proxy APIs")
public class LlmController {
  private final LlmChatClient llmChatClient;
  private final ExecutorService streamExecutor = Executors.newCachedThreadPool();

  /** 注入 LLM 客户端，向前端暴露普通回答和 SSE 流式回答两个代理入口。 */
  public LlmController(LlmChatClient llmChatClient) {
    this.llmChatClient = llmChatClient;
  }

  @PostMapping(value = "/chat", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  @Operation(summary = "Run a non-streaming assistant chat request")
  @ApiResponse(responseCode = "200", description = "Assistant response")
  @ApiResponse(responseCode = "400", description = "Invalid message payload")
  /** 发送普通 LLM 对话请求并返回完整响应。 */
  ResponseEntity<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {
    return ResponseEntity.ok(llmChatClient.chat(request));
  }

  @PostMapping(value = "/chat/stream", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  @Operation(summary = "Run a streaming assistant chat request with SSE")
  @ApiResponse(responseCode = "200", description = "SSE stream with delta, done, or error events")
  @ApiResponse(responseCode = "400", description = "Invalid message payload")
  /** 通过 SSE 接收 LLM 流式响应并逐段回调给界面。 */
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
  /** 应用关闭时停止流式响应线程池，避免正在等待上游模型的后台线程泄漏。 */
  void shutdown() {
    streamExecutor.shutdownNow();
  }

  /** 向前端发送 SSE 事件，delta 用于逐字显示回答，done/error 用于结束本次提问。 */
  private void sendEvent(SseEmitter emitter, String event, Object data) {
    try {
      emitter.send(SseEmitter.event().name(event).data(data));
    } catch (IOException ignored) {
      // The client disconnected; the emitter will be completed by the caller.
    }
  }
}
