package com.deepvision.studio.training;

import com.deepvision.studio.auth.JwtService;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TrainingStreamHandler extends TextWebSocketHandler {
  private final TrainingJobService jobService;
  private final JwtService jwtService;

  public TrainingStreamHandler(TrainingJobService jobService, JwtService jwtService) {
    this.jobService = jobService;
    this.jwtService = jwtService;
  }

  @Override
  // 建立训练指标流连接，要求 URL 带 jobId 和 JWT token，并校验任务归属。
  public void afterConnectionEstablished(WebSocketSession session) throws Exception {
    String jobId = TrainingJobService.jobIdFromSession(session);
    if (jobId == null || jobId.isBlank()) {
      session.close(CloseStatus.BAD_DATA.withReason("jobId is required."));
      return;
    }
    try {
      String token = TrainingJobService.queryParamFromSession(session, "token");
      if (token == null || token.isBlank()) {
        throw new IllegalArgumentException("Authentication is required.");
      }
      jobService.addSession(jwtService.subject(token), jobId, session);
    } catch (RuntimeException ex) {
      session.close(CloseStatus.POLICY_VIOLATION.withReason("Training job is not available."));
    }
  }

  @Override
  // 训练指标由服务端单向推送，客户端发来的消息不参与业务处理。
  protected void handleTextMessage(WebSocketSession session, TextMessage message) {
    // Metrics are server-pushed; client messages are intentionally ignored.
  }

  @Override
  // 连接关闭后移除订阅，避免后续广播写入无效 session。
  public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
    jobService.removeSession(session);
  }

  @Override
  // WebSocket 传输异常时同样清理 session。
  public void handleTransportError(WebSocketSession session, Throwable exception) {
    jobService.removeSession(session);
  }
}
