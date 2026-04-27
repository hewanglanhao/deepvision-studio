package com.deepvision.studio.training;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TrainingStreamHandler extends TextWebSocketHandler {
  private final TrainingJobService jobService;

  public TrainingStreamHandler(TrainingJobService jobService) {
    this.jobService = jobService;
  }

  @Override
  public void afterConnectionEstablished(WebSocketSession session) throws Exception {
    String jobId = TrainingJobService.jobIdFromSession(session);
    if (jobId == null || jobId.isBlank()) {
      session.close(CloseStatus.BAD_DATA.withReason("jobId is required."));
      return;
    }
    try {
      jobService.addSession(jobId, session);
    } catch (IllegalArgumentException ex) {
      session.close(CloseStatus.BAD_DATA.withReason(ex.getMessage()));
    }
  }

  @Override
  protected void handleTextMessage(WebSocketSession session, TextMessage message) {
    // Metrics are server-pushed; client messages are intentionally ignored.
  }

  @Override
  public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
    jobService.removeSession(session);
  }

  @Override
  public void handleTransportError(WebSocketSession session, Throwable exception) {
    jobService.removeSession(session);
  }
}
