package com.deepvision.studio.training;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TrainingCollaborationStreamHandler extends TextWebSocketHandler {
  private final TrainingJobService jobService;
  private final TrainingCollaborationHandler collaborationHandler;

  public TrainingCollaborationStreamHandler(
      TrainingJobService jobService,
      TrainingCollaborationHandler collaborationHandler
  ) {
    this.jobService = jobService;
    this.collaborationHandler = collaborationHandler;
  }

  @Override
  public void afterConnectionEstablished(WebSocketSession session) throws Exception {
    String jobId = TrainingJobService.jobIdFromSession(session);
    if (jobId == null || jobId.isBlank()) {
      session.close(CloseStatus.BAD_DATA.withReason("jobId is required."));
      return;
    }
    String clientId = TrainingJobService.queryParamFromSession(session, "clientId");
    if (!collaborationHandler.hasParticipant(jobId, clientId)) {
      session.close(CloseStatus.POLICY_VIOLATION.withReason("Join the training chat room before observing."));
      return;
    }
    try {
      jobService.addCollaborationObserverSession(jobId, session);
    } catch (RuntimeException ex) {
      session.close(CloseStatus.POLICY_VIOLATION.withReason("Training job is not available."));
    }
  }

  @Override
  protected void handleTextMessage(WebSocketSession session, TextMessage message) {
    // Collaboration observers receive a read-only stream.
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
