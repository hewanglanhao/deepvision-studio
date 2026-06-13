package com.deepvision.studio.training;

import com.deepvision.studio.museum.MuseumPresenceHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class TrainingWebSocketConfig implements WebSocketConfigurer {
  private static final String[] LOCAL_FRONTEND_ORIGINS = {
      "http://localhost:4200",
      "http://127.0.0.1:4200",
      "http://localhost:4201",
      "http://127.0.0.1:4201",
      "http://localhost:4202",
      "http://127.0.0.1:4202"
  };

  private final TrainingStreamHandler streamHandler;
  private final TrainingCollaborationHandler collaborationHandler;
  private final TrainingCollaborationStreamHandler collaborationStreamHandler;
  private final MuseumPresenceHandler museumPresenceHandler;

  public TrainingWebSocketConfig(
      TrainingStreamHandler streamHandler,
      TrainingCollaborationHandler collaborationHandler,
      TrainingCollaborationStreamHandler collaborationStreamHandler,
      MuseumPresenceHandler museumPresenceHandler
  ) {
    this.streamHandler = streamHandler;
    this.collaborationHandler = collaborationHandler;
    this.collaborationStreamHandler = collaborationStreamHandler;
    this.museumPresenceHandler = museumPresenceHandler;
  }

  @Override
  public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
    registry.addHandler(streamHandler, "/api/training/stream")
        .setAllowedOrigins(LOCAL_FRONTEND_ORIGINS);
    registry.addHandler(collaborationHandler, "/api/training/collaboration")
        .setAllowedOrigins(LOCAL_FRONTEND_ORIGINS);
    registry.addHandler(collaborationStreamHandler, "/api/training/collaboration/stream")
        .setAllowedOrigins(LOCAL_FRONTEND_ORIGINS);
    registry.addHandler(museumPresenceHandler, "/api/museum/presence")
        .setAllowedOrigins(LOCAL_FRONTEND_ORIGINS);
  }
}
