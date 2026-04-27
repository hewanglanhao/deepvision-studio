package com.deepvision.studio.training;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class TrainingWebSocketConfig implements WebSocketConfigurer {
  private final TrainingStreamHandler streamHandler;

  public TrainingWebSocketConfig(TrainingStreamHandler streamHandler) {
    this.streamHandler = streamHandler;
  }

  @Override
  public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
    registry.addHandler(streamHandler, "/api/training/stream")
        .setAllowedOrigins("http://localhost:4200", "http://127.0.0.1:4200");
  }
}
