package com.deepvision.studio.training;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.auth.JwtService;
import com.deepvision.studio.training.TrainingDtos.CollaborationRoomSummary;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.Comparator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TrainingCollaborationHandler extends TextWebSocketHandler {
  private static final int MAX_RECENT_MESSAGES = 60;

  private final TrainingJobService jobService;
  private final JwtService jwtService;
  private final AppUserRepository users;
  private final ObjectMapper objectMapper;
  private final Map<String, RoomState> rooms = new ConcurrentHashMap<>();
  private final Map<String, Participant> participants = new ConcurrentHashMap<>();

  public TrainingCollaborationHandler(
      TrainingJobService jobService,
      JwtService jwtService,
      AppUserRepository users,
      ObjectMapper objectMapper
  ) {
    this.jobService = jobService;
    this.jwtService = jwtService;
    this.users = users;
    this.objectMapper = objectMapper;
  }

  public List<CollaborationRoomSummary> listRooms() {
    return rooms.entrySet().stream()
        .map(entry -> {
          String jobId = entry.getKey();
          RoomState room = entry.getValue();
          List<String> names = room.sessions.stream()
              .map(session -> participants.get(session.getId()))
              .filter(participant -> participant != null)
              .map(Participant::displayName)
              .distinct()
              .toList();
          return new CollaborationRoomSummary(jobId, names.size(), room.createdAt, names);
        })
        .sorted(Comparator.comparing(CollaborationRoomSummary::createdAt).reversed())
        .toList();
  }

  @Override
  public void afterConnectionEstablished(WebSocketSession session) throws Exception {
    Map<String, String> query = queryParams(session.getUri());
    String jobId = query.getOrDefault("jobId", "");
    if (jobId.isBlank()) {
      session.close(CloseStatus.BAD_DATA.withReason("jobId is required."));
      return;
    }
    try {
      jobService.status(jobId);
    } catch (IllegalArgumentException ex) {
      session.close(CloseStatus.BAD_DATA.withReason(ex.getMessage()));
      return;
    }
    boolean createRoom = "true".equalsIgnoreCase(query.getOrDefault("create", "false"));
    if (!createRoom && !rooms.containsKey(jobId)) {
      session.close(CloseStatus.BAD_DATA.withReason("Training chat room does not exist."));
      return;
    }

    Participant participant = identifyParticipant(session, query, jobId);
    participants.put(session.getId(), participant);
    RoomState room = rooms.computeIfAbsent(jobId, ignored -> new RoomState());
    room.sessions.add(session);

    send(session, Map.of(
        "type", "history",
        "jobId", jobId,
        "messages", room.recentMessages()
    ));
    broadcastPresence(jobId);
    broadcastSystem(jobId, participant.displayName() + " 加入了训练房间。");
  }

  @Override
  protected void handleTextMessage(WebSocketSession session, TextMessage message) {
    Participant participant = participants.get(session.getId());
    if (participant == null) {
      return;
    }
    try {
      JsonNode node = objectMapper.readTree(message.getPayload());
      String type = node.path("type").asText("");
      if (!"chat".equals(type)) {
        return;
      }
      String text = node.path("text").asText("").trim();
      if (text.isBlank()) {
        return;
      }
      if (text.length() > 800) {
        text = text.substring(0, 800);
      }
      Map<String, Object> payload = new LinkedHashMap<>();
      payload.put("type", "chat");
      payload.put("id", UUID.randomUUID().toString());
      payload.put("jobId", participant.jobId());
      payload.put("username", participant.username());
      payload.put("displayName", participant.displayName());
      payload.put("text", text);
      payload.put("createdAt", Instant.now().toString());
      RoomState room = rooms.computeIfAbsent(participant.jobId(), ignored -> new RoomState());
      room.addMessage(payload);
      broadcast(participant.jobId(), payload);
    } catch (JsonProcessingException ignored) {
      // Ignore malformed client messages.
    }
  }

  @Override
  public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
    Participant participant = participants.remove(session.getId());
    if (participant == null) {
      return;
    }
    RoomState room = rooms.get(participant.jobId());
    if (room != null) {
      room.sessions.remove(session);
      broadcastSystem(participant.jobId(), participant.displayName() + " 离开了训练房间。");
      broadcastPresence(participant.jobId());
      if (room.sessions.isEmpty()) {
        rooms.remove(participant.jobId());
      }
    }
  }

  @Override
  public void handleTransportError(WebSocketSession session, Throwable exception) {
    afterConnectionClosed(session, CloseStatus.SERVER_ERROR);
  }

  private Participant identifyParticipant(WebSocketSession session, Map<String, String> query, String jobId) {
    String token = query.getOrDefault("token", "");
    if (!token.isBlank()) {
      try {
        String username = jwtService.subject(token);
        AppUser user = users.findByUsername(username).orElse(null);
        if (user != null) {
          return new Participant(jobId, user.getUsername(), user.getDisplayName());
        }
      } catch (RuntimeException ignored) {
        // Fall back to guest identity.
      }
    }
    String guestName = query.getOrDefault("name", "").trim();
    if (guestName.isBlank()) {
      guestName = "访客-" + session.getId().substring(0, Math.min(4, session.getId().length()));
    }
    return new Participant(jobId, "guest-" + session.getId(), guestName);
  }

  private void broadcastPresence(String jobId) {
    RoomState room = rooms.get(jobId);
    if (room == null) {
      return;
    }
    List<Map<String, String>> activeUsers = room.sessions.stream()
        .map(session -> participants.get(session.getId()))
        .filter(participant -> participant != null)
        .map(participant -> Map.of(
            "username", participant.username(),
            "displayName", participant.displayName()
        ))
        .distinct()
        .toList();
    broadcast(jobId, Map.of(
        "type", "presence",
        "jobId", jobId,
        "users", activeUsers
    ));
  }

  private void broadcastSystem(String jobId, String text) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("type", "system");
    payload.put("jobId", jobId);
    payload.put("text", text);
    payload.put("createdAt", Instant.now().toString());
    broadcast(jobId, payload);
  }

  private void broadcast(String jobId, Object payload) {
    RoomState room = rooms.get(jobId);
    if (room == null) {
      return;
    }
    String json;
    try {
      json = objectMapper.writeValueAsString(payload);
    } catch (JsonProcessingException ex) {
      return;
    }
    for (WebSocketSession session : room.sessions) {
      if (!session.isOpen()) {
        room.sessions.remove(session);
        continue;
      }
      try {
        session.sendMessage(new TextMessage(json));
      } catch (IOException ex) {
        room.sessions.remove(session);
      }
    }
  }

  private void send(WebSocketSession session, Object payload) {
    if (!session.isOpen()) {
      return;
    }
    try {
      session.sendMessage(new TextMessage(objectMapper.writeValueAsString(payload)));
    } catch (IOException ignored) {
      // The close callback will clean up the session.
    }
  }

  private Map<String, String> queryParams(URI uri) {
    Map<String, String> params = new LinkedHashMap<>();
    if (uri == null || uri.getQuery() == null) {
      return params;
    }
    for (String part : uri.getQuery().split("&")) {
      String[] pair = part.split("=", 2);
      if (pair.length == 2) {
        params.put(
            URLDecoder.decode(pair[0], StandardCharsets.UTF_8),
            URLDecoder.decode(pair[1], StandardCharsets.UTF_8)
        );
      }
    }
    return params;
  }

  private record Participant(String jobId, String username, String displayName) {}

  private static final class RoomState {
    private final CopyOnWriteArraySet<WebSocketSession> sessions = new CopyOnWriteArraySet<>();
    private final ArrayDeque<Map<String, Object>> recentMessages = new ArrayDeque<>();
    private final Instant createdAt = Instant.now();

    private synchronized void addMessage(Map<String, Object> message) {
      recentMessages.addLast(message);
      while (recentMessages.size() > MAX_RECENT_MESSAGES) {
        recentMessages.removeFirst();
      }
    }

    private synchronized List<Map<String, Object>> recentMessages() {
      return new ArrayList<>(recentMessages);
    }
  }
}
