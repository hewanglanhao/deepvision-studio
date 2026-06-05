package com.deepvision.studio.museum;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.auth.JwtService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class MuseumPresenceHandler extends TextWebSocketHandler {
  private static final int ROOM_LIMIT = 8;

  private final JwtService jwtService;
  private final AppUserRepository users;
  private final ObjectMapper objectMapper;
  private final Map<String, RoomState> rooms = new ConcurrentHashMap<>();
  private final Map<String, Participant> participants = new ConcurrentHashMap<>();

  /** 注入 JWT、用户仓库和 JSON 工具，WebSocket 连接可识别登录用户并同步博物馆中的在线位置。 */
  public MuseumPresenceHandler(JwtService jwtService, AppUserRepository users, ObjectMapper objectMapper) {
    this.jwtService = jwtService;
    this.users = users;
    this.objectMapper = objectMapper;
  }

  @Override
  /** 建立博物馆 WebSocket 连接并加入在线房间。 */
  public void afterConnectionEstablished(WebSocketSession session) throws Exception {
    Map<String, String> query = queryParams(session.getUri());
    String token = query.getOrDefault("token", "");
    String username;
    String displayName;
    if (token.isBlank()) {
      displayName = nextGuestDisplayName();
      username = "guest-" + displayName + "-" + session.getId();
    } else {
      AppUser user;
      try {
        username = jwtService.subject(token);
        user = users.findByUsername(username).orElse(null);
      } catch (RuntimeException ex) {
        session.close(CloseStatus.POLICY_VIOLATION.withReason("Invalid token."));
        return;
      }
      if (user == null) {
        session.close(CloseStatus.POLICY_VIOLATION.withReason("User not found."));
        return;
      }
      displayName = user.getDisplayName();
    }

    RoomState room;
    Participant participant;
    List<Map<String, Object>> currentParticipants;
    synchronized (this) {
      room = availableRoom();
      participant = new Participant(
          session.getId(),
          room.id,
          username,
          displayName,
          colorFor(username),
          0,
          1.75,
          -68,
          0,
          Instant.now().toString()
      );
      room.sessions.add(session);
      participants.put(session.getId(), participant);
      currentParticipants = roomParticipants(room);
    }

    send(session, Map.of(
        "type", "welcome",
        "selfId", participant.id(),
        "roomId", room.id,
        "limit", ROOM_LIMIT,
        "participants", currentParticipants
    ));
    broadcast(room, Map.of("type", "join", "participant", toPayload(participant)), session.getId());
  }

  @Override
  /** 处理博物馆 WebSocket 位姿消息并广播给同房间用户。 */
  protected void handleTextMessage(WebSocketSession session, TextMessage message) {
    Participant current = participants.get(session.getId());
    if (current == null) return;
    try {
      JsonNode node = objectMapper.readTree(message.getPayload());
      if (!"pose".equals(node.path("type").asText(""))) {
        return;
      }
      Participant updated = new Participant(
          current.id(),
          current.roomId(),
          current.username(),
          current.displayName(),
          current.color(),
          clamp(node.path("x").asDouble(current.x()), -8.0, 8.0),
          clamp(node.path("y").asDouble(current.y()), 0.8, 3.0),
          clamp(node.path("z").asDouble(current.z()), -80.0, 80.0),
          node.path("ry").asDouble(current.ry()),
          Instant.now().toString()
      );
      participants.put(session.getId(), updated);
      RoomState room = rooms.get(updated.roomId());
      if (room != null) {
        broadcast(room, Map.of("type", "pose", "participant", toPayload(updated)), session.getId());
      }
    } catch (JsonProcessingException ignored) {
      // Ignore malformed messages from clients.
    }
  }

  @Override
  /** 清理断开的博物馆连接并通知其他在线用户。 */
  public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
    Participant participant = participants.remove(session.getId());
    if (participant == null) return;
    RoomState room = rooms.get(participant.roomId());
    if (room == null) return;
    room.sessions.remove(session);
    broadcast(room, Map.of("type", "leave", "id", participant.id()), session.getId());
    if (room.sessions.isEmpty()) {
      rooms.remove(room.id);
    }
  }

  @Override
  /** 传输异常时复用断开清理逻辑，及时把离线游客从同房间其他客户端中移除。 */
  public void handleTransportError(WebSocketSession session, Throwable exception) {
    /** 清理断开的博物馆连接并通知其他在线用户。 */
    afterConnectionClosed(session, CloseStatus.SERVER_ERROR);
  }

  /** 选择尚未满员的最早房间；都满员时创建新房间，控制每个博物馆实例的在线人数。 */
  private RoomState availableRoom() {
    return rooms.values().stream()
        .filter(room -> room.sessions.size() < ROOM_LIMIT)
        .min(Comparator.comparing(RoomState::createdAt))
        .orElseGet(() -> {
          String id = "museum-" + (rooms.size() + 1) + "-" + UUID.randomUUID().toString().substring(0, 5);
          RoomState room = new RoomState(id, Instant.now().toString());
          rooms.put(id, room);
          return room;
        });
  }

  /** 收集房间当前参与者快照，新连接进入时用它一次性生成已有游客头像。 */
  private List<Map<String, Object>> roomParticipants(RoomState room) {
    List<Map<String, Object>> result = new ArrayList<>();
    for (WebSocketSession session : room.sessions) {
      Participant participant = participants.get(session.getId());
      if (participant != null) {
        result.add(toPayload(participant));
      }
    }
    return result;
  }

  /** 将服务端参与者对象转成前端 presence 协议字段，包含位置、朝向和展示颜色。 */
  private Map<String, Object> toPayload(Participant participant) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("id", participant.id());
    payload.put("roomId", participant.roomId());
    payload.put("username", participant.username());
    payload.put("displayName", participant.displayName());
    payload.put("color", participant.color());
    payload.put("x", participant.x());
    payload.put("y", participant.y());
    payload.put("z", participant.z());
    payload.put("ry", participant.ry());
    payload.put("updatedAt", participant.updatedAt());
    return payload;
  }

  /** 向同一房间广播 join、pose、leave 消息，排除消息来源以减少重复回显。 */
  private void broadcast(RoomState room, Object payload, String exceptSessionId) {
    String json;
    try {
      json = objectMapper.writeValueAsString(payload);
    } catch (JsonProcessingException ex) {
      return;
    }
    for (WebSocketSession session : room.sessions) {
      if (session.getId().equals(exceptSessionId)) continue;
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

  /** 向单个 WebSocket 会话发送 JSON，例如首次连接时的 welcome 房间快照。 */
  private void send(WebSocketSession session, Object payload) {
    if (!session.isOpen()) return;
    try {
      session.sendMessage(new TextMessage(objectMapper.writeValueAsString(payload)));
    } catch (IOException ignored) {
      // Close callback cleans up.
    }
  }

  /** 根据用户名稳定分配头像颜色，让同一用户刷新页面后仍容易被其他访客识别。 */
  private String colorFor(String username) {
    String[] colors = {"#38bdf8", "#34d399", "#f59e0b", "#f472b6", "#a78bfa", "#fb7185", "#84cc16", "#60a5fa"};
    return colors[Math.floorMod(username.hashCode(), colors.length)];
  }

  /** 给未登录访客分配递增展示名，避免多人同时参观时全都显示匿名用户。 */
  private String nextGuestDisplayName() {
    Set<Integer> used = new HashSet<>();
    for (Participant participant : participants.values()) {
      String name = participant.displayName();
      if (name != null && name.matches("^游客\\d+$")) {
        try {
          used.add(Integer.parseInt(name.substring(2)));
        } catch (NumberFormatException ignored) {
          // Ignore malformed guest names from older sessions.
        }
      }
    }
    int next = 1;
    while (used.contains(next)) {
      next++;
    }
    return "游客" + next;
  }

  /** 限制客户端上报的坐标范围，防止头像被同步到博物馆墙体外或天空/地下。 */
  private double clamp(double value, double min, double max) {
    return Math.max(min, Math.min(max, value));
  }

  /** 解析 WebSocket URL 查询参数，主要用于读取可选的 JWT token。 */
  private Map<String, String> queryParams(URI uri) {
    Map<String, String> params = new LinkedHashMap<>();
    if (uri == null || uri.getQuery() == null) return params;
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

  /** 记录一个在线访客的 presence 状态，前端用这些字段渲染远端头像。 */
  private record Participant(
      String id,
      String roomId,
      String username,
      String displayName,
      String color,
      double x,
      double y,
      double z,
      double ry,
      String updatedAt
  ) {}

  private static final class RoomState {
    private final String id;
    private final String createdAt;
    private final CopyOnWriteArraySet<WebSocketSession> sessions = new CopyOnWriteArraySet<>();

    /** 创建一个博物馆在线房间，sessions 保存当前房间的 WebSocket 连接集合。 */
    private RoomState(String id, String createdAt) {
      this.id = id;
      this.createdAt = createdAt;
    }

    /** 返回房间创建时间，用于优先复用最早且未满员的房间。 */
    private String createdAt() {
      return createdAt;
    }
  }
}
