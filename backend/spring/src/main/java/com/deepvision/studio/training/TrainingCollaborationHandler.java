package com.deepvision.studio.training;

import com.deepvision.studio.auth.AppUser;
import com.deepvision.studio.auth.AppUserRepository;
import com.deepvision.studio.auth.JwtService;
import com.deepvision.studio.llm.LlmChatClient;
import com.deepvision.studio.llm.LlmDtos.ChatMessage;
import com.deepvision.studio.llm.LlmDtos.ChatRequest;
import com.deepvision.studio.llm.LlmDtos.ContentPart;
import com.deepvision.studio.training.TrainingDtos.CollaborationRoomSummary;
import com.deepvision.studio.training.TrainingDtos.TrainingStatusResponse;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.annotation.PreDestroy;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

@Component
public class TrainingCollaborationHandler extends TextWebSocketHandler {
  private static final int MAX_RECENT_MESSAGES = 60;
  private static final String ASSISTANT_NAME = "智能助手";
  private static final String ASSISTANT_USERNAME = "robot-assistant";

  private final TrainingJobService jobService;
  private final LlmChatClient llmChatClient;
  private final JwtService jwtService;
  private final AppUserRepository users;
  private final ObjectMapper objectMapper;
  private final ExecutorService assistantExecutor = Executors.newCachedThreadPool();
  private final Map<String, RoomState> rooms = new ConcurrentHashMap<>();
  private final Map<String, Participant> participants = new ConcurrentHashMap<>();

  public TrainingCollaborationHandler(
      TrainingJobService jobService,
      LlmChatClient llmChatClient,
      JwtService jwtService,
      AppUserRepository users,
      ObjectMapper objectMapper
  ) {
    this.jobService = jobService;
    this.llmChatClient = llmChatClient;
    this.jwtService = jwtService;
    this.users = users;
    this.objectMapper = objectMapper;
  }

  // 返回当前仍有成员在线的训练协作房间摘要。
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

  // 校验某个 clientId 是否已经加入指定训练房间，供旁观训练流鉴权。
  public boolean hasParticipant(String jobId, String clientId) {
    if (jobId == null || clientId == null || clientId.isBlank()) {
      return false;
    }
    return participants.values().stream()
        .anyMatch(participant -> jobId.equals(participant.jobId()) && clientId.equals(participant.clientId()));
  }

  @Override
  // 建立聊天室连接：校验 jobId、房间是否存在，并向新成员补发历史消息。
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
  // 处理客户端聊天消息，写入房间历史并在提到智能助手时触发 LLM 回复。
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
      if (mentionsAssistant(text)) {
        requestAssistantReply(participant, text, room.recentMessages());
      }
    } catch (JsonProcessingException ignored) {
      // Ignore malformed client messages.
    }
  }

  @Override
  // 成员离开时更新在线列表；如果房间没人则销毁房间状态。
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
  // WebSocket 异常时复用关闭逻辑清理成员状态。
  public void handleTransportError(WebSocketSession session, Throwable exception) {
    afterConnectionClosed(session, CloseStatus.SERVER_ERROR);
  }

  @PreDestroy
  void shutdown() {
    assistantExecutor.shutdownNow();
  }

  // 根据 JWT 或访客名称识别聊天室参与者，并保留 clientId 供旁观流校验。
  private Participant identifyParticipant(WebSocketSession session, Map<String, String> query, String jobId) {
    String clientId = query.getOrDefault("clientId", "").trim();
    if (clientId.isBlank()) {
      clientId = session.getId();
    }
    String token = query.getOrDefault("token", "");
    if (!token.isBlank()) {
      try {
        String username = jwtService.subject(token);
        AppUser user = users.findByUsername(username).orElse(null);
        if (user != null) {
          return new Participant(jobId, user.getUsername(), user.getDisplayName(), clientId);
        }
      } catch (RuntimeException ignored) {
        // Fall back to guest identity.
      }
    }
    String guestName = query.getOrDefault("name", "").trim();
    if (guestName.isBlank()) {
      guestName = "访客-" + session.getId().substring(0, Math.min(4, session.getId().length()));
    }
    return new Participant(jobId, "guest-" + session.getId(), guestName, clientId);
  }

  // 广播当前房间在线成员列表，并把智能助手作为固定成员展示。
  private void broadcastPresence(String jobId) {
    RoomState room = rooms.get(jobId);
    if (room == null) {
      return;
    }
    List<Map<String, String>> activeUsers = new ArrayList<>(room.sessions.stream()
        .map(session -> participants.get(session.getId()))
        .filter(participant -> participant != null)
        .map(participant -> Map.of(
            "username", participant.username(),
            "displayName", participant.displayName()
        ))
        .distinct()
        .toList());
    activeUsers.add(Map.of(
        "username", ASSISTANT_USERNAME,
        "displayName", ASSISTANT_NAME
    ));
    broadcast(jobId, Map.of(
        "type", "presence",
        "jobId", jobId,
        "users", activeUsers
    ));
  }

  // 向房间广播系统提示，例如成员加入或离开。
  private void broadcastSystem(String jobId, String text) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("type", "system");
    payload.put("jobId", jobId);
    payload.put("text", text);
    payload.put("createdAt", Instant.now().toString());
    broadcast(jobId, payload);
  }

  // 调用 LLM 流式生成智能助手回复，并将增量内容广播到房间。
  private void requestAssistantReply(Participant requester, String rawText, List<Map<String, Object>> recentMessages) {
    String question = stripAssistantMention(rawText);
    if (question.isBlank()) {
      broadcastAssistant(requester.jobId(), "你可以在 @智能助手 后面直接写问题，例如：@智能助手 现在训练损失下降正常吗？");
      return;
    }
    String messageId = UUID.randomUUID().toString();
    broadcastAssistantStart(requester.jobId(), messageId);
    assistantExecutor.submit(() -> {
      StringBuilder answer = new StringBuilder();
      try {
        TrainingStatusResponse status = jobService.status(requester.jobId());
        llmChatClient.stream(new ChatRequest(
            null,
            "medium",
            assistantSystemPrompt(status),
            List.of(new ChatMessage(
                "user",
                List.of(new ContentPart("text", assistantUserPrompt(requester, question, recentMessages), null))
            ))
        ), (event, delta) -> {
          if (!"delta".equals(event)) {
            return;
          }
          answer.append(delta);
          broadcastAssistantUpdate(requester.jobId(), messageId, answer.toString(), true);
        });
        String finalText = answer.isEmpty() ? "我没有生成有效回答，可以换个问法再试一次。" : answer.toString();
        broadcastAssistantUpdate(requester.jobId(), messageId, finalText, false);
      } catch (RuntimeException ex) {
        broadcastAssistantUpdate(requester.jobId(), messageId, "智能助手暂时无法回答：" + ex.getMessage(), false);
      }
    });
  }

  // 发送一条完整的智能助手普通消息。
  private void broadcastAssistant(String jobId, String text) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("type", "chat");
    payload.put("id", UUID.randomUUID().toString());
    payload.put("jobId", jobId);
    payload.put("username", ASSISTANT_USERNAME);
    payload.put("displayName", ASSISTANT_NAME);
    payload.put("text", text);
    payload.put("createdAt", Instant.now().toString());
    RoomState room = rooms.get(jobId);
    if (room != null) {
      room.addMessage(payload);
    }
    broadcast(jobId, payload);
  }

  // 先广播一条空的助手消息，占位用于后续流式更新。
  private void broadcastAssistantStart(String jobId, String messageId) {
    Map<String, Object> payload = assistantPayload(jobId, messageId, "", true);
    RoomState room = rooms.get(jobId);
    if (room != null) {
      room.addMessage(payload);
    }
    broadcast(jobId, payload);
  }

  // 更新指定助手消息的流式内容和 streaming 状态。
  private void broadcastAssistantUpdate(String jobId, String messageId, String text, boolean streaming) {
    RoomState room = rooms.get(jobId);
    if (room != null) {
      room.updateMessage(messageId, text, streaming);
    }
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("type", "chat_update");
    payload.put("id", messageId);
    payload.put("jobId", jobId);
    payload.put("text", text);
    payload.put("streaming", streaming);
    payload.put("createdAt", Instant.now().toString());
    broadcast(jobId, payload);
  }

  // 构造智能助手消息载荷，统一普通消息和流式占位消息格式。
  private Map<String, Object> assistantPayload(String jobId, String messageId, String text, boolean streaming) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("type", "chat");
    payload.put("id", messageId);
    payload.put("jobId", jobId);
    payload.put("username", ASSISTANT_USERNAME);
    payload.put("displayName", ASSISTANT_NAME);
    payload.put("text", text);
    payload.put("streaming", streaming);
    payload.put("createdAt", Instant.now().toString());
    return payload;
  }

  // 判断聊天文本是否显式 @智能助手。
  private boolean mentionsAssistant(String text) {
    return text.startsWith("@" + ASSISTANT_NAME) || text.contains("@" + ASSISTANT_NAME + " ");
  }

  // 去掉 @智能助手 前缀，提取真正要发给 LLM 的问题。
  private String stripAssistantMention(String text) {
    return text.replace("@" + ASSISTANT_NAME, "").trim();
  }

  // 构造智能助手系统提示词，把当前训练状态注入回答上下文。
  private String assistantSystemPrompt(TrainingStatusResponse status) {
    return """
        你是深度学习演示平台训练聊天室中的智能助手。
        你的任务是帮助学生和教师理解当前训练状态、排查模型训练问题、解释日志和指标。
        回答请使用中文，简洁、具体，优先结合给定训练状态，不要编造不存在的日志或指标。
        当前训练状态：
        - jobId: %s
        - status: %s
        - epoch: %d / %d
        - batch: %d / %d
        - latestLoss: %.6f
        - latestValLoss: %s
        - latestAccuracy: %.6f
        - latestValAccuracy: %s
        - elapsedSeconds: %d
        - etaSeconds: %d
        """.formatted(
        status.jobId(),
        status.status(),
        status.epoch(),
        status.totalEpochs(),
        status.batch(),
        status.totalBatches(),
        status.latestLoss(),
        status.latestValLoss() == null ? "N/A" : String.format(java.util.Locale.US, "%.6f", status.latestValLoss()),
        status.latestAccuracy(),
        status.latestValAccuracy() == null ? "N/A" : String.format(java.util.Locale.US, "%.6f", status.latestValAccuracy()),
        status.elapsedSeconds(),
        status.etaSeconds()
    );
  }

  // 构造用户提示词，包含提问者、问题和最近聊天室上下文。
  private String assistantUserPrompt(Participant requester, String question, List<Map<String, Object>> recentMessages) {
    List<String> context = recentMessages.stream()
        .filter(message -> "chat".equals(String.valueOf(message.get("type"))))
        .skip(Math.max(0, recentMessages.size() - 10))
        .map(message -> String.valueOf(message.getOrDefault("displayName", message.getOrDefault("username", "成员")))
            + ": " + String.valueOf(message.getOrDefault("text", "")))
        .toList();
    return "提问者：" + requester.displayName()
        + "\n问题：" + question
        + "\n\n最近聊天室上下文：\n" + (context.isEmpty() ? "无" : String.join("\n", context));
  }

  // 将消息序列化后广播给房间内所有仍然打开的 WebSocket session。
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

  // 向单个 session 发送 JSON 消息，主要用于新成员补发历史记录。
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

  // 解析 WebSocket 连接 URI 中的 query 参数。
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

  private record Participant(String jobId, String username, String displayName, String clientId) {}

  private static final class RoomState {
    private final CopyOnWriteArraySet<WebSocketSession> sessions = new CopyOnWriteArraySet<>();
    private final ArrayDeque<Map<String, Object>> recentMessages = new ArrayDeque<>();
    private final Instant createdAt = Instant.now();

    // 添加房间历史消息，并限制缓存条数避免内存无限增长。
    private synchronized void addMessage(Map<String, Object> message) {
      recentMessages.addLast(message);
      while (recentMessages.size() > MAX_RECENT_MESSAGES) {
        recentMessages.removeFirst();
      }
    }

    // 返回房间历史消息快照，用于新成员加入时补发。
    private synchronized List<Map<String, Object>> recentMessages() {
      return new ArrayList<>(recentMessages);
    }

    // 根据消息 ID 更新助手流式回复内容。
    private synchronized void updateMessage(String id, String text, boolean streaming) {
      for (Map<String, Object> message : recentMessages) {
        if (id.equals(String.valueOf(message.get("id")))) {
          message.put("text", text);
          message.put("streaming", streaming);
          return;
        }
      }
    }
  }
}
