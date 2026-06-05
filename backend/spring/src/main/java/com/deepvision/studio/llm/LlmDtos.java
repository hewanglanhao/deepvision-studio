package com.deepvision.studio.llm;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

public final class LlmDtos {
  /** DTO 容器类只定义 LLM 代理接口的数据结构，不参与实例化。 */
  private LlmDtos() {}

  @Schema(description = "LLM chat request")
  /** 前端助手提交的完整对话请求，可包含系统提示词、模型参数和多轮消息上下文。 */
  public record ChatRequest(
      @Schema(description = "Optional model override")
      String model,
      @Schema(description = "Optional reasoning effort value supported by the upstream model")
      String reasoningEffort,
      @Schema(description = "System prompt applied to this request")
      String systemPrompt,
      @Schema(description = "Conversation messages")
      @NotEmpty(message = "messages are required")
      List<ChatMessage> messages
  ) {}

  @Schema(description = "LLM chat message")
  /** 单条对话消息，role 区分用户/助手/系统，content 支持文本和图片混合输入。 */
  public record ChatMessage(
      @Schema(description = "Message role", example = "user")
      @NotBlank(message = "role is required")
      String role,
      @Schema(description = "Message content parts")
      @NotEmpty(message = "content is required")
      List<ContentPart> content
  ) {}

  @Schema(description = "Text or image content part")
  /** 多模态内容片段，A 模式可把网络结构说明文本和推理截图一起传给视觉语言模型。 */
  public record ContentPart(
      @Schema(description = "Content type", example = "text")
      @NotBlank(message = "type is required")
      String type,
      @Schema(description = "Text content when type is text")
      String text,
      @Schema(description = "Image URL or Data URL when type is image_url")
      String imageUrl
  ) {}

  @Schema(description = "LLM chat response")
  /** 后端统一返回的助手回答，保留模型名和请求 id 便于排查上游模型调用。 */
  public record ChatResponse(
      String content,
      String model,
      String id
  ) {}
}
