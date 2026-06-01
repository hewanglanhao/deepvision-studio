package com.deepvision.studio.llm;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

public final class LlmDtos {
  private LlmDtos() {}

  @Schema(description = "LLM chat request")
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
  public record ChatMessage(
      @Schema(description = "Message role", example = "user")
      @NotBlank(message = "role is required")
      String role,
      @Schema(description = "Message content parts")
      @NotEmpty(message = "content is required")
      List<ContentPart> content
  ) {}

  @Schema(description = "Text or image content part")
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
  public record ChatResponse(
      String content,
      String model,
      String id
  ) {}
}
