package com.deepvision.studio.llm;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

public final class LlmDtos {
  private LlmDtos() {}

  public record ChatRequest(
      String model,
      String reasoningEffort,
      String systemPrompt,
      @NotEmpty(message = "messages are required")
      List<ChatMessage> messages
  ) {}

  public record ChatMessage(
      @NotBlank(message = "role is required")
      String role,
      @NotEmpty(message = "content is required")
      List<ContentPart> content
  ) {}

  public record ContentPart(
      @NotBlank(message = "type is required")
      String type,
      String text,
      String imageUrl
  ) {}

  public record ChatResponse(
      String content,
      String model,
      String id
  ) {}
}
