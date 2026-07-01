package com.deepvision.studio.quiz;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import java.time.Instant;
import java.util.List;
import java.util.Map;

public final class QuizDtos {
  private QuizDtos() {}

  public record QuizProfileResponse(
      Map<String, Double> scores,
      int answeredCount,
      int correctCount,
      double accuracy,
      int currentStreak,
      Instant updatedAt,
      List<QuizAttemptSummary> recentAttempts
  ) {}

  public record QuizQuestionResponse(
      String code,
      String topic,
      int difficulty,
      String prompt,
      List<String> options,
      String recommendationReason
  ) {}

  public record QuizRecommendationResponse(
      String mode,
      String strategyName,
      String strategyDescription,
      List<QuizQuestionResponse> questions
  ) {}

  public record QuizAnswerRequest(
      @NotBlank(message = "Question code is required.")
      String questionCode,
      @Min(value = 0, message = "Selected option index is invalid.")
      @Max(value = 9, message = "Selected option index is invalid.")
      int selectedIndex
  ) {}

  public record QuizAnswerResponse(
      String questionCode,
      boolean correct,
      int selectedIndex,
      int answerIndex,
      String explanation,
      QuizProfileResponse profile
  ) {}

  public record QuizAttemptSummary(
      String questionCode,
      String topic,
      int difficulty,
      boolean correct,
      Instant answeredAt
  ) {}
}
