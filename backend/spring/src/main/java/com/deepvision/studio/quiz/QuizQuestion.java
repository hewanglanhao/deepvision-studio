package com.deepvision.studio.quiz;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(name = "quiz_questions")
public class QuizQuestion {
  @Id
  @Column(length = 64)
  private String code;

  @Column(nullable = false, length = 40)
  private String topic;

  @Column(nullable = false)
  private int difficulty;

  @Column(nullable = false, length = 500)
  private String prompt;

  @Column(nullable = false, length = 2000)
  private String optionsJson;

  @Column(nullable = false)
  private int answerIndex;

  @Column(nullable = false, length = 800)
  private String explanation;

  @Column(nullable = false)
  private boolean active = true;

  @Column(nullable = false)
  private Instant updatedAt = Instant.now();

  protected QuizQuestion() {}

  public QuizQuestion(String code, String topic, int difficulty, String prompt, String optionsJson, int answerIndex, String explanation) {
    this.code = code;
    this.topic = topic;
    this.difficulty = difficulty;
    this.prompt = prompt;
    this.optionsJson = optionsJson;
    this.answerIndex = answerIndex;
    this.explanation = explanation;
  }

  public void replaceWith(QuizQuestion other) {
    this.topic = other.topic;
    this.difficulty = other.difficulty;
    this.prompt = other.prompt;
    this.optionsJson = other.optionsJson;
    this.answerIndex = other.answerIndex;
    this.explanation = other.explanation;
    this.active = true;
    this.updatedAt = Instant.now();
  }

  public String getCode() { return code; }

  public String getTopic() { return topic; }

  public int getDifficulty() { return difficulty; }

  public String getPrompt() { return prompt; }

  public String getOptionsJson() { return optionsJson; }

  public int getAnswerIndex() { return answerIndex; }

  public String getExplanation() { return explanation; }

  public boolean isActive() { return active; }

  public Instant getUpdatedAt() { return updatedAt; }
}
