package com.deepvision.studio.quiz;

import com.deepvision.studio.auth.AppUser;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

@Entity
@Table(
    name = "quiz_user_profiles",
    indexes = @Index(name = "idx_quiz_profiles_user", columnList = "user_id", unique = true)
)
public class QuizUserProfile {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @OneToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "user_id", nullable = false, unique = true)
  private AppUser user;

  @Column(nullable = false)
  private double aiFoundations = 50;

  @Column(nullable = false)
  private double machineLearning = 50;

  @Column(nullable = false)
  private double neuralNetworks = 50;

  @Column(nullable = false)
  private double deepLearningTraining = 50;

  @Column(nullable = false)
  private double convolutionVision = 50;

  @Column(nullable = false)
  private double sequenceModels = 50;

  @Column(nullable = false)
  private double evaluationMetrics = 50;

  @Column(nullable = false)
  private double responsibleAi = 50;

  @Column(nullable = false)
  private int answeredCount = 0;

  @Column(nullable = false)
  private int correctCount = 0;

  @Column(nullable = false)
  private int currentStreak = 0;

  @Column(nullable = false)
  private Instant updatedAt = Instant.now();

  protected QuizUserProfile() {}

  public QuizUserProfile(AppUser user) {
    this.user = user;
  }

  public void applyAnswer(String topic, boolean correct, int difficulty) {
    double delta = correct ? 3.5 + difficulty * 0.8 : -(2.0 + difficulty * 0.5);
    setScore(topic, clamp(score(topic) + delta));
    answeredCount += 1;
    if (correct) {
      correctCount += 1;
      currentStreak += 1;
    } else {
      currentStreak = 0;
    }
    updatedAt = Instant.now();
  }

  public Map<String, Double> scores() {
    Map<String, Double> scores = new LinkedHashMap<>();
    scores.put("ai_foundations", aiFoundations);
    scores.put("machine_learning", machineLearning);
    scores.put("neural_networks", neuralNetworks);
    scores.put("deep_learning_training", deepLearningTraining);
    scores.put("convolution_vision", convolutionVision);
    scores.put("sequence_models", sequenceModels);
    scores.put("evaluation_metrics", evaluationMetrics);
    scores.put("responsible_ai", responsibleAi);
    return scores;
  }

  public double score(String topic) {
    return switch (topic) {
      case "ai_foundations" -> aiFoundations;
      case "machine_learning" -> machineLearning;
      case "neural_networks" -> neuralNetworks;
      case "deep_learning_training" -> deepLearningTraining;
      case "convolution_vision" -> convolutionVision;
      case "sequence_models" -> sequenceModels;
      case "evaluation_metrics" -> evaluationMetrics;
      case "responsible_ai" -> responsibleAi;
      default -> 50;
    };
  }

  private void setScore(String topic, double value) {
    switch (topic) {
      case "ai_foundations" -> aiFoundations = value;
      case "machine_learning" -> machineLearning = value;
      case "neural_networks" -> neuralNetworks = value;
      case "deep_learning_training" -> deepLearningTraining = value;
      case "convolution_vision" -> convolutionVision = value;
      case "sequence_models" -> sequenceModels = value;
      case "evaluation_metrics" -> evaluationMetrics = value;
      case "responsible_ai" -> responsibleAi = value;
      default -> {
      }
    }
  }

  private double clamp(double value) {
    return Math.max(0, Math.min(100, value));
  }

  public Long getId() { return id; }

  public AppUser getUser() { return user; }

  public int getAnsweredCount() { return answeredCount; }

  public int getCorrectCount() { return correctCount; }

  public int getCurrentStreak() { return currentStreak; }

  public Instant getUpdatedAt() { return updatedAt; }
}
