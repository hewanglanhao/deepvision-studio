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
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.Instant;

@Entity
@Table(
    name = "quiz_attempts",
    indexes = {
        @Index(name = "idx_quiz_attempt_user", columnList = "user_id"),
        @Index(name = "idx_quiz_attempt_question", columnList = "question_code")
    }
)
public class QuizAttempt {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long id;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "user_id", nullable = false)
  private AppUser user;

  @ManyToOne(fetch = FetchType.LAZY, optional = false)
  @JoinColumn(name = "question_code", nullable = false)
  private QuizQuestion question;

  @Column(nullable = false)
  private int selectedIndex;

  @Column(nullable = false)
  private boolean correct;

  @Column(nullable = false)
  private Instant answeredAt = Instant.now();

  protected QuizAttempt() {}

  public QuizAttempt(AppUser user, QuizQuestion question, int selectedIndex, boolean correct) {
    this.user = user;
    this.question = question;
    this.selectedIndex = selectedIndex;
    this.correct = correct;
  }

  public Long getId() { return id; }

  public AppUser getUser() { return user; }

  public QuizQuestion getQuestion() { return question; }

  public int getSelectedIndex() { return selectedIndex; }

  public boolean isCorrect() { return correct; }

  public Instant getAnsweredAt() { return answeredAt; }
}
