package com.deepvision.studio.quiz;

import com.deepvision.studio.auth.AppUser;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuizAttemptRepository extends JpaRepository<QuizAttempt, Long> {
  boolean existsByUserAndQuestion(AppUser user, QuizQuestion question);

  List<QuizAttempt> findTop20ByUserOrderByAnsweredAtDesc(AppUser user);

  List<QuizAttempt> findTop200ByUserOrderByAnsweredAtDesc(AppUser user);

  List<QuizAttempt> findByUserOrderByAnsweredAtDesc(AppUser user);
}
