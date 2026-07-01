package com.deepvision.studio.quiz;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuizQuestionRepository extends JpaRepository<QuizQuestion, String> {
  List<QuizQuestion> findByActiveTrueOrderByTopicAscDifficultyAscCodeAsc();
}
