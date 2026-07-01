package com.deepvision.studio.quiz;

import com.deepvision.studio.auth.AppUser;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuizUserProfileRepository extends JpaRepository<QuizUserProfile, Long> {
  Optional<QuizUserProfile> findByUser(AppUser user);
}
