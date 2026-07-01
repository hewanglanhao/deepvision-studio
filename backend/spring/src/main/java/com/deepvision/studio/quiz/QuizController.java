package com.deepvision.studio.quiz;

import com.deepvision.studio.quiz.QuizDtos.QuizAnswerRequest;
import com.deepvision.studio.quiz.QuizDtos.QuizAnswerResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizDashboardResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizProfileResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizRecommendationResponse;
import com.deepvision.studio.quiz.QuizDtos.QuizQuestionResponse;
import jakarta.validation.Valid;
import java.security.Principal;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/quiz")
public class QuizController {
  private final QuizService quizService;

  public QuizController(QuizService quizService) {
    this.quizService = quizService;
  }

  @GetMapping("/profile")
  public QuizProfileResponse profile(Principal principal) {
    return quizService.profile(username(principal));
  }

  @GetMapping("/dashboard")
  public QuizDashboardResponse dashboard(Principal principal) {
    return quizService.dashboard(username(principal));
  }

  @GetMapping("/questions")
  public List<QuizQuestionResponse> questions() {
    return quizService.allQuestions();
  }

  @GetMapping("/recommendations")
  public QuizRecommendationResponse recommendations(
      Principal principal,
      @RequestParam(value = "mode", required = false, defaultValue = "weakness") String mode,
      @RequestParam(value = "limit", required = false, defaultValue = "10") int limit
  ) {
    return quizService.recommended(username(principal), mode, limit);
  }

  @PostMapping("/answers")
  public QuizAnswerResponse answer(Principal principal, @Valid @RequestBody QuizAnswerRequest request) {
    return quizService.answer(username(principal), request);
  }

  private String username(Principal principal) {
    return principal == null ? null : principal.getName();
  }
}
