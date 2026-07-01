import { Injectable } from '@angular/core';
import { ApiClientService } from '@core/api/api-client.service';

export type QuizMode = 'weakness' | 'spaced' | 'exam';

export interface QuizAttemptSummary {
  questionCode: string;
  topic: string;
  difficulty: number;
  correct: boolean;
  answeredAt: string;
}

export interface QuizProfileResponse {
  scores: Record<string, number>;
  answeredCount: number;
  correctCount: number;
  accuracy: number;
  currentStreak: number;
  updatedAt: string;
  recentAttempts: QuizAttemptSummary[];
}

export interface QuizQuestionResponse {
  code: string;
  topic: string;
  difficulty: number;
  prompt: string;
  options: string[];
  recommendationReason: string;
}

export interface QuizRecommendationResponse {
  mode: QuizMode;
  strategyName: string;
  strategyDescription: string;
  questions: QuizQuestionResponse[];
}

export interface QuizAnswerResponse {
  questionCode: string;
  correct: boolean;
  selectedIndex: number;
  answerIndex: number;
  explanation: string;
  profile: QuizProfileResponse;
}

@Injectable({ providedIn: 'root' })
export class QuizApiService {
  constructor(private readonly api: ApiClientService) {}

  profile(): Promise<QuizProfileResponse> {
    return this.api.request<QuizProfileResponse>('/api/quiz/profile');
  }

  recommendations(mode: QuizMode, limit: number): Promise<QuizRecommendationResponse> {
    const query = `?mode=${encodeURIComponent(mode)}&limit=${encodeURIComponent(String(limit))}`;
    return this.api.request<QuizRecommendationResponse>(`/api/quiz/recommendations${query}`);
  }

  answer(questionCode: string, selectedIndex: number): Promise<QuizAnswerResponse> {
    return this.api.request<QuizAnswerResponse>('/api/quiz/answers', {
      method: 'POST',
      body: JSON.stringify({ questionCode, selectedIndex })
    });
  }
}
