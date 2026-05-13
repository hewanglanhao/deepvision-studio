import { Injectable } from '@angular/core';
import { ApiClientService } from '../../services/api-client.service';
import { LLM_CLIENT_CONFIG } from './llm.config';
import { LlmChatRequest, LlmChatResponse } from './llm.models';

@Injectable({ providedIn: 'root' })
export class LlmChatService {
  constructor(private api: ApiClientService) {}

  chat(request: LlmChatRequest): Promise<LlmChatResponse> {
    const body: Record<string, unknown> = {
      reasoningEffort: request.reasoningEffort ?? LLM_CLIENT_CONFIG.reasoningEffort,
      systemPrompt: request.systemPrompt,
      messages: request.messages
    };
    if (request.model?.trim()) {
      body['model'] = request.model.trim();
    }

    return this.api.request<LlmChatResponse>('/api/llm/chat', {
      method: 'POST',
      body: JSON.stringify(body)
    });
  }
}
