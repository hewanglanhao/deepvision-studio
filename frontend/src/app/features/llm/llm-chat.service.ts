import { Injectable } from '@angular/core';
import { ApiClientService } from '../../services/api-client.service';
import { LLM_CLIENT_CONFIG } from './llm.config';
import { LlmChatRequest, LlmChatResponse } from './llm.models';

@Injectable({ providedIn: 'root' })
export class LlmChatService {
  constructor(private api: ApiClientService) {}

  chat(request: LlmChatRequest): Promise<LlmChatResponse> {
    return this.api.request<LlmChatResponse>('/api/llm/chat', {
      method: 'POST',
      body: JSON.stringify({
        model: request.model ?? LLM_CLIENT_CONFIG.model,
        reasoningEffort: request.reasoningEffort ?? LLM_CLIENT_CONFIG.reasoningEffort,
        systemPrompt: request.systemPrompt,
        messages: request.messages
      })
    });
  }
}
