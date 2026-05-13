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
      body: JSON.stringify(this.requestBody(request))
    });
  }

  async streamChat(request: LlmChatRequest, onDelta: (delta: string) => void): Promise<LlmChatResponse> {
    const headers = new Headers({ 'Content-Type': 'application/json' });
    if (this.api.token) {
      headers.set('Authorization', `Bearer ${this.api.token}`);
    }

    const response = await fetch(`${this.api.baseUrl}/api/llm/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(this.requestBody(request))
    });
    if (!response.ok) {
      throw new Error(await this.errorMessage(response));
    }
    if (!response.body) {
      throw new Error('当前浏览器不支持流式响应。');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';
    let doneResponse: LlmChatResponse | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? '';
      for (const eventText of events) {
        const event = this.parseSseEvent(eventText);
        if (!event.data) continue;
        if (event.name === 'delta') {
          const delta = this.parseData<{ text?: string }>(event.data).text ?? '';
          if (delta) {
            fullText += delta;
            onDelta(delta);
          }
        } else if (event.name === 'done') {
          doneResponse = this.parseData<LlmChatResponse>(event.data);
        } else if (event.name === 'error') {
          throw new Error(this.parseData<{ message?: string }>(event.data).message ?? 'LLM stream failed.');
        }
      }
    }

    if (buffer.trim()) {
      const event = this.parseSseEvent(buffer);
      if (event.name === 'done' && event.data) {
        doneResponse = this.parseData<LlmChatResponse>(event.data);
      }
    }

    return doneResponse ?? { content: fullText, model: '', id: '' };
  }

  private requestBody(request: LlmChatRequest): Record<string, unknown> {
    const body: Record<string, unknown> = {
      reasoningEffort: request.reasoningEffort ?? LLM_CLIENT_CONFIG.reasoningEffort,
      systemPrompt: request.systemPrompt,
      messages: request.messages
    };
    if (request.model?.trim()) {
      body['model'] = request.model.trim();
    }
    return body;
  }

  private parseSseEvent(raw: string): { name: string; data: string } {
    let name = 'message';
    const data: string[] = [];
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith('event:')) {
        name = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        data.push(line.slice(5).trimStart());
      }
    }
    return { name, data: data.join('\n') };
  }

  private parseData<T>(data: string): T {
    try {
      return JSON.parse(data) as T;
    } catch {
      return {} as T;
    }
  }

  private async errorMessage(response: Response): Promise<string> {
    try {
      const body = await response.json();
      return body?.message ?? body?.error ?? `HTTP ${response.status}`;
    } catch {
      return await response.text() || `HTTP ${response.status}`;
    }
  }
}
