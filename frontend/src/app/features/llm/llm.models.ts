export type LlmContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; imageUrl: string };

export interface LlmChatMessage {
  role: 'user' | 'assistant';
  content: LlmContentPart[];
}

export interface LlmChatContext {
  text: string;
  images: Array<{ title: string; url: string }>;
}

export interface LlmChatRequest {
  model?: string;
  reasoningEffort?: 'low' | 'medium' | 'high';
  systemPrompt: string;
  messages: LlmChatMessage[];
}

export interface LlmChatResponse {
  content: string;
  model: string;
  id: string;
}
