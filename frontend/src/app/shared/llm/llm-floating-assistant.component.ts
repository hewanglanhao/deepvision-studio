import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DEFAULT_LLM_SYSTEM_PROMPT } from './llm-prompts';
import { LlmChatService } from './llm-chat.service';
import { LlmChatContext, LlmChatMessage, LlmContentPart } from './llm.models';

interface UiChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

export interface LlmQuickPrompt {
  label: string;
  question: string;
}

const DEFAULT_QUICK_PROMPTS: LlmQuickPrompt[] = [
  { label: '解释当前层', question: '请结合当前页面上下文，解释当前选中内容为什么会产生这样的结果。' },
  { label: '看关键特征', question: '请根据当前页面的图像、层输出和张量信息，说明模型更关注哪些特征。' },
  { label: '学习建议', question: '如果我要进一步理解这里的结果，下一步最值得关注哪些信息？' }
];

@Component({
  selector: 'app-llm-floating-assistant',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="llm-dock">
      @if (open) {
        <section class="llm-popover">
          <header class="llm-head">
            <div class="title-wrap">
              <div class="llm-title">{{ title }}</div>
              <div class="llm-subtitle">{{ includeContext ? '结合当前页面数据回答' : '普通问答模式' }}</div>
            </div>
            <button type="button" class="icon-btn" aria-label="关闭" (click)="open = false">×</button>
          </header>

          <div class="context-row">
            <label class="context-toggle">
              <input type="checkbox" [(ngModel)]="includeContext" (ngModelChange)="onContextToggle()" />
              <span>传入页面上下文</span>
            </label>
            @if (includeContext) {
              <button type="button" class="text-btn" (click)="refreshContext()">刷新</button>
            }
          </div>

          <div class="context-summary">{{ includeContext ? contextSummary : '当前不会附带页面上下文。' }}</div>

          <div class="chat-log">
            @if (!messages.length) {
              <div class="empty-tip">
                打开页面上下文后，你可以直接提问当前网络结构、层输出、预测结果和可视化内容。
              </div>
            }

            @for (message of messages; track $index) {
              <article [class]="'chat-bubble ' + message.role">
                @if (message.role === 'assistant') {
                  <div class="md-body" [innerHTML]="renderMarkdown(message.text)"></div>
                } @else {
                  <div class="plain-body">{{ message.text }}</div>
                }
              </article>
            }

            @if (busy) {
              <article class="chat-bubble assistant pending">正在分析...</article>
            }

            @if (error) {
              <div class="chat-error">{{ error }}</div>
            }
          </div>

          <div class="quick-row">
            @for (prompt of quickPrompts; track prompt.label) {
              <button type="button" [title]="prompt.question" (click)="askPreset(prompt.question)">
                {{ prompt.label }}
              </button>
            }
          </div>

          <form class="chat-input-row" (ngSubmit)="send()">
            <textarea
              name="llmQuestion"
              [(ngModel)]="draft"
              rows="2"
              placeholder="问当前页面，或输入任何和深度学习相关的问题..."
              [disabled]="busy"
            ></textarea>
            <button type="submit" [disabled]="busy || !draft.trim()">发送</button>
          </form>
        </section>
      }

      <button type="button" class="llm-fab" aria-label="打开 AI 助手" (click)="toggle()">AI</button>
    </div>
  `,
  styles: [`
    :host {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 1800;
      font-family: 'Manrope', 'Segoe UI', 'Noto Sans SC', sans-serif;
      font-size: 12px;
    }

    .llm-dock {
      position: relative;
    }

    .llm-fab {
      width: 46px;
      height: 46px;
      border: 1px solid #1d4ed8;
      border-radius: 50%;
      background: #2563eb;
      color: #fff;
      font-size: 13px;
      font-weight: 900;
      box-shadow: 0 10px 26px rgba(37, 99, 235, .32);
      cursor: pointer;
    }

    .llm-popover {
      position: absolute;
      right: 0;
      bottom: 56px;
      width: 460px;
      height: 620px;
      display: grid;
      grid-template-rows: 42px 32px 24px minmax(0, 1fr) 28px 56px;
      overflow: hidden;
      border: 1px solid #d7dde7;
      border-radius: 10px;
      background: #fff;
      color: #182132;
      box-shadow: 0 18px 46px rgba(15, 23, 42, .2);
    }

    .llm-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid #e6ebf2;
      background: #f8fafc;
    }

    .title-wrap {
      min-width: 0;
    }

    .llm-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #0f172a;
      font-size: 12px;
      font-weight: 800;
      line-height: 1.2;
    }

    .llm-subtitle {
      margin-top: 1px;
      color: #64748b;
      font-size: 10px;
      line-height: 1.2;
    }

    .icon-btn {
      width: 24px;
      height: 24px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      border: 1px solid #d7dde7;
      border-radius: 7px;
      background: #fff;
      color: #475569;
      font-size: 16px;
      line-height: 1;
      padding: 0;
      cursor: pointer;
    }

    .context-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 5px 10px;
      border-bottom: 1px solid #edf1f6;
      background: #fff;
    }

    .context-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      margin: 0;
      color: #334155;
      font-size: 11px;
      line-height: 1;
    }

    .context-toggle input {
      width: 13px;
      height: 13px;
      accent-color: #2563eb;
    }

    .text-btn {
      border: 0;
      background: transparent;
      color: #2563eb;
      font: inherit;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 0;
      cursor: pointer;
    }

    .context-summary {
      height: 24px;
      padding: 4px 10px;
      border-bottom: 1px solid #dbeafe;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 10px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .chat-log {
      min-height: 0;
      overflow-y: auto;
      display: grid;
      align-content: start;
      gap: 7px;
      padding: 10px 11px;
      background: #f8fafc;
    }

    .empty-tip {
      padding: 9px 10px;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      background: #fff;
      color: #64748b;
      font-size: 11px;
      line-height: 1.45;
    }

    .chat-bubble {
      max-width: 94%;
      overflow-wrap: anywhere;
      border: 1px solid #e1e7ef;
      border-radius: 9px;
      padding: 7px 9px;
      font-size: 11px;
      line-height: 1.48;
    }

    .chat-bubble.user {
      justify-self: end;
      border-color: #2563eb;
      background: #2563eb;
      color: #fff;
    }

    .chat-bubble.assistant {
      justify-self: start;
      background: #fff;
      color: #1f2937;
    }

    .chat-bubble.pending {
      color: #64748b;
    }

    .plain-body {
      white-space: pre-wrap;
    }

    .md-body {
      display: grid;
      gap: 5px;
    }

    .md-body :where(p, ul, ol, pre, h1, h2, h3) {
      margin: 0;
    }

    .md-body :where(h1, h2, h3) {
      color: #0f172a;
      font-size: 12px;
      line-height: 1.35;
    }

    .md-body :where(ul, ol) {
      padding-left: 16px;
    }

    .md-body li {
      margin: 2px 0;
    }

    .md-body code {
      padding: 1px 4px;
      border-radius: 4px;
      background: #eef2f7;
      color: #0f172a;
      font-family: Consolas, 'SFMono-Regular', monospace;
      font-size: 10px;
    }

    .md-body pre {
      overflow-x: auto;
      padding: 7px 8px;
      border: 1px solid #e1e7ef;
      border-radius: 7px;
      background: #f1f5f9;
    }

    .md-body pre code {
      padding: 0;
      background: transparent;
    }

    .chat-error {
      padding: 7px 8px;
      border: 1px solid #fca5a5;
      border-radius: 8px;
      background: #fef2f2;
      color: #b91c1c;
      font-size: 11px;
      line-height: 1.4;
    }

    .quick-row {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      border-top: 1px solid #e6ebf2;
      background: #fff;
      overflow-x: auto;
    }

    .quick-row button {
      flex: 0 0 auto;
      height: 18px;
      border: 0;
      border-radius: 999px;
      background: #f1f5f9;
      color: #64748b;
      font: inherit;
      font-size: 9px;
      font-weight: 600;
      padding: 0 7px;
      cursor: pointer;
    }

    .quick-row button:hover {
      color: #1d4ed8;
      background: #eff6ff;
    }

    .chat-input-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 48px;
      gap: 7px;
      padding: 8px 10px;
      border-top: 1px solid #e6ebf2;
      background: #fff;
    }

    .chat-input-row textarea {
      resize: none;
      min-height: 38px;
      max-height: 38px;
      padding: 7px 8px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      color: #0f172a;
      font: inherit;
      font-size: 11px;
      line-height: 1.35;
      outline: none;
    }

    .chat-input-row textarea:focus {
      border-color: #2563eb;
      box-shadow: 0 0 0 2px rgba(37, 99, 235, .12);
    }

    .chat-input-row button {
      border: 1px solid #2563eb;
      border-radius: 8px;
      background: #2563eb;
      color: #fff;
      font: inherit;
      font-size: 11px;
      font-weight: 800;
      cursor: pointer;
    }

    button:disabled,
    textarea:disabled {
      opacity: .5;
      cursor: not-allowed;
    }

    @media (max-width: 430px) {
      :host {
        right: 12px;
        bottom: 12px;
      }

      .llm-popover {
        width: calc(100vw - 24px);
        height: min(620px, calc(100vh - 82px));
      }
    }
  `]
})
export class LlmFloatingAssistantComponent {
  @Input() title = 'AI 学习助手';
  @Input() systemPrompt = DEFAULT_LLM_SYSTEM_PROMPT;
  @Input() contextProvider?: () => LlmChatContext;
  @Input() quickPrompts: LlmQuickPrompt[] = DEFAULT_QUICK_PROMPTS;

  open = false;
  includeContext = false;
  draft = '';
  busy = false;
  error = '';
  contextSummary = '';
  messages: UiChatMessage[] = [];

  constructor(private readonly llm: LlmChatService) {}

  toggle(): void {
    this.open = !this.open;
    if (this.open && this.includeContext) {
      this.refreshContext();
    }
  }

  onContextToggle(): void {
    if (this.includeContext) {
      this.refreshContext();
    } else {
      this.contextSummary = '';
    }
  }

  refreshContext(): void {
    const context = this.contextProvider?.();
    this.contextSummary = context
      ? `${context.text.length} 字上下文 · ${context.images.length} 张图`
      : '当前页面没有可传入的上下文。';
  }

  askPreset(question: string): void {
    this.draft = question;
    void this.send();
  }

  async send(): Promise<void> {
    const question = this.draft.trim();
    if (!question || this.busy) return;

    this.draft = '';
    this.error = '';
    this.busy = true;
    this.messages = [...this.messages, { role: 'user', text: question }];

    const context = this.includeContext ? this.contextProvider?.() : undefined;
    if (this.includeContext) {
      this.contextSummary = context
        ? `${context.text.length} 字上下文 · ${context.images.length} 张图`
        : '当前页面没有可传入的上下文。';
    }

    try {
      const apiMessages = this.toApiMessages(question, context);
      let assistantText = '';
      this.messages = [...this.messages, { role: 'assistant', text: '' }];
      const response = await this.llm.streamChat(
        {
          systemPrompt: this.systemPrompt,
          messages: apiMessages
        },
        delta => {
          assistantText += delta;
          this.messages = [
            ...this.messages.slice(0, -1),
            { role: 'assistant', text: assistantText }
          ];
        }
      );

      if (!assistantText) {
        this.messages = [
          ...this.messages.slice(0, -1),
          { role: 'assistant', text: response.content || '模型没有返回内容。' }
        ];
      }
    } catch (err) {
      this.error = err instanceof Error ? err.message : '大模型请求失败。';
    } finally {
      this.busy = false;
    }
  }

  renderMarkdown(markdown: string): string {
    const escaped = this.escapeHtml(markdown);
    const blocks = escaped.split(/```/);
    return blocks
      .map((block, index) => {
        if (index % 2 === 1) {
          return `<pre><code>${block.trim()}</code></pre>`;
        }
        return this.renderInlineMarkdownBlock(block);
      })
      .join('');
  }

  private renderInlineMarkdownBlock(block: string): string {
    const lines = block.split(/\r?\n/);
    const html: string[] = [];
    let listType: 'ul' | 'ol' | '' = '';

    const closeList = () => {
      if (listType) {
        html.push(`</${listType}>`);
        listType = '';
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        closeList();
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(3, heading[1].length);
        html.push(`<h${level}>${this.renderInline(heading[2])}</h${level}>`);
        continue;
      }

      const unordered = line.match(/^[-*]\s+(.+)$/);
      if (unordered) {
        if (listType !== 'ul') {
          closeList();
          html.push('<ul>');
          listType = 'ul';
        }
        html.push(`<li>${this.renderInline(unordered[1])}</li>`);
        continue;
      }

      const ordered = line.match(/^\d+\.\s+(.+)$/);
      if (ordered) {
        if (listType !== 'ol') {
          closeList();
          html.push('<ol>');
          listType = 'ol';
        }
        html.push(`<li>${this.renderInline(ordered[1])}</li>`);
        continue;
      }

      closeList();
      html.push(`<p>${this.renderInline(line)}</p>`);
    }

    closeList();
    return html.join('');
  }

  private renderInline(value: string): string {
    return value
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  }

  private escapeHtml(value: string): string {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  private toApiMessages(question: string, context?: LlmChatContext): LlmChatMessage[] {
    const history = this.messages.slice(-6, -1).map<LlmChatMessage>(message => ({
      role: message.role,
      content: [{ type: 'text', text: message.text }]
    }));

    const content: LlmContentPart[] = [];
    if (context) {
      content.push({
        type: 'text',
        text: ['下面是当前页面传入的数据上下文，请结合这些内容回答用户问题。', context.text].join('\n\n')
      });
      for (const image of context.images.slice(0, 4)) {
        content.push({ type: 'text', text: `图片：${image.title}` });
        content.push({ type: 'image_url', imageUrl: image.url });
      }
    }
    content.push({ type: 'text', text: question });

    return [...history, { role: 'user', content }];
  }
}
