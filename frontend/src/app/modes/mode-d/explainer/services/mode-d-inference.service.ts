import { Injectable } from '@angular/core';
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { ModeDInferenceResult, ModeDPhaseZeroProbeResult, ModeDTokenScore } from '../models/mode-d.types';

interface BrowserTokenizer {
  encode(input: string): number[];
  decode(ids: number[]): string | string[];
}

interface BrowserTransformersModule {
  AutoTokenizer: {
    from_pretrained(modelId: string): Promise<BrowserTokenizer>;
  };
  env?: {
    allowRemoteModels?: boolean;
    allowLocalModels?: boolean;
    localModelPath?: string;
  };
}

type OrtTensorLike = {
  data?: ArrayLike<number>;
  cpuData?: ArrayLike<number>;
  dims: readonly number[];
};

@Injectable({ providedIn: 'root' })
export class ModeDInferenceService {
  private readonly assetBase = '/mode-d-assets';
  private tokenizerPromise: Promise<BrowserTokenizer> | null = null;
  private sessionPromise: Promise<InferenceSession> | null = null;
  private runtimeModulePromise: Promise<BrowserTransformersModule> | null = null;
  private readonly runtimeImporter = new Function(
    'path',
    'return import(path);'
  ) as (path: string) => Promise<BrowserTransformersModule>;

  async probeRuntime(inputText = 'transformer attention explains context'): Promise<ModeDPhaseZeroProbeResult> {
    const tokenizer = await this.loadTokenizer();
    const session = await this.loadSession();
    const encoded = this.encodeText(tokenizer, inputText);
    const tokenTexts = this.decodeTokens(tokenizer, encoded);

    const ort = await import('onnxruntime-web');
    const inputTensor = new ort.Tensor('int64', encoded, [1, encoded.length]) as Tensor;
    const results = await session.run({ input: inputTensor });
    const logits = results['linear_output']?.data;
    const attentionKeys = Object.keys(results).filter(key => key.includes('_attn'));

    return {
      tokenizerReady: true,
      sessionReady: true,
      logitsReady: Array.isArray(logits) ? logits.length > 0 : !!logits?.length,
      attentionKeys,
      tokenIds: encoded,
      tokenTexts
    };
  }

  async runInference(inputText: string, topK = 10): Promise<ModeDInferenceResult> {
    const tokenizer = await this.loadTokenizer();
    const session = await this.loadSession();
    const tokenIds = this.encodeText(tokenizer, inputText);
    const tokenTexts = this.decodeTokens(tokenizer, tokenIds);

    const ort = await import('onnxruntime-web');
    const inputTensor = new ort.Tensor('int64', tokenIds, [1, tokenIds.length]) as Tensor;
    const results = await session.run({ input: inputTensor });

    const logitsTensor = results['linear_output'] as OrtTensorLike | undefined;
    const logits = logitsTensor?.data ? Array.from(logitsTensor.data as ArrayLike<number>) : [];
    if (logits.length === 0) {
      throw new Error('未能从 Transformer 模型输出中提取 logits。');
    }

    const attentionEntries = Object.entries(results).filter(([key]) => key.endsWith('_attn_dropout'));
    const attentionByKey = Object.fromEntries(
      attentionEntries.map(([key, value]) => [key, this.extractAttentionMatrix(value as OrtTensorLike)])
    );

    return {
      tokenIds,
      tokenTexts,
      topK: this.buildTopK(logits, tokenizer, topK),
      attentionByKey,
      attentionKeys: attentionEntries.map(([key]) => key)
    };
  }

  private async loadTokenizer(): Promise<BrowserTokenizer> {
    if (!this.tokenizerPromise) {
      this.tokenizerPromise = (async () => {
        const runtime = await this.loadTransformersModule();
        if (!runtime?.AutoTokenizer) {
          throw new Error('未能加载浏览器端 Transformers 运行时。');
        }

        if (runtime.env) {
          runtime.env.allowRemoteModels = true;
          runtime.env.allowLocalModels = true;
          runtime.env.localModelPath = `${this.assetBase}/models/`;
        }

        try {
          return await runtime.AutoTokenizer.from_pretrained('Xenova/gpt2');
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(
            `Tokenizer 初始化失败。当前运行时会尝试加载 Xenova/gpt2 的 tokenizer 资源；如果浏览器无法访问外网，或本地 tokenizer 文件尚未提供，就会失败。原始错误：${message}`
          );
        }
      })();
    }
    return this.tokenizerPromise;
  }

  private async loadSession(): Promise<InferenceSession> {
    if (!this.sessionPromise) {
      this.sessionPromise = (async () => {
        const ort = await import('onnxruntime-web');
        ort.env.wasm.wasmPaths = {
          mjs: `${this.assetBase}/vendor/onnxruntime/ort-wasm-simd-threaded.jsep.js`,
          wasm: `${this.assetBase}/vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm`
        };
        ort.env.logLevel = 'error';

        const chunkCount = 63;
        const chunkUrls = Array.from(
          { length: chunkCount },
          (_, index) => `${this.assetBase}/model-v2/gpt2.onnx.part${index}`
        );
        const mergedArray = await this.fetchAndMergeChunks(chunkUrls);
        const blob = new Blob([mergedArray], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        try {
          return await ort.InferenceSession.create(url);
        } finally {
          URL.revokeObjectURL(url);
        }
      })();
    }
    return this.sessionPromise;
  }

  private encodeText(tokenizer: BrowserTokenizer, inputText: string): number[] {
    return tokenizer.encode(inputText.trim() || ' ');
  }

  private decodeTokens(tokenizer: BrowserTokenizer, tokenIds: number[]): string[] {
    return tokenIds.map(id => tokenizer.decode([id])).flat().map(token => this.formatTokenForDisplay(token));
  }

  private buildTopK(logits: number[], tokenizer: BrowserTokenizer, topK: number): ModeDTokenScore[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExpLogits = expLogits.reduce((sum, value) => sum + value, 0);

    return logits
      .map((logit, tokenId) => ({
        tokenId,
        token: this.formatTokenForDisplay(tokenizer.decode([tokenId])),
        probability: expLogits[tokenId]! / sumExpLogits,
        logit
      }))
      .sort((a, b) => b.logit - a.logit)
      .slice(0, topK)
      .map((item, index) => ({
        tokenId: item.tokenId,
        token: item.token,
        probability: item.probability,
        rank: index + 1
      }));
  }

  private extractAttentionMatrix(tensor: OrtTensorLike): number[][] {
    const raw = Array.from((tensor.cpuData ?? tensor.data ?? []) as ArrayLike<number>);
    const rows = tensor.dims[tensor.dims.length - 2] ?? 0;
    const cols = tensor.dims[tensor.dims.length - 1] ?? 0;

    if (!rows || !cols || raw.length < rows * cols) {
      return [];
    }

    const offset = raw.length - rows * cols;
    return Array.from({ length: rows }, (_, row) =>
      raw.slice(offset + row * cols, offset + (row + 1) * cols)
    );
  }

  private formatTokenForDisplay(token: string | string[]): string {
    const normalized = Array.isArray(token) ? token.join('') : token;
    return normalized
      .replace(/\n/g, '[换行]')
      .replace(/\t/g, '[制表]')
      .replace(/\r/g, '[回车]')
      .replace(/^ +$/g, match => `[${match.length} 个空格]`);
  }

  private async fetchAndMergeChunks(urls: string[]): Promise<ArrayBuffer> {
    const modelBuffers = await Promise.all(urls.map(url => this.fetchArrayBuffer(url)));
    const totalSize = modelBuffers.reduce((acc, chunk) => acc + chunk.byteLength, 0);
    const mergedArray = new Uint8Array(totalSize);
    let offset = 0;

    for (const chunk of modelBuffers) {
      mergedArray.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    return mergedArray.buffer;
  }

  private async fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`无法加载模型分片：${url}`);
    }
    return response.arrayBuffer();
  }

  private async loadTransformersModule(): Promise<BrowserTransformersModule> {
    if (!this.runtimeModulePromise) {
      this.runtimeModulePromise = this.runtimeImporter(`${this.assetBase}/vendor/transformers/transformers.min.js`);
    }
    return this.runtimeModulePromise;
  }
}
