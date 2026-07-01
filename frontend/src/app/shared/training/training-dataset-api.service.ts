import { Injectable, OnDestroy } from '@angular/core';
import { TrainingDatasetDetail, TrainingDatasetOption } from '@shared/simulation/sim-models';
import { ApiClientService } from '@core/api/api-client.service';

export interface DatasetImportResponse {
  datasetId: string;
  detail: TrainingDatasetDetail;
}

@Injectable({ providedIn: 'root' })
export class TrainingDatasetApiService implements OnDestroy {
  private readonly basePath = '/api/training/datasets';
  private readonly privatePreviewUrls = new Map<string, Promise<string>>();
  private previewToken = '';

  constructor(private api: ApiClientService) {}

  // 查询训练数据集列表，可按 source 过滤内置或上传数据集。
  async listDatasets(source?: string, signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    const query = source ? `?source=${encodeURIComponent(source)}` : '';
    const response = await fetch(`${this.api.baseUrl}${this.basePath}${query}`, {
      headers: this.authHeaders(),
      signal
    });
    return this.readJson<TrainingDatasetOption[]>(response);
  }

  // 查询内置训练数据集，供 B 端默认数据集选择区使用。
  async listBuiltinDatasets(signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    return this.listDatasets('builtin', signal);
  }

  // 获取数据集完整详情，并把私有预览图片 URL 转换为浏览器可展示的 object URL。
  async getDatasetDetail(datasetId: string, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, {
      headers: this.authHeaders(),
      signal
    });
    return this.normalizeDatasetDetail(await this.readJson<TrainingDatasetDetail>(response), signal);
  }

  // 上传 CSV、图片或 ZIP 数据集，并把后端返回的预览资源做本地规范化。
  async importDataset(files: File[], labelColumn?: string, classCount?: number, signal?: AbortSignal): Promise<DatasetImportResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }
    if (labelColumn) {
      form.append('labelColumn', labelColumn);
    }
    if (typeof classCount === 'number' && Number.isFinite(classCount)) {
      form.append('classCount', String(classCount));
    }

    const response = await fetch(`${this.api.baseUrl}${this.basePath}/imports`, {
      method: 'POST',
      headers: this.authHeaders(),
      body: form,
      signal
    });
    const result = await this.readJson<DatasetImportResponse>(response);
    return { ...result, detail: await this.normalizeDatasetDetail(result.detail, signal) };
  }

  // 删除当前用户上传的数据集，同时释放该数据集对应的预览 object URL。
  async deleteDataset(datasetId: string, signal?: AbortSignal): Promise<void> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, {
      method: 'DELETE',
      headers: this.authHeaders(),
      signal
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    this.releaseDatasetPreviewUrls(datasetId);
  }

  // 服务销毁时释放所有本地预览 URL，避免图片 Blob 长期占用内存。
  ngOnDestroy(): void {
    this.releaseAllPreviewUrls();
  }

  // 统一修正数据集详情中的图片预览地址，兼容公有静态资源和私有上传资源。
  private async normalizeDatasetDetail(detail: TrainingDatasetDetail, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    this.ensurePreviewCacheOwner();
    return {
      ...detail,
      imagePreview: await Promise.all((detail.imagePreview ?? []).map(async item => ({
        ...item,
        url: await this.resolvePreviewUrl(item.url, signal)
      })))
    };
  }

  // 根据 URL 类型决定直接返回静态路径，还是拉取鉴权资源并生成 object URL。
  private async resolvePreviewUrl(url: string, signal?: AbortSignal): Promise<string> {
    const normalizedUrl = this.normalizeResourceUrl(url);
    if (!this.isPrivateDatasetFileUrl(normalizedUrl)) {
      return normalizedUrl;
    }

    let cached = this.privatePreviewUrls.get(normalizedUrl);
    if (!cached) {
      cached = this.fetchPrivatePreview(normalizedUrl, signal);
      this.privatePreviewUrls.set(normalizedUrl, cached);
    }
    try {
      return await cached;
    } catch {
      this.privatePreviewUrls.delete(normalizedUrl);
      return normalizedUrl;
    }
  }

  // 携带 JWT 拉取上传数据集的私有图片预览，并缓存成 object URL。
  private async fetchPrivatePreview(url: string, signal?: AbortSignal): Promise<string> {
    const response = await fetch(url, {
      headers: this.authHeaders(),
      signal
    });
    if (!response.ok) {
      throw new Error(`Preview HTTP ${response.status}`);
    }
    return URL.createObjectURL(await response.blob());
  }

  // 判断预览地址是否属于需要后端鉴权读取的上传数据集文件。
  private isPrivateDatasetFileUrl(url: string): boolean {
    try {
      const path = new URL(url, window.location.origin).pathname;
      return path.startsWith(`${this.basePath}/`) && path.includes('/files/');
    } catch {
      return url.includes('/api/training/datasets/') && url.includes('/files/');
    }
  }

  // 登录用户变化时清空预览缓存，避免不同用户之间复用私有图片 URL。
  private ensurePreviewCacheOwner(): void {
    const token = this.api.token;
    if (token === this.previewToken) return;
    this.releaseAllPreviewUrls();
    this.previewToken = token;
  }

  // 释放指定数据集的私有预览 object URL。
  private releaseDatasetPreviewUrls(datasetId: string): void {
    const encodedId = encodeURIComponent(datasetId);
    for (const [url, preview] of this.privatePreviewUrls) {
      if (!url.includes(`/datasets/${encodedId}/`)) continue;
      void preview.then(value => URL.revokeObjectURL(value)).catch(() => undefined);
      this.privatePreviewUrls.delete(url);
    }
  }

  // 释放当前服务缓存的全部私有预览 object URL。
  private releaseAllPreviewUrls(): void {
    for (const preview of this.privatePreviewUrls.values()) {
      void preview.then(value => URL.revokeObjectURL(value)).catch(() => undefined);
    }
    this.privatePreviewUrls.clear();
  }

  // 将后端返回的相对资源路径补成当前 API baseUrl 下的完整 URL。
  private normalizeResourceUrl(url: string): string {
    if (!url || /^https?:\/\//i.test(url) || url.startsWith('data:') || url.startsWith('blob:')) {
      return url;
    }
    return `${this.api.baseUrl}${url.startsWith('/') ? url : '/' + url}`;
  }

  // 构造携带 JWT 的请求头，用于访问数据集和私有预览文件。
  private authHeaders(): Headers {
    const headers = new Headers();
    if (this.api.token) {
      headers.set('Authorization', `Bearer ${this.api.token}`);
    }
    return headers;
  }

  // 统一读取 JSON 响应，并把后端错误文本转换成前端可展示的异常。
  private async readJson<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}
