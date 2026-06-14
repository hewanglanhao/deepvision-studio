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

  async listDatasets(source?: string, signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    const query = source ? `?source=${encodeURIComponent(source)}` : '';
    const response = await fetch(`${this.api.baseUrl}${this.basePath}${query}`, {
      headers: this.authHeaders(),
      signal
    });
    return this.readJson<TrainingDatasetOption[]>(response);
  }

  async listBuiltinDatasets(signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    return this.listDatasets('builtin', signal);
  }

  async getDatasetDetail(datasetId: string, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, {
      headers: this.authHeaders(),
      signal
    });
    return this.normalizeDatasetDetail(await this.readJson<TrainingDatasetDetail>(response), signal);
  }

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

  ngOnDestroy(): void {
    this.releaseAllPreviewUrls();
  }

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

  private isPrivateDatasetFileUrl(url: string): boolean {
    try {
      const path = new URL(url, window.location.origin).pathname;
      return path.startsWith(`${this.basePath}/`) && path.includes('/files/');
    } catch {
      return url.includes('/api/training/datasets/') && url.includes('/files/');
    }
  }

  private ensurePreviewCacheOwner(): void {
    const token = this.api.token;
    if (token === this.previewToken) return;
    this.releaseAllPreviewUrls();
    this.previewToken = token;
  }

  private releaseDatasetPreviewUrls(datasetId: string): void {
    const encodedId = encodeURIComponent(datasetId);
    for (const [url, preview] of this.privatePreviewUrls) {
      if (!url.includes(`/datasets/${encodedId}/`)) continue;
      void preview.then(value => URL.revokeObjectURL(value)).catch(() => undefined);
      this.privatePreviewUrls.delete(url);
    }
  }

  private releaseAllPreviewUrls(): void {
    for (const preview of this.privatePreviewUrls.values()) {
      void preview.then(value => URL.revokeObjectURL(value)).catch(() => undefined);
    }
    this.privatePreviewUrls.clear();
  }

  private normalizeResourceUrl(url: string): string {
    if (!url || /^https?:\/\//i.test(url) || url.startsWith('data:') || url.startsWith('blob:')) {
      return url;
    }
    return `${this.api.baseUrl}${url.startsWith('/') ? url : '/' + url}`;
  }

  private authHeaders(): Headers {
    const headers = new Headers();
    if (this.api.token) {
      headers.set('Authorization', `Bearer ${this.api.token}`);
    }
    return headers;
  }

  private async readJson<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}
