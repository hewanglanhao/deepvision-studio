import { Injectable } from '@angular/core';
import { TrainingDatasetDetail, TrainingDatasetOption } from '@shared/simulation/sim-models';
import { ApiClientService } from '@core/api/api-client.service';

export interface DatasetImportResponse {
  datasetId: string;
  detail: TrainingDatasetDetail;
}

@Injectable({ providedIn: 'root' })
export class TrainingDatasetApiService {
  private readonly basePath = '/api/training/datasets';

  constructor(private api: ApiClientService) {}

  async listDatasets(source?: string, signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    const query = source ? `?source=${encodeURIComponent(source)}` : '';
    const response = await fetch(`${this.api.baseUrl}${this.basePath}${query}`, { signal });
    return this.readJson<TrainingDatasetOption[]>(response);
  }

  async listBuiltinDatasets(signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    return this.listDatasets('builtin', signal);
  }

  async getDatasetDetail(datasetId: string, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, { signal });
    return this.normalizeDatasetDetail(await this.readJson<TrainingDatasetDetail>(response));
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
      body: form,
      signal
    });
    const result = await this.readJson<DatasetImportResponse>(response);
    return { ...result, detail: this.normalizeDatasetDetail(result.detail) };
  }

  async deleteDataset(datasetId: string, signal?: AbortSignal): Promise<void> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, {
      method: 'DELETE',
      signal
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
  }

  private normalizeDatasetDetail(detail: TrainingDatasetDetail): TrainingDatasetDetail {
    return {
      ...detail,
      imagePreview: (detail.imagePreview ?? []).map(item => ({
        ...item,
        url: this.normalizeResourceUrl(item.url)
      }))
    };
  }

  private normalizeResourceUrl(url: string): string {
    if (!url || /^https?:\/\//i.test(url) || url.startsWith('data:') || url.startsWith('blob:')) {
      return url;
    }
    return `${this.api.baseUrl}${url.startsWith('/') ? url : '/' + url}`;
  }

  private async readJson<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}
