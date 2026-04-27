import { Injectable } from '@angular/core';
import { TrainingDatasetDetail, TrainingDatasetOption } from '../sim-models';
import { ApiClientService } from './api-client.service';

export interface DatasetImportResponse {
  datasetId: string;
  detail: TrainingDatasetDetail;
}

@Injectable({ providedIn: 'root' })
export class TrainingDatasetApiService {
  private readonly basePath = '/api/training/datasets';

  constructor(private api: ApiClientService) {}

  async listBuiltinDatasets(signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/builtin`, { signal });
    return this.readJson<TrainingDatasetOption[]>(response);
  }

  async getDatasetDetail(datasetId: string, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    const response = await fetch(`${this.api.baseUrl}${this.basePath}/${encodeURIComponent(datasetId)}`, { signal });
    return this.readJson<TrainingDatasetDetail>(response);
  }

  async importDataset(files: File[], signal?: AbortSignal): Promise<DatasetImportResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }

    const response = await fetch(`${this.api.baseUrl}${this.basePath}/imports`, {
      method: 'POST',
      body: form,
      signal
    });
    return this.readJson<DatasetImportResponse>(response);
  }

  private async readJson<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json() as Promise<T>;
  }
}
