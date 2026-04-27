import { Injectable } from '@angular/core';
import { TrainingDatasetDetail, TrainingDatasetOption } from '../sim-models';

export interface DatasetImportResponse {
  datasetId: string;
  detail: TrainingDatasetDetail;
}

@Injectable({ providedIn: 'root' })
export class TrainingDatasetApiService {
  private readonly baseUrl = 'http://127.0.0.1:5000/api/training/datasets';

  async listBuiltinDatasets(signal?: AbortSignal): Promise<TrainingDatasetOption[]> {
    const response = await fetch(`${this.baseUrl}/builtin`, { signal });
    return this.readJson<TrainingDatasetOption[]>(response);
  }

  async getDatasetDetail(datasetId: string, signal?: AbortSignal): Promise<TrainingDatasetDetail> {
    const response = await fetch(`${this.baseUrl}/${encodeURIComponent(datasetId)}`, { signal });
    return this.readJson<TrainingDatasetDetail>(response);
  }

  async importDataset(files: File[], signal?: AbortSignal): Promise<DatasetImportResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }

    const response = await fetch(`${this.baseUrl}/imports`, {
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
