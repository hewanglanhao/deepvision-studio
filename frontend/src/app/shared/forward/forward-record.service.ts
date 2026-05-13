import { Injectable } from '@angular/core';
import {
  ForwardRecordDetail,
  ForwardRecordSummary,
  SaveForwardRecordRequest
} from '@shared/forward/forward-record.models';
import { ApiClientService } from '@core/api/api-client.service';

@Injectable({ providedIn: 'root' })
export class ForwardRecordService {
  constructor(private api: ApiClientService) {}

  list(): Promise<ForwardRecordSummary[]> {
    return this.api.request<ForwardRecordSummary[]>('/api/a/forward-records');
  }

  create(payload: SaveForwardRecordRequest): Promise<ForwardRecordDetail> {
    return this.api.request<ForwardRecordDetail>('/api/a/forward-records', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  detail(id: number): Promise<ForwardRecordDetail> {
    return this.api.request<ForwardRecordDetail>(`/api/a/forward-records/${id}`);
  }

  delete(id: number): Promise<void> {
    return this.api.request<void>(`/api/a/forward-records/${id}`, { method: 'DELETE' });
  }

  imageUrl(path: string | null): string {
    return path ? `${this.api.baseUrl}${path}` : '';
  }
}

