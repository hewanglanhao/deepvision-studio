import { Injectable } from '@angular/core';
import {
  ForwardRecordDetail,
  ForwardRecordSummary,
  SaveForwardRecordRequest
} from '@shared/forward/forward-record.models';
import { ApiClientService } from '@core/api/api-client.service';

@Injectable({ providedIn: 'root' })
export class ForwardRecordService {
  /** 注入统一 API 客户端，用于读取、保存和删除当前用户的 A 模式实验快照。 */
  constructor(private api: ApiClientService) {}

  /** 查询当前用户的记录列表或接口数据集合。 */
  list(): Promise<ForwardRecordSummary[]> {
    return this.api.request<ForwardRecordSummary[]>('/api/a/forward-records');
  }

  /** 创建并持久化一条 A 模式历史记录。 */
  create(payload: SaveForwardRecordRequest): Promise<ForwardRecordDetail> {
    return this.api.request<ForwardRecordDetail>('/api/a/forward-records', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  /** 读取当前用户指定的 A 模式历史记录详情。 */
  detail(id: number): Promise<ForwardRecordDetail> {
    return this.api.request<ForwardRecordDetail>(`/api/a/forward-records/${id}`);
  }

  /** 删除当前用户指定的 A 模式历史记录。 */
  delete(id: number): Promise<void> {
    return this.api.request<void>(`/api/a/forward-records/${id}`, { method: 'DELETE' });
  }

  /** 将后端返回的相对图片路径拼成完整 URL，用于显示历史记录输入缩略图。 */
  imageUrl(path: string | null): string {
    return path ? `${this.api.baseUrl}${path}` : '';
  }
}
