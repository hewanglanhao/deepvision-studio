import { Injectable } from '@angular/core';
import { ApiClientService } from '@core/api/api-client.service';
import { Connection, ForwardPassResult, ForwardTensor, NetworkLayer } from '@shared/simulation/sim-models';

interface ForwardRequestPayload {
  layers: NetworkLayer[];
  connections: Connection[];
  inputTensor: ForwardTensor;
}

@Injectable({ providedIn: 'root' })
export class ForwardBackendService {
  /** 注入统一 API 客户端，A 模式的前向传播请求都通过它发送到 Spring 代理。 */
  constructor(private api: ApiClientService) {}

  /** 向 Spring forward 代理提交网络和输入张量，获取 Python 计算结果。 */
  async executeForward(payload: ForwardRequestPayload): Promise<ForwardPassResult> {
    return this.api.request<ForwardPassResult>('/api/forward', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}
