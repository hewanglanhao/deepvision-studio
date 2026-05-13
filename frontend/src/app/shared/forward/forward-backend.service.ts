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
  constructor(private api: ApiClientService) {}

  async executeForward(payload: ForwardRequestPayload): Promise<ForwardPassResult> {
    return this.api.request<ForwardPassResult>('/api/forward', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}
