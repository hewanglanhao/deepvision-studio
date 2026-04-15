import { Injectable } from '@angular/core';
import { Connection, ForwardPassResult, ForwardTensor, NetworkLayer } from '../sim-models';

interface ForwardRequestPayload {
  layers: NetworkLayer[];
  connections: Connection[];
  inputTensor: ForwardTensor;
}

@Injectable({ providedIn: 'root' })
export class ForwardBackendService {
  private readonly endpoint = 'http://127.0.0.1:5000/api/forward';

  async executeForward(payload: ForwardRequestPayload, signal?: AbortSignal): Promise<ForwardPassResult> {
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    return response.json() as Promise<ForwardPassResult>;
  }
}
