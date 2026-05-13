import { Connection, ForwardPassResult, NetworkLayer } from '@shared/simulation/sim-models';

export interface ForwardRecordSnapshot {
  selectedTemplateId: string;
  selectedDataset: string;
  selectedSampleId: number;
  selectedLayerId: number;
  uploadComputeProfile: 'fast' | 'balanced' | 'quality' | 'original';
  uploadedImageUrl: string;
  layers: NetworkLayer[];
  connections: Connection[];
  forwardResult: ForwardPassResult | null;
}

export interface ForwardRecordSummary {
  id: number;
  name: string;
  templateId: string;
  datasetName: string;
  layerCount: number;
  parameterCount: number;
  imagePath: string | null;
  createdAt: string;
}

export interface ForwardRecordDetail extends ForwardRecordSummary {
  snapshot: ForwardRecordSnapshot;
}

export interface SaveForwardRecordRequest {
  name: string;
  templateId: string;
  datasetName: string;
  layerCount: number;
  parameterCount: number;
  previewImageDataUrl: string | null;
  snapshot: ForwardRecordSnapshot;
}

