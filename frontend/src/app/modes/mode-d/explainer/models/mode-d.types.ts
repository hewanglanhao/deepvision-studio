export interface ModeDExample {
  id: string;
  title: string;
  subtitle: string;
  text: string;
  focus: string;
  candidateTokens: string[];
}

export interface ModeDTokenScore {
  tokenId: number;
  token: string;
  probability: number;
  rank: number;
}

export interface ModeDAttentionSummary {
  sourceToken: string;
  targetToken: string;
  weight: number;
  narrative: string;
}

export interface ModeDReportSection {
  title: string;
  body: string;
}

export interface ModeDPhaseZeroProbeResult {
  tokenizerReady: boolean;
  sessionReady: boolean;
  logitsReady: boolean;
  attentionKeys: string[];
  tokenIds: number[];
  tokenTexts: string[];
}

export interface ModeDInferenceResult {
  tokenIds: number[];
  tokenTexts: string[];
  topK: ModeDTokenScore[];
  attentionByKey: Record<string, number[][]>;
  attentionKeys: string[];
}

export interface ModeDVectorBar {
  label: string;
  value: number;
}

export interface ModeDQkvTeachingData {
  queryToken: string;
  keyToken: string;
  valueToken: string;
  queryIndex: number;
  keyIndex: number;
  attentionWeight: number;
  queryVector: ModeDVectorBar[];
  keyVector: ModeDVectorBar[];
  valueVector: ModeDVectorBar[];
  summary: string;
}

