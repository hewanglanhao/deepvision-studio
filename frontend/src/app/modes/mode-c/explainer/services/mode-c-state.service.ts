import { Injectable, computed, signal } from '@angular/core';
import { ModeCAssetsService } from './mode-c-assets.service';
import {
  ModeCFocusArea,
  ModeCLayerActivationSummary,
  ModeCLayerDetailRuntime,
  ModeCNetworkLayer,
  ModeCLayerPreview,
  ModeCSamplePrediction
} from '../models/mode-c.types';
import { ModeCInferenceService } from './mode-c-inference.service';

@Injectable({ providedIn: 'root' })
export class ModeCStateService {
  readonly sampleOptions;
  readonly overviewStages;
  readonly detailTopics;
  readonly milestones;
  readonly articleSections;

  readonly networkLayers;
  readonly networkLayersLoading;
  readonly networkLayersError;
  readonly inferenceLoading;
  readonly inferenceError;
  readonly activeFocus;
  readonly currentSampleId;
  readonly selectedTopicId;
  readonly selectedLayerId;
  readonly selectedChannelIndex;

  readonly currentSample;
  readonly selectedTopic;
  readonly selectedLayer;
  readonly currentSamplePrediction;
  readonly layerSummaries;
  readonly layerPreviews;
  readonly layerDetails;
  readonly selectedLayerSummary;
  readonly selectedLayerPreview;
  readonly selectedLayerDetail;
  readonly previousLayer;
  readonly previousLayerSummary;

  constructor(
    private readonly assets: ModeCAssetsService,
    private readonly inference: ModeCInferenceService
  ) {
    this.sampleOptions = this.assets.sampleOptions;
    this.overviewStages = this.assets.overviewStages;
    this.detailTopics = this.assets.detailTopics;
    this.milestones = this.assets.milestones;
    this.articleSections = this.assets.articleSections;
    this.networkLayers = signal<ModeCNetworkLayer[]>(this.assets.fallbackNetworkLayers);
    this.networkLayersLoading = signal(false);
    this.networkLayersError = signal('');
    this.inferenceLoading = signal(false);
    this.inferenceError = signal('');
    this.activeFocus = signal<ModeCFocusArea>('overview');
    this.currentSampleId = signal(this.sampleOptions[0]?.id ?? '');
    this.selectedTopicId = signal(this.detailTopics[0]?.id ?? '');
    this.selectedLayerId = signal(this.networkLayers()[0]?.id ?? '');
    this.selectedChannelIndex = signal(0);
    this.layerSummaries = signal<Record<string, ModeCLayerActivationSummary>>({});
    this.layerPreviews = signal<Record<string, ModeCLayerPreview>>({});
    this.layerDetails = signal<Record<string, ModeCLayerDetailRuntime>>({});
    this.currentSample = computed(() =>
      this.sampleOptions.find(sample => sample.id === this.currentSampleId()) ?? null
    );
    this.selectedTopic = computed(() =>
      this.detailTopics.find(topic => topic.id === this.selectedTopicId()) ?? null
    );
    this.selectedLayer = computed(() =>
      this.networkLayers().find(layer => layer.id === this.selectedLayerId()) ?? null
    );
    this.currentSamplePrediction = signal<ModeCSamplePrediction | null>(null);
    this.selectedLayerSummary = computed(() => {
      const layer = this.selectedLayer();
      if (!layer) return null;
      return this.layerSummaries()[layer.id] ?? null;
    });
    this.selectedLayerPreview = computed(() => {
      const layer = this.selectedLayer();
      if (!layer) return null;
      return this.layerPreviews()[layer.id] ?? null;
    });
    this.selectedLayerDetail = computed(() => {
      const layer = this.selectedLayer();
      if (!layer) return null;
      return this.layerDetails()[layer.id] ?? null;
    });
    this.previousLayer = computed(() => {
      const currentId = this.selectedLayerId();
      const layers = this.networkLayers();
      const index = layers.findIndex(layer => layer.id === currentId);
      if (index <= 0) return null;
      return layers[index - 1] ?? null;
    });
    this.previousLayerSummary = computed(() => {
      const previous = this.previousLayer();
      if (!previous) return null;
      return this.layerSummaries()[previous.id] ?? null;
    });
  }

  async initializeNetworkLayers(): Promise<void> {
    if (this.networkLayersLoading()) return;

    this.networkLayersLoading.set(true);
    this.networkLayersError.set('');

    try {
      const layers = await this.assets.loadNetworkLayers();
      this.networkLayers.set(layers);
      if (!layers.some(layer => layer.id === this.selectedLayerId())) {
        this.selectedLayerId.set(layers[0]?.id ?? '');
      }
      await this.refreshInference();
    } catch (error) {
      this.networkLayersError.set(error instanceof Error ? error.message : 'Failed to load network layers.');
    } finally {
      this.networkLayersLoading.set(false);
    }
  }

  setCurrentSample(sampleId: string): void {
    if (!this.sampleOptions.some(sample => sample.id === sampleId)) return;
    this.currentSampleId.set(sampleId);
    void this.refreshInference();
  }

  setActiveFocus(focus: ModeCFocusArea): void {
    this.activeFocus.set(focus);
  }

  setSelectedTopic(topicId: string): void {
    if (!this.detailTopics.some(topic => topic.id === topicId)) return;
    this.selectedTopicId.set(topicId);
    this.activeFocus.set('detail');
  }

  setSelectedLayer(layerId: string): void {
    const layer = this.networkLayers().find(item => item.id === layerId);
    if (!layer) return;

    this.selectedLayerId.set(layerId);
    this.selectedChannelIndex.set(0);
    this.activeFocus.set('overview');

    if (layer.type === 'conv') {
      this.selectedTopicId.set('conv-panel');
    } else if (layer.type === 'output') {
      this.selectedTopicId.set('softmax-panel');
    } else {
      this.selectedTopicId.set('overview-graph');
    }
  }

  setSelectedChannel(index: number): void {
    if (index < 0) return;
    this.selectedChannelIndex.set(index);
    this.activeFocus.set('overview');
  }

  private async refreshInference(): Promise<void> {
    const sample = this.currentSample();
    if (!sample || this.inferenceLoading()) return;

    this.inferenceLoading.set(true);
    this.inferenceError.set('');

    try {
      const result = await this.inference.runSample(sample.assetPath);
      this.currentSamplePrediction.set(result.prediction);
      this.layerSummaries.set(
        result.summaries.reduce<Record<string, ModeCLayerActivationSummary>>((acc, summary) => {
          acc[summary.layerId] = summary;
          return acc;
        }, {})
      );
      this.layerPreviews.set(
        result.previews.reduce<Record<string, ModeCLayerPreview>>((acc, preview) => {
          acc[preview.layerId] = preview;
          return acc;
        }, {})
      );
      this.layerDetails.set(
        result.details.reduce<Record<string, ModeCLayerDetailRuntime>>((acc, detail) => {
          acc[detail.layerId] = detail;
          return acc;
        }, {})
      );
      const currentLayerId = this.selectedLayerId();
      const currentDetail = result.details.find(detail => detail.layerId === currentLayerId);
      const availableChannels = currentDetail?.channelPreviews ?? [];
      if (!availableChannels.length) {
        this.selectedChannelIndex.set(0);
      } else if (!availableChannels.some(channel => channel.index === this.selectedChannelIndex())) {
        this.selectedChannelIndex.set(availableChannels[0]?.index ?? 0);
      }
    } catch (error) {
      this.currentSamplePrediction.set(null);
      this.layerSummaries.set({});
      this.layerPreviews.set({});
      this.layerDetails.set({});
      this.inferenceError.set(error instanceof Error ? error.message : 'Failed to run Mode C inference.');
    } finally {
      this.inferenceLoading.set(false);
    }
  }
}
