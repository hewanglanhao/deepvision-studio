import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LayerType, NetworkLayer, TensorShape } from '@shared/simulation/sim-models';
import { buildNetwork3dLayerViews } from './network-3d-layout';
import { NETWORK_3D_SESSION_KEY, Network3dLayerSnapshot, Network3dLayerView, Network3dPayload } from './network-3d.models';

type LayerFace = { center: THREE.Vector3; width: number; height: number; layerId?: number; type?: LayerType };
type FlowPath = { curve: THREE.QuadraticBezierCurve3; particle: THREE.Mesh; offset: number; speed: number };

const MODEL_OFFSET_X = -1.05;
const MODEL_OFFSET_Y = 8.4;
const CAMERA_TARGET_Y = 8.65;

@Component({
  selector: 'app-network-3d-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="viewer-shell">
      <header class="viewer-topbar">
        <div class="title-block">
          <div class="viewer-title">{{ payload?.title || 'Network 3D Viewer' }}</div>
          <div class="viewer-subtitle">{{ subtitle }}</div>
        </div>
        <div class="viewer-actions">
          <div class="mode-switch" role="group" aria-label="画面操作模式">
            <button type="button" [class.active]="interactionMode === 'rotate'" (click)="setInteractionMode('rotate')" [disabled]="!payload">旋转</button>
            <button type="button" [class.active]="interactionMode === 'pan'" (click)="setInteractionMode('pan')" [disabled]="!payload">平移</button>
          </div>
          <button type="button" (click)="toggleFlow()" [disabled]="!payload">{{ flowPlaying ? '暂停传播' : '播放传播' }}</button>
          <button type="button" (click)="resetCamera()" [disabled]="!payload">重置视角</button>
          <button type="button" (click)="closeWindow()">关闭</button>
        </div>
      </header>

      @if (payload) {
        <main class="viewer-main">
          <section class="stage-wrap">
            <div #stage class="stage"></div>
            <div class="stage-hint">当前：{{ interactionMode === 'rotate' ? '拖拽旋转' : '拖拽平移画布' }} · 单击查看详情 · 双击聚焦 · 滚轮缩放</div>
          </section>

          <aside class="layer-panel">
            <section class="panel-section">
              <div class="panel-title">传播路径</div>
              @if (payload.shapePath?.length) {
                <div class="shape-path">
                  @for (step of payload.shapePath; track $index) {
                    <span>{{ step }}</span>
                  }
                </div>
              } @else {
                <div class="muted">暂无形状路径，显示结构推断结果。</div>
              }
            </section>

            <section class="panel-section">
              <div class="panel-title">网络层</div>
              <div class="layer-list">
                @for (item of layerViews; track item.layer.id) {
                  <button
                    type="button"
                    class="layer-row"
                    [class.active]="item.layer.id === selectedLayerId"
                    [class.hovered]="item.layer.id === hoverLayerId"
                    (click)="focusLayer(item.layer.id)"
                  >
                    <span class="layer-dot" [style.background]="item.color"></span>
                    <span class="layer-main">
                      <strong>{{ item.layer.name }}</strong>
                      <small>{{ typeLabel(item.layer.type) }} · {{ item.shapeLabel }}</small>
                    </span>
                  </button>
                }
              </div>
            </section>

            @if (selectedView; as view) {
              <section class="panel-section detail-section">
                <div class="panel-title">层详情</div>
                <div class="detail-head">
                  <span class="type-pill" [style.borderColor]="view.color">{{ typeLabel(view.layer.type) }}</span>
                  <strong>{{ view.layer.name }}</strong>
                </div>
                @if (selectedSnapshot; as snap) {
                  <div class="shape-compare">
                    <span>{{ snap.inputShapeLabel }}</span>
                    <span>→</span>
                    <span>{{ snap.outputShapeLabel }}</span>
                  </div>
                  <p class="detail-note">{{ snap.transitionNote }}</p>
                  @if (snap.previewImageUrl) {
                    <img class="layer-preview" [src]="snap.previewImageUrl" [alt]="view.layer.name + ' output'" />
                  }
                  @if (snap.channelPreviews.length) {
                    <div class="channel-grid">
                      @for (channel of snap.channelPreviews; track channel.channel) {
                        <div class="channel-tile">
                          <img [src]="channel.imageUrl" [alt]="'channel ' + channel.channel" />
                          <span>ch {{ channel.channel }}</span>
                        </div>
                      }
                    </div>
                  }
                  <div class="stats-grid">
                    <div><span>min</span><strong>{{ snap.stats.min | number:'1.3-3' }}</strong></div>
                    <div><span>max</span><strong>{{ snap.stats.max | number:'1.3-3' }}</strong></div>
                    <div><span>mean</span><strong>{{ snap.stats.mean | number:'1.3-3' }}</strong></div>
                    <div><span>non-zero</span><strong>{{ snap.stats.nonZeroRatio | percent:'1.0-0' }}</strong></div>
                  </div>
                  @if (snap.topK.length) {
                    <div class="topk-list">
                      @for (item of snap.topK; track item.index) {
                        <div class="topk-row">
                          <span>{{ item.label ?? ('#' + item.index) }}</span>
                          <div><i [style.width.%]="topKWidth(item.value, snap.topK)"></i></div>
                          <strong>{{ item.value | number:'1.3-3' }}</strong>
                        </div>
                      }
                    </div>
                  }
                  @if (snap.paramsSummary.length) {
                    <div class="param-list">
                      @for (param of snap.paramsSummary; track $index) {
                        <span>{{ param }}</span>
                      }
                    </div>
                  }
                } @else {
                  <div class="shape-compare"><span>{{ view.shapeLabel }}</span></div>
                  <p class="detail-note">当前快照没有真实前向结果，3D 场景使用结构与 shape 推断展示。</p>
                }
              </section>
            }
          </aside>
        </main>
      } @else {
        <div class="empty-state">
          <div class="empty-title">暂无可展示的网络快照</div>
          <div class="empty-copy">请从 A 模式的网络结构区域点击“3D化显示”。</div>
        </div>
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      min-height: 100vh;
      background: #101620;
      color: #e8eef7;
      font-family: 'Manrope', 'Segoe UI', 'Noto Sans SC', sans-serif;
    }

    .viewer-shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .viewer-topbar {
      min-height: 60px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 16px;
      border-bottom: 1px solid rgba(166, 184, 208, .22);
      background: rgba(16, 22, 32, .96);
    }

    .title-block {
      min-width: 0;
    }

    .viewer-title {
      font-size: 16px;
      font-weight: 800;
      color: #f8fafc;
    }

    .viewer-subtitle {
      margin-top: 2px;
      font-size: 12px;
      color: #9aa9bd;
    }

    .viewer-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .mode-switch {
      display: flex;
      gap: 0;
      border: 1px solid rgba(166, 184, 208, .36);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(34, 45, 61, .72);
    }

    button {
      border: 1px solid rgba(166, 184, 208, .36);
      border-radius: 8px;
      padding: 7px 11px;
      background: rgba(34, 45, 61, .88);
      color: #e8eef7;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }

    .mode-switch button {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding-inline: 10px;
    }

    .mode-switch button.active {
      background: rgba(85, 194, 215, .24);
      color: #d8f3f8;
    }

    button:hover:not(:disabled) {
      border-color: #55c2d7;
      background: rgba(35, 128, 150, .22);
    }

    button:disabled {
      opacity: .45;
      cursor: not-allowed;
    }

    .viewer-main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
    }

    .stage-wrap {
      min-width: 0;
      min-height: 0;
      position: relative;
    }

    .stage {
      position: absolute;
      inset: 0;
      overflow: hidden;
    }

    .stage-hint {
      position: absolute;
      left: 14px;
      bottom: 14px;
      padding: 7px 10px;
      border: 1px solid rgba(166, 184, 208, .2);
      border-radius: 8px;
      background: rgba(16, 22, 32, .74);
      color: #b8c6d8;
      font-size: 12px;
      pointer-events: none;
    }

    .layer-panel {
      min-height: 0;
      border-left: 1px solid rgba(166, 184, 208, .2);
      background: rgba(16, 22, 32, .82);
      padding: 12px;
      overflow-y: auto;
    }

    .panel-section {
      padding-bottom: 12px;
      margin-bottom: 12px;
      border-bottom: 1px solid rgba(166, 184, 208, .14);
    }

    .panel-title {
      color: #9aa9bd;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .muted {
      color: #9aa9bd;
      font-size: 12px;
    }

    .shape-path {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
    }

    .shape-path span {
      max-width: 100%;
      border: 1px solid rgba(129, 212, 250, .28);
      border-radius: 7px;
      padding: 3px 6px;
      color: #c8edf7;
      background: rgba(35, 128, 150, .14);
      font: 11px Consolas, monospace;
    }

    .layer-list {
      display: grid;
      gap: 7px;
    }

    .layer-row {
      display: grid;
      grid-template-columns: 10px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      width: 100%;
      text-align: left;
      background: rgba(34, 45, 61, .62);
    }

    .layer-row.active {
      border-color: #55c2d7;
      background: rgba(35, 128, 150, .22);
    }

    .layer-row.hovered:not(.active) {
      border-color: rgba(245, 158, 11, .65);
    }

    .layer-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    .layer-main {
      min-width: 0;
      display: grid;
      gap: 2px;
    }

    .layer-main strong {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      color: #f8fafc;
    }

    .layer-main small {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #9aa9bd;
      font-size: 11px;
    }

    .detail-section {
      border-bottom: 0;
    }

    .detail-head {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }

    .detail-head strong {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .type-pill {
      border: 1px solid;
      border-radius: 999px;
      padding: 2px 7px;
      color: #f8fafc;
      font-size: 11px;
      background: rgba(255, 255, 255, .06);
    }

    .shape-compare {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      color: #c8edf7;
      font: 12px Consolas, monospace;
    }

    .detail-note {
      margin: 8px 0;
      color: #b8c6d8;
      font-size: 12px;
      line-height: 1.5;
    }

    .layer-preview {
      width: 100%;
      max-height: 170px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid rgba(166, 184, 208, .2);
      background: rgba(255, 255, 255, .04);
    }

    .channel-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 7px;
      margin-top: 8px;
    }

    .channel-tile {
      min-width: 0;
      display: grid;
      gap: 3px;
      color: #9aa9bd;
      font-size: 10px;
      text-align: center;
    }

    .channel-tile img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid rgba(166, 184, 208, .18);
      background: rgba(255, 255, 255, .04);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
      margin-top: 10px;
    }

    .stats-grid div {
      min-width: 0;
      border: 1px solid rgba(166, 184, 208, .16);
      border-radius: 8px;
      padding: 6px;
      background: rgba(255, 255, 255, .04);
    }

    .stats-grid span {
      display: block;
      color: #9aa9bd;
      font-size: 10px;
    }

    .stats-grid strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      color: #f8fafc;
      font-size: 12px;
      margin-top: 2px;
    }

    .topk-list {
      display: grid;
      gap: 5px;
      margin-top: 10px;
    }

    .topk-row {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr) 52px;
      gap: 6px;
      align-items: center;
      color: #b8c6d8;
      font-size: 11px;
    }

    .topk-row > span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .topk-row div {
      height: 6px;
      border-radius: 999px;
      background: rgba(166, 184, 208, .18);
      overflow: hidden;
    }

    .topk-row i {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: #55c2d7;
    }

    .topk-row strong {
      text-align: right;
      color: #f8fafc;
      font-size: 11px;
    }

    .param-list {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 10px;
    }

    .param-list span {
      max-width: 100%;
      border: 1px solid rgba(166, 184, 208, .18);
      border-radius: 7px;
      padding: 3px 6px;
      color: #b8c6d8;
      background: rgba(255, 255, 255, .04);
      font: 11px Consolas, monospace;
    }

    .empty-state {
      display: grid;
      place-content: center;
      gap: 8px;
      text-align: center;
      padding: 30px;
    }

    .empty-title {
      font-size: 18px;
      font-weight: 800;
      color: #f8fafc;
    }

    .empty-copy {
      color: #9aa9bd;
      font-size: 13px;
    }

    @media (max-width: 980px) {
      .viewer-main {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(430px, 1fr) auto;
      }

      .stage-wrap {
        min-height: 430px;
      }

      .layer-panel {
        max-height: 44vh;
        border-left: 0;
        border-top: 1px solid rgba(166, 184, 208, .2);
      }
    }
  `]
})
export class Network3dViewerComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('stage') private stageRef?: ElementRef<HTMLDivElement>;

  payload: Network3dPayload | null = null;
  layerViews: Network3dLayerView[] = [];
  selectedLayerId = -1;
  hoverLayerId = -1;
  flowPlaying = true;
  interactionMode: 'rotate' | 'pan' = 'rotate';

  private scene?: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private renderer?: THREE.WebGLRenderer;
  private controls?: OrbitControls;
  private networkGroup?: THREE.Group;
  private animationId = 0;
  private readonly layerObjects = new Map<number, THREE.Object3D>();
  private readonly layerMaterials = new Map<number, THREE.MeshStandardMaterial[]>();
  private readonly flowPaths: FlowPath[] = [];
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly textureLoader = new THREE.TextureLoader();
  private readonly resizeObserver = new ResizeObserver(() => this.resizeRenderer());
  private pointerDown: { x: number; y: number } | null = null;

  /** 生成 3D 查看器副标题，概括快照来源、数据集、层数和参数量。 */
  get subtitle(): string {
    if (!this.payload) return '等待 A/B/C/D 模式传入网络层数据';
    const meta = [
      this.payload.sourceMode,
      this.payload.datasetName,
      this.payload.inputLabel,
      `${this.layerViews.length} 层`,
      this.payload.parameterCount ? `${this.payload.parameterCount.toLocaleString()} 参数` : ''
    ].filter(Boolean);
    return `${meta.join(' · ')} · ${new Date(this.payload.createdAt).toLocaleString()}`;
  }

  /** 返回当前选中的 3D 层节点，右侧面板据此显示该层 shape 和类型。 */
  get selectedView(): Network3dLayerView | undefined {
    return this.layerViews.find(item => item.layer.id === this.selectedLayerId) ?? this.layerViews[0];
  }

  /** 返回当前层的 forward 快照，包括特征图预览、统计值、通道图和 Top-K。 */
  get selectedSnapshot(): Network3dLayerSnapshot | undefined {
    const id = this.selectedView?.layer.id;
    return id === undefined ? undefined : this.payload?.layerSnapshots?.[id];
  }

  /** 初始化页面状态、订阅数据源并触发首次数据加载。 */
  ngOnInit(): void {
    this.payload = this.readPayload();
    if (!this.payload) return;
    this.selectedLayerId = this.payload.selectedLayerId;
    this.layerViews = buildNetwork3dLayerViews(
      this.payload.layers,
      this.payload.layerShapes,
      this.payload.shapeHints
    );
    if (!this.layerViews.some(view => view.layer.id === this.selectedLayerId)) {
      this.selectedLayerId = this.layerViews[0]?.layer.id ?? -1;
    }
  }

  /** 在视图元素创建后初始化依赖 DOM 的渲染流程。 */
  ngAfterViewInit(): void {
    if (!this.payload || !this.stageRef) return;
    this.initScene(this.stageRef.nativeElement);
  }

  /** 释放组件订阅、定时器和渲染资源，避免页面离开后继续占用内存。 */
  ngOnDestroy(): void {
    if (this.animationId) cancelAnimationFrame(this.animationId);
    this.resizeObserver.disconnect();
    this.controls?.dispose();
    this.renderer?.domElement.removeEventListener('pointerdown', this.onCanvasPointerDown);
    this.renderer?.domElement.removeEventListener('click', this.onCanvasClick);
    this.renderer?.domElement.removeEventListener('dblclick', this.onCanvasDoubleClick);
    this.renderer?.domElement.removeEventListener('pointermove', this.onCanvasPointerMove);
    this.renderer?.dispose();
    this.disposeObject(this.networkGroup);
  }

  /** 重置相机到能完整观察网络层序列的位置。 */
  resetCamera(): void {
    if (!this.camera || !this.controls) return;
    const length = Math.max(1, this.layerViews.length);
    this.camera.position.set(MODEL_OFFSET_X + 5.8, CAMERA_TARGET_Y + 3.3, Math.max(11.2, length * 1.2));
    this.controls.target.set(MODEL_OFFSET_X, CAMERA_TARGET_Y, 0);
    this.controls.update();
  }

  /** 关闭 3D 查看窗口，回到产生快照的 A 模式页面。 */
  closeWindow(): void {
    window.close();
  }

  /** 开关前向传播粒子动画，用动态粒子表示张量从输入层流向输出层。 */
  toggleFlow(): void {
    this.flowPlaying = !this.flowPlaying;
  }

  /** 切换鼠标交互模式：旋转观察整体网络，或平移查看局部层结构。 */
  setInteractionMode(mode: 'rotate' | 'pan'): void {
    this.interactionMode = mode;
    this.applyInteractionMode();
  }

  /** 将相机聚焦到某一层，便于近距离查看特征图堆叠或神经元网格。 */
  focusLayer(layerId: number): void {
    this.selectedLayerId = layerId;
    this.updateSelectionMaterials();
    const object = this.layerObjects.get(layerId);
    if (!object || !this.camera || !this.controls) return;
    const position = new THREE.Vector3();
    object.getWorldPosition(position);
    this.controls.target.copy(position);
    this.camera.position.set(position.x + 4.0, position.y + 3.0, position.z + 5.0);
    this.controls.update();
  }

  /** 选中一层并更新材质高亮，右侧详情会同步显示该层 forward 信息。 */
  selectLayer(layerId: number): void {
    this.selectedLayerId = layerId;
    this.updateSelectionMaterials();
  }

  /** 把层类型转成可读标签，帮助区分 Conv2D、Pooling、Flatten、Dense 等角色。 */
  typeLabel(type: string): string {
    const labels: Record<string, string> = {
      input: 'Input',
      conv2d: 'Conv2D',
      pool2d: 'Pool2D',
      flatten: 'Flatten',
      dense: 'Dense',
      activation: 'Activation',
      dropout: 'Dropout',
      output: 'Output'
    };
    return labels[type] ?? type;
  }

  /** 将 Top-K 数值映射成条形宽度，用于展示类别概率或神经元响应强弱。 */
  topKWidth(value: number, items: Array<{ value: number }>): number {
    const max = Math.max(...items.map(item => Math.abs(item.value)), 1e-6);
    return Math.max(4, Math.min(100, Math.abs(value) / max * 100));
  }

  /** 读取 A 模式写入的网络快照；3D 页面只消费已有结果，不重新执行 forward。 */
  private readPayload(): Network3dPayload | null {
    try {
      const raw = sessionStorage.getItem(NETWORK_3D_SESSION_KEY)
        ?? localStorage.getItem(NETWORK_3D_SESSION_KEY);
      return raw ? JSON.parse(raw) as Network3dPayload : null;
    } catch {
      return null;
    }
  }

  /** 初始化 Three.js 场景、相机、渲染器和控制器，为网络层 3D 展示做准备。 */
  private initScene(host: HTMLDivElement): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101620');
    this.scene.fog = new THREE.Fog('#101620', 18, 46);

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 120);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    host.appendChild(this.renderer.domElement);
    this.renderer.domElement.addEventListener('pointerdown', this.onCanvasPointerDown);
    this.renderer.domElement.addEventListener('click', this.onCanvasClick);
    this.renderer.domElement.addEventListener('dblclick', this.onCanvasDoubleClick);
    this.renderer.domElement.addEventListener('pointermove', this.onCanvasPointerMove);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.applyInteractionMode();

    this.addLights();
    this.addFloor();
    this.buildNetworkObjects();
    this.resetCamera();

    this.resizeObserver.observe(host);
    this.resizeRenderer();
    this.animate();
  }

  /** 添加环境光、方向光和点光，让层板、特征图和连接线在 3D 空间中清晰可见。 */
  private addLights(): void {
    if (!this.scene) return;
    this.scene.add(new THREE.HemisphereLight('#e0f2fe', '#1e293b', 1.45));

    const key = new THREE.DirectionalLight('#ffffff', 2.0);
    key.position.set(-5, 8, 6);
    key.castShadow = true;
    this.scene.add(key);

    const warm = new THREE.PointLight('#fbbf24', 18, 22);
    warm.position.set(-3, 2, -6);
    this.scene.add(warm);

    const cool = new THREE.PointLight('#55c2d7', 28, 26);
    cool.position.set(5, 4, 2);
    this.scene.add(cool);
  }

  /** 添加地面网格，给网络层在深度方向上的排列提供空间参照。 */
  private addFloor(): void {
    if (!this.scene) return;
    const grid = new THREE.GridHelper(46, 46, '#42526a', '#222d3d');
    grid.position.y = -4.4;
    this.scene.add(grid);
  }

  /** 根据 layerViews 创建所有 3D 层对象、连接线和 forward 流动粒子。 */
  private buildNetworkObjects(): void {
    if (!this.scene || !this.payload) return;
    this.networkGroup = new THREE.Group();
    this.networkGroup.position.set(MODEL_OFFSET_X, MODEL_OFFSET_Y, 0);
    this.scene.add(this.networkGroup);

    const spacing = 2.55;
    let cursorZ = -((this.layerViews.length - 1) * spacing) / 2;
    let previousFace: LayerFace | null = null;

    if (this.payload.inputImageUrl) {
      const imageZ = cursorZ - spacing;
      const imageSize = this.addInputImage(imageZ, this.payload.inputImageUrl);
      previousFace = {
        center: new THREE.Vector3(0, 0, imageZ + 0.04),
        width: imageSize.width,
        height: imageSize.height,
        type: 'input'
      };
    }

    for (const view of this.layerViews) {
      const group = this.createLayerObject(view);
      group.position.set(0, 0, cursorZ);
      this.networkGroup.add(group);
      this.layerObjects.set(view.layer.id, group);

      const label = this.createLabelSprite(view.layer.name, view.shapeLabel, view.layer.type);
      label.position.set(0, view.height / 2 + 0.7, cursorZ);
      this.networkGroup.add(label);

      const currentBackFace: LayerFace = {
        center: new THREE.Vector3(0, 0, cursorZ - view.depth / 2 - 0.04),
        width: view.width,
        height: view.height,
        layerId: view.layer.id,
        type: view.layer.type
      };
      if (previousFace) {
        this.addSemanticConnections(previousFace, currentBackFace);
      }

      previousFace = {
        center: new THREE.Vector3(0, 0, cursorZ + view.depth / 2 + 0.04),
        width: view.width,
        height: view.height,
        layerId: view.layer.id,
        type: view.layer.type
      };
      cursorZ += spacing;
    }

    this.updateSelectionMaterials();
  }

  /** 在输入层前放置输入图片，让用户看到网络从哪张图开始提取特征。 */
  private addInputImage(z: number, imageUrl: string): { width: number; height: number } {
    const ratio = this.imageRatioFromShape(this.payload?.layerShapes[this.payload.layers[0]?.id] ?? []);
    const width = 2.35 * ratio.width;
    const height = 2.35 * ratio.height;
    if (!this.networkGroup) return { width, height };
    this.textureLoader.load(imageUrl, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(width, height),
        new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide })
      );
      plane.position.set(0, 0, z);
      this.networkGroup?.add(plane);

      const frame = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(width + 0.08, height + 0.08, 0.06)),
        new THREE.LineBasicMaterial({ color: '#d8f3f8', transparent: true, opacity: 0.75 })
      );
      frame.position.copy(plane.position);
      this.networkGroup?.add(frame);
    });
    return { width, height };
  }

  /** 按层类型创建 3D 表达：卷积/池化画特征图堆叠，Flatten 画展开带，Dense 画神经元网格。 */
  private createLayerObject(view: Network3dLayerView): THREE.Group {
    const group = new THREE.Group();
    group.userData['layerId'] = view.layer.id;
    const snapshot = this.payload?.layerSnapshots?.[view.layer.id];

    if (this.shouldRenderFeatureStack(view)) {
      this.addFeatureStack(group, view, snapshot);
    } else if (view.layer.type === 'flatten') {
      this.addFlattenRibbon(group, view);
    } else if (view.layer.type === 'dense' || view.layer.type === 'output') {
      this.addUnitGrid(group, view, view.layer.type === 'output' ? 'output' : 'dense');
    } else if (view.shape.length === 1) {
      this.addUnitGrid(group, view, 'vector');
    } else if (view.shape.length === 2) {
      this.addMatrixSheet(group, view);
    } else {
      this.addAbstractLayer(group, view);
    }

    if (snapshot?.previewImageUrl && this.shouldRenderFeatureStack(view)) {
      this.addPreviewPlane(group, snapshot.previewImageUrl, view.width * 0.72, view.height * 0.72, view.depth / 2 + 0.035);
    }

    return group;
  }

  /** 绘制多通道特征图堆叠；每张薄片代表一个卷积输出通道或池化后的通道。 */
  private addFeatureStack(group: THREE.Group, view: Network3dLayerView, snapshot?: Network3dLayerSnapshot): void {
    const channels = Math.max(1, view.shape.length === 3 ? view.shape[2] : snapshot?.channelPreviews.length ?? 1);
    const visible = Math.min(channels, 10);
    const step = visible <= 1 ? 0 : view.depth / Math.max(visible - 1, 1);
    for (let i = 0; i < visible; i += 1) {
      const scale = 1 - i * 0.018;
      const material = this.layerMaterial(view, 0.5 + i / Math.max(visible, 1) * 0.18);
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry(view.width * scale, view.height * scale, 0.06),
        material
      );
      slab.position.z = -view.depth / 2 + i * step;
      slab.castShadow = true;
      slab.receiveShadow = true;
      slab.userData['layerId'] = view.layer.id;
      group.add(slab);
      this.registerMaterial(view.layer.id, material);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(slab.geometry),
        new THREE.LineBasicMaterial({ color: '#e8eef7', transparent: true, opacity: 0.28 })
      );
      slab.add(edges);
    }

    if (channels > visible) {
      const badge = this.createBadgeSprite(`+${channels - visible} ch`);
      badge.position.set(view.width / 2 + 0.28, -view.height / 2 - 0.12, 0);
      group.add(badge);
    }
  }

  /** 用带状块表现 Flatten，把二维/三维特征图展开成一维向量。 */
  private addFlattenRibbon(group: THREE.Group, view: Network3dLayerView): void {
    this.addLayerFrame(group, view, '#fbbf24');
    const bars = Math.min(Math.max(view.shape[0] ?? 32, 18), 72);
    const cols = Math.min(bars, Math.max(8, Math.ceil(Math.sqrt(bars) * 1.8)));
    const rows = Math.ceil(bars / cols);
    const cell = Math.min(view.width / cols, view.height / rows) * 0.7;
    for (let i = 0; i < bars; i += 1) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const material = this.layerMaterial(view, 0.62 + (i % cols) / cols * 0.18);
      const bar = new THREE.Mesh(new THREE.BoxGeometry(cell * 1.35, cell * 0.74, 0.14), material);
      bar.position.set((col - (cols - 1) / 2) * cell * 1.48, ((rows - 1) / 2 - row) * cell * 1.12, 0);
      bar.userData['layerId'] = view.layer.id;
      group.add(bar);
      this.registerMaterial(view.layer.id, material);
    }
    if ((view.shape[0] ?? 0) > bars) {
      const badge = this.createBadgeSprite(`${view.shape[0]} values`);
      badge.position.set(0, -view.height / 2 - 0.34, 0);
      group.add(badge);
    }
  }

  /** 绘制 Dense/Output 的神经元网格，每个小块代表一个隐藏单元或类别单元。 */
  private addUnitGrid(group: THREE.Group, view: Network3dLayerView, role: 'dense' | 'output' | 'vector'): void {
    const frameColor = role === 'output' ? '#f87171' : role === 'dense' ? '#c084fc' : view.color;
    this.addLayerFrame(group, view, frameColor);
    const units = Math.max(1, view.shape[0] ?? this.layerUnits(view.layer));
    const visible = Math.min(units, role === 'output' ? 80 : 64);
    const cols = Math.ceil(Math.sqrt(visible));
    const rows = Math.ceil(visible / cols);
    const cell = Math.min(view.width / Math.max(cols, 1), view.height / Math.max(rows, 1)) * 0.58;
    const stepX = cols <= 1 ? 0 : (view.width * 0.78) / (cols - 1);
    const stepY = rows <= 1 ? 0 : (view.height * 0.78) / (rows - 1);
    for (let i = 0; i < visible; i += 1) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const material = this.layerMaterial(view, 0.68 + (i % 7) * 0.025);
      const node = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.075, cell), Math.max(0.075, cell), role === 'output' ? 0.2 : 0.16),
        material
      );
      node.position.set((col - (cols - 1) / 2) * stepX, ((rows - 1) / 2 - row) * stepY, 0);
      node.userData['layerId'] = view.layer.id;
      group.add(node);
      this.registerMaterial(view.layer.id, material);

      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(node.geometry),
        new THREE.LineBasicMaterial({ color: '#f8fafc', transparent: true, opacity: role === 'output' ? 0.32 : 0.22 })
      );
      node.add(edge);
    }
    if (units > visible) {
      const badge = this.createBadgeSprite(`${units} units`);
      badge.position.set(0, -view.height / 2 - 0.35, 0);
      group.add(badge);
    }
  }

  /** 绘制二维矩阵型张量，用于展示抽象响应面或矩阵结构。 */
  private addMatrixSheet(group: THREE.Group, view: Network3dLayerView): void {
    this.addLayerFrame(group, view, view.color);
    const [rows, cols] = view.shape as [number, number];
    const visibleRows = Math.min(rows, 12);
    const visibleCols = Math.min(cols, 12);
    const cell = Math.min(view.width / Math.max(visibleCols, 1), view.height / Math.max(visibleRows, 1)) * 0.72;
    for (let row = 0; row < visibleRows; row += 1) {
      for (let col = 0; col < visibleCols; col += 1) {
        const material = this.layerMaterial(view, 0.58 + ((row + col) % 6) * 0.025);
        const tile = new THREE.Mesh(new THREE.BoxGeometry(cell, cell, 0.12), material);
        tile.position.set((col - (visibleCols - 1) / 2) * cell * 1.26, ((visibleRows - 1) / 2 - row) * cell * 1.26, 0);
        tile.userData['layerId'] = view.layer.id;
        group.add(tile);
        this.registerMaterial(view.layer.id, material);
      }
    }
    if (rows > visibleRows || cols > visibleCols) {
      const badge = this.createBadgeSprite(`${rows} x ${cols}`);
      badge.position.set(0, -view.height / 2 - 0.35, 0);
      group.add(badge);
    }
  }

  /** 判断当前层是否适合用特征图堆叠表示，通常是 [H, W, C] 的图像类张量。 */
  private shouldRenderFeatureStack(view: Network3dLayerView): boolean {
    return view.shape.length === 3
      && ['input', 'conv2d', 'pool2d', 'residual', 'activation', 'dropout'].includes(view.layer.type);
  }

  /** 给层对象添加外框，区分不同网络层在 3D 空间中的边界。 */
  private addLayerFrame(group: THREE.Group, view: Network3dLayerView, color: string): void {
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(view.width, view.height, Math.max(view.depth, 0.18))),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.34 })
    );
    frame.userData['layerId'] = view.layer.id;
    group.add(frame);
  }

  /** 为无法具体展开的层创建抽象块，例如激活层、Dropout 或未知 shape 层。 */
  private addAbstractLayer(group: THREE.Group, view: Network3dLayerView): void {
    const material = this.layerMaterial(view, 0.68);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(view.width, view.height, view.depth), material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData['layerId'] = view.layer.id;
    group.add(mesh);
    this.registerMaterial(view.layer.id, material);

    const ring = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: '#e8eef7', transparent: true, opacity: 0.36 })
    );
    mesh.add(ring);
  }

  /** 把 A 模式传来的特征图预览贴到层对象上，避免 3D 页面重新计算张量。 */
  private addPreviewPlane(group: THREE.Group, imageUrl: string, width: number, height: number, z: number): void {
    this.textureLoader.load(imageUrl, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(width, height),
        new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.88, side: THREE.DoubleSide })
      );
      plane.position.z = z;
      plane.userData['layerId'] = group.userData['layerId'];
      group.add(plane);
    });
  }

  /** 创建层对象材质，颜色来自层类型，透明度用于表现特征图堆叠层次。 */
  private layerMaterial(view: Network3dLayerView, opacity: number): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color: view.color,
      roughness: 0.42,
      metalness: 0.08,
      transparent: true,
      opacity
    });
  }

  /** 记录某层使用过的材质，选中层时可以统一调整高亮效果。 */
  private registerMaterial(layerId: number, material: THREE.MeshStandardMaterial): void {
    const list = this.layerMaterials.get(layerId) ?? [];
    list.push(material);
    this.layerMaterials.set(layerId, list);
  }

  /** 绘制层间连接线，表示上一层输出张量被送入下一层继续前向传播。 */
  private addSemanticConnections(from: LayerFace, to: LayerFace): void {
    if (!this.networkGroup) return;
    const style = this.connectionStyle(to.type);
    const sourcePoints = this.facePoints(from.center, from.width, from.height, style.points);
    const targetPoints = this.facePoints(to.center, to.width, to.height, style.points);

    sourcePoints.forEach((source, index) => {
      const target = targetPoints[index % targetPoints.length];
      const mid = source.clone().lerp(target, 0.5);
      const radial = new THREE.Vector3(source.x - from.center.x, source.y - from.center.y, 0);
      if (radial.lengthSq() > 0.0001) mid.add(radial.normalize().multiplyScalar(style.bulge));
      const curve = new THREE.QuadraticBezierCurve3(source, mid, target);
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(18));
      const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: style.color, transparent: true, opacity: style.opacity })
      );
      this.networkGroup?.add(line);
      this.addFlowParticle(curve, style.color, index);
    });
  }

  /** 根据目标层类型设置连接线数量、颜色和弯曲程度，突出卷积/池化/Dense 的不同连接语义。 */
  private connectionStyle(type?: LayerType): { color: string; opacity: number; points: number; bulge: number } {
    if (type === 'conv2d') return { color: '#55c2d7', opacity: 0.58, points: 9, bulge: 0.08 };
    if (type === 'pool2d') return { color: '#4ade80', opacity: 0.52, points: 5, bulge: 0.04 };
    if (type === 'flatten') return { color: '#fbbf24', opacity: 0.48, points: 12, bulge: 0.1 };
    if (type === 'dense' || type === 'output') return { color: '#c084fc', opacity: 0.42, points: 16, bulge: 0.05 };
    return { color: '#9fb4ce', opacity: 0.44, points: 9, bulge: 0.06 };
  }

  /** 沿连接曲线添加流动粒子，动态表达一次 forward 中特征张量的传递方向。 */
  private addFlowParticle(curve: THREE.QuadraticBezierCurve3, color: string, seed: number): void {
    if (!this.networkGroup) return;
    const particle = new THREE.Mesh(
      new THREE.SphereGeometry(0.045, 12, 8),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 })
    );
    this.networkGroup.add(particle);
    this.flowPaths.push({
      curve,
      particle,
      offset: (seed * 0.137) % 1,
      speed: 0.0028 + (seed % 5) * 0.00018
    });
  }

  /** 在层的前后表面生成连接点，连接点数量越多越能表现 Dense/Flatten 的多路特征流。 */
  private facePoints(center: THREE.Vector3, width: number, height: number, count: number): THREE.Vector3[] {
    if (count === 5) {
      const x = width * 0.31;
      const y = height * 0.31;
      return [
        new THREE.Vector3(center.x - x, center.y - y, center.z),
        new THREE.Vector3(center.x + x, center.y - y, center.z),
        center.clone(),
        new THREE.Vector3(center.x - x, center.y + y, center.z),
        new THREE.Vector3(center.x + x, center.y + y, center.z)
      ];
    }
    if (count === 12) return this.gridFacePoints(center, width, height, 4, 3);
    if (count === 16) return this.gridFacePoints(center, width, height, 4, 4);
    return this.gridFacePoints(center, width, height, 3, 3);
  }

  /** 按网格生成表面连接点，让层间连接线从特征图或神经元区域均匀发出。 */
  private gridFacePoints(center: THREE.Vector3, width: number, height: number, cols: number, rows: number): THREE.Vector3[] {
    const points: THREE.Vector3[] = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const x = cols <= 1 ? 0 : (col / (cols - 1) - 0.5) * width * 0.72;
        const y = rows <= 1 ? 0 : (0.5 - row / (rows - 1)) * height * 0.72;
        points.push(new THREE.Vector3(center.x + x, center.y + y, center.z));
      }
    }
    return points;
  }

  /** 创建层标签贴图，显示层名、层类型和 shape，便于在 3D 场景中对应回网络结构。 */
  private createLabelSprite(name: string, shapeLabel: string, type: LayerType): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 560;
    canvas.height = 166;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgba(16, 22, 32, .88)';
      ctx.strokeStyle = 'rgba(166, 184, 208, .45)';
      ctx.lineWidth = 3;
      ctx.roundRect(12, 12, 536, 140, 18);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#f8fafc';
      ctx.font = '700 34px Segoe UI, sans-serif';
      ctx.fillText(this.truncate(name, 22), 34, 61);
      ctx.fillStyle = '#9ddce8';
      ctx.font = '500 23px Consolas, monospace';
      ctx.fillText(this.truncate(`${this.typeLabel(type)} · ${shapeLabel}`, 31), 34, 110);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.scale.set(2.55, 0.76, 1);
    return sprite;
  }

  /** 创建小徽标，用于提示隐藏的通道数、神经元数或矩阵尺寸。 */
  private createBadgeSprite(text: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgba(16, 22, 32, .86)';
      ctx.strokeStyle = 'rgba(251, 191, 36, .58)';
      ctx.lineWidth = 3;
      ctx.roundRect(14, 18, 228, 54, 16);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#fde68a';
      ctx.font = '700 24px Segoe UI, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, 128, 53);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.scale.set(1.1, 0.42, 1);
    return sprite;
  }

  /** 根据输入张量的高宽比设置图片平面比例，避免输入图在 3D 中被拉伸。 */
  private imageRatioFromShape(shape: TensorShape): { width: number; height: number } {
    if (shape.length !== 3) return { width: 1, height: 1 };
    const [height, width] = shape;
    const max = Math.max(width, height, 1);
    return { width: width / max, height: height / max };
  }

  /** 估算 Dense/Output 的单元数量，3D 神经元网格用它决定需要画多少小块。 */
  private layerUnits(layer: NetworkLayer): number {
    if (layer.type === 'dense' || layer.type === 'output') return layer.params.units;
    return 32;
  }

  /** 根据选中/悬停状态更新层材质，帮助定位当前正在分析的网络层。 */
  private updateSelectionMaterials(): void {
    for (const [layerId, materials] of this.layerMaterials) {
      const selected = layerId === this.selectedLayerId;
      const hovered = layerId === this.hoverLayerId;
      for (const material of materials) {
        material.opacity = selected ? 0.96 : hovered ? 0.82 : Math.min(material.opacity, 0.72);
        material.emissive = new THREE.Color(selected ? '#12354a' : hovered ? '#3b2a0d' : '#000000');
        material.needsUpdate = true;
      }
    }
  }

  /** 将当前交互模式应用到 OrbitControls，控制鼠标左键是旋转还是平移。 */
  private applyInteractionMode(): void {
    if (!this.controls) return;
    this.controls.enableRotate = true;
    this.controls.enablePan = true;
    this.controls.mouseButtons = this.interactionMode === 'pan'
      ? {
          LEFT: THREE.MOUSE.PAN,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.ROTATE
        }
      : {
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN
        };
    this.controls.touches = this.interactionMode === 'pan'
      ? {
          ONE: THREE.TOUCH.PAN,
          TWO: THREE.TOUCH.DOLLY_ROTATE
        }
      : {
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN
        };
    this.controls.update();
  }

  private onCanvasPointerDown = (event: PointerEvent): void => {
    this.pointerDown = { x: event.clientX, y: event.clientY };
  };

  private onCanvasClick = (event: MouseEvent): void => {
    if (this.wasDrag(event)) return;
    const layerId = this.pickLayerId(event);
    if (layerId !== -1) this.selectLayer(layerId);
  };

  private onCanvasDoubleClick = (event: MouseEvent): void => {
    if (this.wasDrag(event)) return;
    const layerId = this.pickLayerId(event);
    if (layerId !== -1) this.focusLayer(layerId);
  };

  private onCanvasPointerMove = (event: PointerEvent): void => {
    const layerId = this.pickLayerId(event);
    if (layerId === this.hoverLayerId) return;
    this.hoverLayerId = layerId;
    this.updateSelectionMaterials();
  };

  /** 用射线检测鼠标指向的 3D 对象，并沿父节点查找对应的 layerId。 */
  private pickLayerId(event: MouseEvent | PointerEvent): number {
    if (!this.camera || !this.renderer || !this.networkGroup) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersects = this.raycaster.intersectObjects(this.networkGroup.children, true);
    for (const hit of intersects) {
      let object: THREE.Object3D | null = hit.object;
      while (object) {
        const layerId = object.userData['layerId'];
        if (typeof layerId === 'number') return layerId;
        object = object.parent;
      }
    }
    return -1;
  }

  /** 判断一次点击是否其实是拖拽，避免旋转场景时误选中网络层。 */
  private wasDrag(event: MouseEvent | PointerEvent): boolean {
    if (!this.pointerDown) return false;
    return Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y) > 5;
  }

  /** 根据容器尺寸调整 renderer 和相机比例，保证 3D 网络在窗口变化后不变形。 */
  private resizeRenderer(): void {
    if (!this.stageRef || !this.renderer || !this.camera) return;
    const host = this.stageRef.nativeElement;
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  /** 每帧更新控制器和 forward 粒子位置，并渲染当前 3D 网络场景。 */
  private animate(): void {
    this.animationId = requestAnimationFrame(() => this.animate());
    this.controls?.update();
    if (this.flowPlaying) {
      for (const path of this.flowPaths) {
        path.offset = (path.offset + path.speed) % 1;
        path.particle.position.copy(path.curve.getPoint(path.offset));
      }
    }
    if (this.scene && this.camera) {
      this.renderer?.render(this.scene, this.camera);
    }
  }

  /** 释放 Three.js geometry/material，页面销毁时避免 WebGL 资源泄漏。 */
  private disposeObject(object?: THREE.Object3D): void {
    object?.traverse((item) => {
      const mesh = item as THREE.Mesh;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (Array.isArray(material)) {
        material.forEach(entry => entry.dispose());
      } else {
        material?.dispose();
      }
    });
  }

  /** 截断过长层名或 shape 文本，避免 Canvas 标签中文字溢出。 */
  private truncate(value: string, max: number): string {
    return value.length > max ? `${value.slice(0, max - 1)}...` : value;
  }
}
