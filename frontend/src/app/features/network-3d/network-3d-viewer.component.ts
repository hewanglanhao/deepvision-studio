import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TensorShape } from '../../sim-models';
import { buildNetwork3dLayerViews } from './network-3d-layout';
import { NETWORK_3D_SESSION_KEY, Network3dLayerView, Network3dPayload } from './network-3d.models';

@Component({
  selector: 'app-network-3d-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="viewer-shell">
      <header class="viewer-topbar">
        <div>
          <div class="viewer-title">{{ payload?.title || 'Network 3D Viewer' }}</div>
          <div class="viewer-subtitle">{{ subtitle }}</div>
        </div>
        <div class="viewer-actions">
          <button type="button" (click)="resetCamera()" [disabled]="!payload">重置视角</button>
          <button type="button" (click)="closeWindow()">关闭</button>
        </div>
      </header>

      @if (payload) {
        <main class="viewer-main">
          <section class="stage-wrap">
            <div #stage class="stage"></div>
          </section>
          <aside class="layer-panel">
            <div class="panel-title">网络层</div>
            <div class="layer-list">
              @for (item of layerViews; track item.layer.id) {
                <button
                  type="button"
                  class="layer-row"
                  [class.active]="item.layer.id === selectedLayerId"
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
      background: #0f172a;
      color: #e5edf7;
      font-family: 'Manrope', 'Segoe UI', 'Noto Sans SC', sans-serif;
    }

    .viewer-shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .viewer-topbar {
      min-height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 16px;
      border-bottom: 1px solid rgba(148, 163, 184, .24);
      background: rgba(15, 23, 42, .94);
    }

    .viewer-title {
      font-size: 16px;
      font-weight: 800;
      color: #f8fafc;
    }

    .viewer-subtitle {
      margin-top: 2px;
      font-size: 12px;
      color: #94a3b8;
    }

    .viewer-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button {
      border: 1px solid rgba(148, 163, 184, .38);
      border-radius: 8px;
      padding: 7px 11px;
      background: rgba(30, 41, 59, .86);
      color: #e5edf7;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }

    button:hover:not(:disabled) {
      border-color: #38bdf8;
      background: rgba(14, 165, 233, .18);
    }

    button:disabled {
      opacity: .45;
      cursor: not-allowed;
    }

    .viewer-main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
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

    .layer-panel {
      min-height: 0;
      border-left: 1px solid rgba(148, 163, 184, .22);
      background: rgba(15, 23, 42, .78);
      padding: 12px;
      overflow-y: auto;
    }

    .panel-title {
      color: #94a3b8;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      margin-bottom: 8px;
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
      background: rgba(30, 41, 59, .62);
    }

    .layer-row.active {
      border-color: #38bdf8;
      background: rgba(14, 165, 233, .18);
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
      color: #94a3b8;
      font-size: 11px;
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
      color: #94a3b8;
      font-size: 13px;
    }

    @media (max-width: 900px) {
      .viewer-main {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(420px, 1fr) auto;
      }

      .stage-wrap {
        min-height: 420px;
      }

      .layer-panel {
        max-height: 34vh;
        border-left: 0;
        border-top: 1px solid rgba(148, 163, 184, .22);
      }
    }
  `]
})
export class Network3dViewerComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('stage') private stageRef?: ElementRef<HTMLDivElement>;

  payload: Network3dPayload | null = null;
  layerViews: Network3dLayerView[] = [];
  selectedLayerId = -1;

  private scene?: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private renderer?: THREE.WebGLRenderer;
  private controls?: OrbitControls;
  private networkGroup?: THREE.Group;
  private animationId = 0;
  private readonly layerObjects = new Map<number, THREE.Object3D>();
  private readonly resizeObserver = new ResizeObserver(() => this.resizeRenderer());

  get subtitle(): string {
    if (!this.payload) return '等待 A/B/C/D 模式传入网络层数据';
    return `${this.payload.sourceMode} · ${this.layerViews.length} 层 · ${new Date(this.payload.createdAt).toLocaleString()}`;
  }

  ngOnInit(): void {
    this.payload = this.readPayload();
    if (!this.payload) return;
    this.selectedLayerId = this.payload.selectedLayerId;
    this.layerViews = buildNetwork3dLayerViews(
      this.payload.layers,
      this.payload.layerShapes,
      this.payload.shapeHints
    );
  }

  ngAfterViewInit(): void {
    if (!this.payload || !this.stageRef) return;
    this.initScene(this.stageRef.nativeElement);
  }

  ngOnDestroy(): void {
    if (this.animationId) cancelAnimationFrame(this.animationId);
    this.resizeObserver.disconnect();
    this.controls?.dispose();
    this.renderer?.dispose();
    this.networkGroup?.traverse((object) => {
      const mesh = object as THREE.Mesh;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (Array.isArray(material)) {
        material.forEach((item) => item.dispose());
      } else {
        material?.dispose();
      }
    });
  }

  resetCamera(): void {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(5.6, 5.2, 10.4);
    this.controls.target.set(0, 2.1, 0);
    this.controls.update();
  }

  closeWindow(): void {
    window.close();
  }

  focusLayer(layerId: number): void {
    this.selectedLayerId = layerId;
    const object = this.layerObjects.get(layerId);
    if (!object || !this.camera || !this.controls) return;
    const position = new THREE.Vector3();
    object.getWorldPosition(position);
    this.controls.target.copy(position);
    this.camera.position.set(position.x + 4.2, position.y + 3.1, position.z + 5.7);
    this.controls.update();
  }

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

  private readPayload(): Network3dPayload | null {
    try {
      const raw = sessionStorage.getItem(NETWORK_3D_SESSION_KEY)
        ?? localStorage.getItem(NETWORK_3D_SESSION_KEY);
      return raw ? JSON.parse(raw) as Network3dPayload : null;
    } catch {
      return null;
    }
  }

  private initScene(host: HTMLDivElement): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0f172a');
    this.scene.fog = new THREE.Fog('#0f172a', 18, 42);

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    host.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.addLights();
    this.addFloor();
    this.buildNetworkObjects();
    this.resetCamera();

    this.resizeObserver.observe(host);
    this.resizeRenderer();
    this.animate();
  }

  private addLights(): void {
    if (!this.scene) return;
    this.scene.add(new THREE.HemisphereLight('#e0f2fe', '#1e293b', 1.6));

    const key = new THREE.DirectionalLight('#ffffff', 2.1);
    key.position.set(-5, 8, 6);
    key.castShadow = true;
    this.scene.add(key);

    const fill = new THREE.PointLight('#38bdf8', 24, 24);
    fill.position.set(4, 3, -4);
    this.scene.add(fill);
  }

  private addFloor(): void {
    if (!this.scene) return;
    const grid = new THREE.GridHelper(40, 40, '#334155', '#1e293b');
    grid.position.y = -4.6;
    this.scene.add(grid);
  }

  private buildNetworkObjects(): void {
    if (!this.scene || !this.payload) return;
    this.networkGroup = new THREE.Group();
    this.networkGroup.position.y = 4.05;
    this.scene.add(this.networkGroup);

    const spacing = 2.35;
    let cursorZ = -((this.layerViews.length - 1) * spacing) / 2;
    let previousFace: { center: THREE.Vector3; width: number; height: number } | null = null;

    if (this.payload.inputImageUrl) {
      const imageZ = cursorZ - spacing;
      const imageSize = this.addInputImage(imageZ, this.payload.inputImageUrl);
      previousFace = {
        center: new THREE.Vector3(0, 0, imageZ + 0.04),
        width: imageSize.width,
        height: imageSize.height
      };
    }

    for (const view of this.layerViews) {
      const mesh = this.createLayerBox(view);
      mesh.position.set(0, 0, cursorZ);
      this.networkGroup.add(mesh);
      this.layerObjects.set(view.layer.id, mesh);

      const label = this.createLabelSprite(view.layer.name, view.shapeLabel);
      label.position.set(0, view.height / 2 + 0.72, cursorZ);
      this.networkGroup.add(label);

      const currentBackFace = new THREE.Vector3(0, 0, cursorZ - view.depth / 2);
      if (previousFace) {
        this.addRadiatingConnections(previousFace, {
          center: currentBackFace,
          width: view.width,
          height: view.height
        });
      }
      previousFace = {
        center: new THREE.Vector3(0, 0, cursorZ + view.depth / 2),
        width: view.width,
        height: view.height
      };
      cursorZ += spacing;
    }
  }

  private addInputImage(z: number, imageUrl: string): { width: number; height: number } {
    const ratio = this.imageRatioFromShape(this.payload?.layerShapes[this.payload.layers[0]?.id] ?? []);
    const width = 2.35 * ratio.width;
    const height = 2.35 * ratio.height;
    if (!this.networkGroup) return { width, height };
    new THREE.TextureLoader().load(imageUrl, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(width, height),
        new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide })
      );
      plane.position.set(0, 0, z);
      this.networkGroup?.add(plane);

      const frame = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(width + 0.06, height + 0.06, 0.06)),
        new THREE.LineBasicMaterial({ color: '#bae6fd' })
      );
      frame.position.copy(plane.position);
      this.networkGroup?.add(frame);
    });
    return { width, height };
  }

  private createLayerBox(view: Network3dLayerView): THREE.Mesh {
    const geometry = new THREE.BoxGeometry(view.width, view.height, view.depth);
    const material = new THREE.MeshStandardMaterial({
      color: view.color,
      roughness: 0.42,
      metalness: 0.08,
      transparent: true,
      opacity: view.layer.id === this.selectedLayerId ? 0.96 : 0.82
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: '#e0f2fe', transparent: true, opacity: 0.45 })
    );
    mesh.add(edges);
    return mesh;
  }

  private addRadiatingConnections(
    from: { center: THREE.Vector3; width: number; height: number },
    to: { center: THREE.Vector3; width: number; height: number }
  ): void {
    if (!this.networkGroup) return;
    const group = this.networkGroup;
    const material = new THREE.LineBasicMaterial({ color: '#93c5fd', transparent: true, opacity: 0.5 });
    const sourcePoints = this.facePoints(from.center, from.width, from.height);
    const targetPoints = this.facePoints(to.center, to.width, to.height);

    sourcePoints.forEach((source, index) => {
      const target = targetPoints[index % targetPoints.length];
      const mid = source.clone().lerp(target, 0.5);
      const radial = new THREE.Vector3(
        source.x - from.center.x,
        source.y - from.center.y,
        0
      );
      if (radial.lengthSq() > 0.0001) {
        radial.normalize().multiplyScalar(0.12);
        mid.add(radial);
      }
      const curve = new THREE.QuadraticBezierCurve3(source, mid, target);
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(12));
      group.add(new THREE.Line(geometry, material));
    });
  }

  private facePoints(center: THREE.Vector3, width: number, height: number): THREE.Vector3[] {
    const x = width * 0.34;
    const y = height * 0.34;
    return [
      new THREE.Vector3(center.x - x, center.y - y, center.z),
      new THREE.Vector3(center.x, center.y - y, center.z),
      new THREE.Vector3(center.x + x, center.y - y, center.z),
      new THREE.Vector3(center.x - x, center.y, center.z),
      center.clone(),
      new THREE.Vector3(center.x + x, center.y, center.z),
      new THREE.Vector3(center.x - x, center.y + y, center.z),
      new THREE.Vector3(center.x, center.y + y, center.z),
      new THREE.Vector3(center.x + x, center.y + y, center.z)
    ];
  }

  private createLabelSprite(name: string, shapeLabel: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgba(15, 23, 42, .84)';
      ctx.strokeStyle = 'rgba(148, 163, 184, .45)';
      ctx.lineWidth = 3;
      ctx.roundRect(12, 12, 488, 132, 18);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#f8fafc';
      ctx.font = '700 34px Segoe UI, sans-serif';
      ctx.fillText(this.truncate(name, 20), 34, 62);
      ctx.fillStyle = '#93c5fd';
      ctx.font = '500 25px Consolas, monospace';
      ctx.fillText(this.truncate(shapeLabel, 28), 34, 108);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.scale.set(2.35, 0.74, 1);
    return sprite;
  }

  private imageRatioFromShape(shape: TensorShape): { width: number; height: number } {
    if (shape.length !== 3) return { width: 1, height: 1 };
    const [height, width] = shape;
    const max = Math.max(width, height, 1);
    return { width: width / max, height: height / max };
  }

  private resizeRenderer(): void {
    if (!this.stageRef || !this.renderer || !this.camera) return;
    const host = this.stageRef.nativeElement;
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  private animate(): void {
    this.animationId = requestAnimationFrame(() => this.animate());
    this.controls?.update();
    if (this.scene && this.camera) {
      this.renderer?.render(this.scene, this.camera);
    }
  }

  private truncate(value: string, max: number): string {
    return value.length > max ? `${value.slice(0, max - 1)}...` : value;
  }
}
