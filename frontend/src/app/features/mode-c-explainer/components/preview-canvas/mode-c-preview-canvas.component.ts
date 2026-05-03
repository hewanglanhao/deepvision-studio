import { AfterViewInit, Component, ElementRef, Input, OnChanges, ViewChild } from '@angular/core';

@Component({
  selector: 'app-mode-c-preview-canvas',
  templateUrl: './mode-c-preview-canvas.component.html',
  styleUrl: './mode-c-preview-canvas.component.css'
})
export class ModeCPreviewCanvasComponent implements AfterViewInit, OnChanges {
  @Input() matrix: number[][] = [];
  @Input() grayscale = false;

  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;

  ngAfterViewInit(): void {
    this.draw();
  }

  ngOnChanges(): void {
    this.draw();
  }

  private draw(): void {
    if (!this.canvasRef) return;
    const canvas = this.canvasRef.nativeElement;
    const context = canvas.getContext('2d');
    if (!context || !this.matrix.length || !this.matrix[0]?.length) return;

    const size = 48;
    canvas.width = size;
    canvas.height = size;
    const imageData = context.createImageData(size, size);
    this.paintScalarField(this.matrix, imageData.data, size, size, this.grayscale);
    context.putImageData(imageData, 0, 0);
  }

  private paintScalarField(
    matrix: number[][],
    pixels: Uint8ClampedArray,
    targetHeight: number,
    targetWidth: number,
    grayscale: boolean
  ): void {
    const sourceHeight = matrix.length;
    const sourceWidth = matrix[0]?.length ?? 0;
    const flattened = matrix.flat().filter(value => Number.isFinite(value));
    const hasNegative = flattened.some(value => value < 0);
    const hasPositive = flattened.some(value => value > 0);
    const maxAbs = this.computePercentile(flattened.map(value => Math.abs(value)), 0.92);
    const minValue = flattened.length ? Math.min(...flattened) : 0;
    const maxValue = flattened.length ? Math.max(...flattened) : 1;
    const signedRange = maxAbs > 1e-6 ? maxAbs : 1;
    const positiveRange = maxValue > 1e-6 ? maxValue : 1;
    const negativeRange = Math.abs(minValue) > 1e-6 ? Math.abs(minValue) : 1;

    for (let row = 0; row < targetHeight; row++) {
      for (let col = 0; col < targetWidth; col++) {
        const sourceRow = Math.min(sourceHeight - 1, Math.floor((row / targetHeight) * sourceHeight));
        const sourceCol = Math.min(sourceWidth - 1, Math.floor((col / targetWidth) * sourceWidth));
        const value = matrix[sourceRow][sourceCol];
        const color = grayscale
          ? this.mapGrayscaleColor(Math.max(0, Math.min(1, value)))
          : hasNegative && hasPositive
            ? this.mapActivationColor(value / signedRange)
            : hasPositive
              ? this.mapPositiveActivationColor(value / positiveRange)
              : this.mapNegativeActivationColor(value / negativeRange);

        const index = (row * targetWidth + col) * 4;
        pixels[index] = color[0];
        pixels[index + 1] = color[1];
        pixels[index + 2] = color[2];
        pixels[index + 3] = 255;
      }
    }
  }

  private mapActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(-1, Math.min(1, normalized));
    const intensity = Math.pow(Math.abs(clamped), 0.55);
    const white = 250 - Math.round(intensity * 10);

    if (clamped >= 0) {
      return [
        Math.round(white - intensity * 70),
        Math.round(white - intensity * 110),
        Math.round(white + intensity * 4)
      ];
    }

    return [
      Math.round(white + intensity * 4),
      Math.round(white - intensity * 108),
      Math.round(white - intensity * 70)
    ];
  }

  private mapPositiveActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(1, normalized));
    const intensity = Math.pow(clamped, 0.45);
    return [
      Math.round(248 - intensity * 82),
      Math.round(250 - intensity * 120),
      255
    ];
  }

  private mapNegativeActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(-1, Math.min(0, normalized));
    const intensity = Math.pow(Math.abs(clamped), 0.45);
    return [
      255,
      Math.round(246 - intensity * 116),
      Math.round(246 - intensity * 82)
    ];
  }

  private mapGrayscaleColor(normalized: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(1, normalized));
    const value = Math.round(clamped * 255);
    return [value, value, value];
  }

  private computePercentile(values: number[], percentile: number): number {
    if (!values.length) return 0;
    const sorted = [...values].sort((left, right) => left - right);
    const position = Math.min(sorted.length - 1, Math.max(0, Math.floor(percentile * (sorted.length - 1))));
    return sorted[position] ?? 0;
  }
}
