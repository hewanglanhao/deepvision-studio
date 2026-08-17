# Mode D runtime assets

This directory intentionally keeps large runtime files out of Git. A complete local installation contains:

```text
mode-d-assets/
├─ asset-manifest.json
├─ model-v2/gpt2.onnx.part0 ... part62
├─ models/Xenova/gpt2/
└─ vendor/
   ├─ onnxruntime/
   └─ transformers/
```

From `frontend/`, install and verify the assets with:

```bash
npm ci
npm run setup:mode-d
npm run check:mode-d
```

The installer downloads the integrated GPT-2 ONNX chunks from the official
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) site, pins the tokenizer revision,
copies browser runtimes from the locked npm dependencies, and verifies all 63 model chunks.

See [`docs/mode-d-assets.md`](../../../docs/mode-d-assets.md) for details and troubleshooting.
