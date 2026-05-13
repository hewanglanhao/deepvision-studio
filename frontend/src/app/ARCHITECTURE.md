# Frontend Structure

The app is organized by ownership first, then by shared capability.

## core

Application-wide infrastructure that is not owned by a single mode.

- `core/api`: low-level HTTP client.
- `core/auth`: auth model, service, and login/register page.

## shared

Reusable building blocks used by more than one mode.

- `shared/components`: global shell components such as the top bar.
- `shared/network`: reusable network structure overview component.
- `shared/network-3d`: 3D network snapshot viewer used by Mode A and experiment compare.
- `shared/llm`: floating assistant, prompts, chat service, and LLM types.
- `shared/forward`: forward-pass backend client and saved forward-record models/services.
- `shared/training`: training runtime, dataset, and collaboration clients used by Mode B surfaces.
- `shared/simulation`: frontend tensor/model helpers.

`shared/simulation/sim-engine.ts` is not just throwaway mock code. It is still used for shape inference, parameter counting, image preprocessing, frontend fallback forward execution, and checkpoint structure visualization. Keep backend computation as the source of truth where available, but keep this module as the frontend's deterministic model utility layer.

## modes

Mode-owned pages and private feature code.

- `modes/mode-a`: forward-pass visualization page.
- `modes/mode-b`: training page plus `experiment-compare` and `training-collaboration`.
- `modes/mode-c`: Mode C page and its CNN explainer implementation.
- `modes/mode-d`: Mode D page and its backpropagation explainer implementation.
- `modes/mode-e`: Transformer explainer page, types, and engine.

Mode folders may import from `core` and `shared`. Shared code should not import from a mode folder.

## shell

App-level screens that are not one of the A-E modes, such as the home portal.

## public assets

Angular's `public` folder is still useful for large or third-party static assets that should be served as files rather than imported into the TypeScript bundle.

- `public/mode-a/samples`: sample images and manifest used by Mode A/B forward-pass image pickers.
- `public/mode-c/cnn-explainer`: legacy CNN Explainer static runtime, TensorFlow.js vendor file, pretrained model data, images, and fonts used by Mode C.
- `public/favicon.ico`: app favicon.

Keep mode-owned static assets under `public/mode-*/...`. Only use a top-level public folder for genuinely global assets.
