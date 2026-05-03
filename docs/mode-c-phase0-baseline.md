# Mode C Phase 0 Baseline

## 1. Purpose

This document freezes the rewrite baseline for turning `cnn-explainer` into a native Angular `Mode C` module.

Phase 0 is complete when we have:

- a clear feature inventory
- a keep/rewrite/defer decision for each major capability
- a resource inventory for model, images, article, and figures
- a visual alignment baseline derived from `Mode A` and `Mode B`
- explicit Phase 1 entry criteria

## 2. Current Baseline

### 2.1 Current integration mode

Today `Mode C` is still an iframe host:

- Angular host page: `frontend/src/app/pages/mode-c/`
- embedded static app: `frontend/public/modules/cnn-explainer/`
- external source of truth: `D:\VS Code\cnn-explainer`

This means the platform shell and auth state already belong to Angular, but the CNN explainer experience still belongs to a separate frontend runtime.

### 2.2 Upstream app structure

The upstream `cnn-explainer` is a standalone Svelte application.

- `src/main.js`
  mounts the app directly on `document.body`
- `src/App.svelte`
  composes the global overlay, header, and explainer
- `src/Explainer.svelte`
  delegates almost everything to `Overview.svelte`
- `src/overview/Overview.svelte`
  is the main orchestration component for model loading, image switching, network drawing, hover/click interactions, and detail mode switching
- `src/detail-view/`
  contains the teaching panels for convolution, activation, pooling, hyperparameters, and softmax
- `src/article/Article.svelte`
  contains the long-form teaching article and auxiliary sections
- `src/stores.js`
  provides the shared state model used by the Svelte app

### 2.3 Architectural reading

The important conclusion from the code review is:

- the data concepts are reusable
- the teaching flow is reusable
- the rendering and state architecture should be rewritten

We should preserve the product idea, not transplant the Svelte page structure as-is.

## 3. Feature Inventory

### 3.1 Priority definition

- `P0`: must exist in the first Angular-native version
- `P1`: should exist soon after the first native version
- `P2`: can be deferred or reduced

### 3.2 Inventory table

| Area | Current source | Priority | Rewrite decision | Notes |
|---|---|---:|---|---|
| Platform shell and topbar | current Angular `Mode C` host | P0 | keep and extend | Reuse `Mode A/B` shell language directly |
| Auth status display | current Angular `Mode C` host | P0 | keep | No iframe communication required |
| CNN overview graph | `src/overview/Overview.svelte` plus draw utilities | P0 | rewrite | Core value of Module C |
| Example image switching | `Overview.svelte` | P0 | rewrite | Needed to demonstrate inference changes |
| Model loading and forward inference | `src/utils/cnn-tf.js` | P0 | rewrite around Angular service | Logic may be adapted, but API surface should become Angular-native |
| Layer selection and highlight | `Overview.svelte` + D3 draw helpers | P0 | rewrite | Needed for the learning flow |
| Detail panel container | `Overview.svelte` + `detail-view/*` | P0 | rewrite | Native Angular panel switching |
| Convolution detail view | `detail-view/Convolutionview.svelte` | P1 | rewrite | First deep-dive panel after MVP overview |
| Activation/ReLU detail view | `detail-view/Activationview.svelte` | P1 | rewrite | High teaching value |
| Pooling detail view | `detail-view/Poolview.svelte` | P1 | rewrite | High teaching value |
| Softmax detail view | `detail-view/Softmaxview.svelte` | P1 | rewrite | Important end-stage interpretation panel |
| Hyperparameter explainer views | `detail-view/Hyperparameter*` | P2 | defer or simplify | Useful, but not required for initial parity |
| Long-form article section | `article/Article.svelte` | P1 | rewrite in structured Angular form | Should be split into sections instead of a single monolithic template |
| Video tutorial section | `Article.svelte` + YouTube dependency | P2 | drop by default | Already removed from embedded version |
| Recommender overlay / modal extras | `App.svelte` overlay usage | P2 | reassess later | Keep only if we find real product value |
| Global Svelte store topology | `src/stores.js` | P0 | replace | Convert into typed Angular state service |
| Direct global DOM control | D3/Svelte page logic | P0 | replace | Must not survive into native Angular implementation |

## 4. Resource Inventory

### 4.1 Model resources

Located under upstream `public/assets/data/`:

- `model.json`
- `group1-shard1of1.bin`
- `nn_10.json`

Decision:

- keep all three during rewrite investigation
- treat `model.json` and weight shard as runtime assets
- treat `nn_10.json` as a reference artifact until we confirm whether it is still needed by the Angular version

### 4.2 Sample image resources

Located under upstream `public/assets/img/`.

Confirmed teaching samples include:

- `boat`
- `bug`
- `bus`
- `car`
- `espresso`
- `koala`
- `orange`
- `panda`
- `pepper`
- `pizza`

Decision:

- keep the sample image set for the Angular MVP
- do not expand the dataset during rewrite

### 4.3 Figure and article assets

Located under upstream `public/assets/figures/`.

Notable assets:

- overview and detail demo gifs
- softmax animation gif
- ReLU graph image
- UI helper icons and preview figures

Decision:

- keep figures that directly support the article or detail panels
- remove assets that only support deprecated video/tutorial flows

### 4.4 Font assets

Located under upstream `public/assets/font/`:

- `A Love of Thunder.ttf`
- `Coffee and Tea.ttf`

Decision:

- do not bring these into the Angular rewrite by default
- prefer the established platform typography used by `Mode A/B`
- revisit only if a specific title treatment truly needs it

## 5. UI Alignment Baseline

### 5.1 Reference source

The visual baseline for Angular-native `Mode C` should come from:

- `frontend/src/app/pages/mode-a/mode-a-page.component.html`
- `frontend/src/app/pages/mode-a/mode-a-page.component.css`
- `frontend/src/app/pages/mode-b/mode-b-page.component.html`
- `frontend/src/app/pages/mode-b/mode-b-page.component.css`

### 5.2 Elements that must stay aligned

`Mode C` should reuse the same visual language for:

- app shell spacing and page background
- topbar height and translucent card-like surface
- brand block layout
- auth/status pill styling
- button classes and interaction states
- card radius, border, and shadow system
- font scale for labels, metadata, and section headings

### 5.3 Elements that can be Mode C-specific

`Mode C` can have its own identity in:

- the visualization canvas layout
- layer cards or SVG node styling
- article section layout
- legend and explanation blocks

Constraint:

- "different" is acceptable
- "visibly from another website" is not

### 5.4 Visual translation rule

When there is a conflict between preserving upstream look and staying inside the platform design system:

- interaction semantics from `cnn-explainer` win
- platform shell and component styling from `Mode A/B` win

## 6. Rewrite Boundaries

### 6.1 Keep conceptually

We keep these concepts:

- CNN layer-by-layer teaching flow
- sample-driven inference exploration
- network overview to detail-panel learning path
- article-assisted explanation model

### 6.2 Rewrite technically

We rewrite these parts completely:

- app entry and routing
- state management
- component boundaries
- DOM event handling
- global script assumptions
- iframe-specific workarounds

### 6.3 Explicit non-goals for Phase 1 MVP

These are not required for the first native delivery:

- full 1:1 visual cloning of the original Svelte page
- YouTube tutorial recovery
- exact duplication of every article interaction
- preservation of Svelte global store naming

## 7. Proposed Angular Mapping

### 7.1 Feature structure

Recommended target structure:

```text
frontend/src/app/features/mode-c-explainer/
  components/
    shell/
    overview/
    detail-panels/
    article/
    shared/
  services/
    mode-c-model.service.ts
    mode-c-state.service.ts
    mode-c-assets.service.ts
  models/
    mode-c.types.ts
  utils/
    mode-c-d3.ts
    mode-c-mappers.ts
```

### 7.2 State mapping

The current Svelte stores should collapse into a typed Angular state service with at least:

- current input sample
- model metadata and load status
- network graph data
- selected layer
- selected detail panel
- hover/focus state
- softmax/detail transient state

## 8. Risks

### 8.1 Highest-risk items

- `Overview.svelte` is highly orchestration-heavy and mixes loading, state, rendering, and interaction logic
- the D3 interaction model currently assumes direct DOM control
- article and detail content are broad enough to create scope creep

### 8.2 Scope protection rules

To keep the rewrite healthy:

- do not port the entire page in one pass
- do not chase perfect pixel parity with the old Svelte UI
- do not mix iframe compatibility patches into the native Angular implementation

## 9. Phase 1 Entry Criteria

Phase 1 can start once the team accepts the following decisions:

- overview graph is the first-class MVP target
- model loading will move into an Angular service
- article migration is not blocking the first native shell
- video tutorial content stays removed
- `Mode A/B` shell tokens are the visual source of truth

## 10. Phase 0 Deliverables

This phase now produces:

- this baseline document
- the existing rewrite roadmap in `docs/mode-c-angular-rewrite-plan.md`
- a frozen P0/P1/P2 scope decision for the first implementation round

## 11. Recommended Next Step

Start Phase 1 by building the Angular-native shell and feature folder skeleton first, before touching D3 migration.

The first implementation slice should be:

1. create `features/mode-c-explainer/`
2. move `Mode C` page into a shell + feature composition structure
3. add typed state and assets services
4. render a static native placeholder overview area using platform styling

That gives us a clean native landing zone for the real overview rewrite.
