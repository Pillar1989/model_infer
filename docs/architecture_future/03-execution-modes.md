# 3. Execution Modes - Technical Design

This document provides technical implementation details for each execution mode.

---

## 3.1 Context Design

All modes share a common context structure for data passing between stages.

### FrameContext

```
┌─────────────────────────────────────────────────────────────────┐
│  FrameContext                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  frame_id: int64           -- Unique frame identifier           │
│  timestamp: time_point     -- Frame capture time                │
│                                                                 │
│  image: Image              -- Original image (for ROI crop)     │
│                                                                 │
│  results: {                -- Model outputs (populated per model│
│    model1: any,            -- e.g., detections                  │
│    model2: any,            -- e.g., keypoints                   │
│    model3: any             -- e.g., features                    │
│  }                                                              │
│                                                                 │
│  metadata: {               -- Optional metadata                 │
│    source: string,         -- Video path or camera ID           │
│    original_size: (w, h)   -- Pre-resize dimensions             │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Context Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│ Create  │ ──▶ │ Model 1 │ ──▶ │ Model 2 │ ──▶ │ Destroy         │
│ Context │     │ writes  │     │ reads 1 │     │ Context         │
│         │     │ results │     │ writes 2│     │                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Single Mode Implementation

### Execution Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│  Single Mode                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Create context                                              │
│     ctx = { frame_id, image, results = {} }                     │
│                                                                 │
│  2. Call Lua preprocess                                         │
│     tensor = model.preprocess(ctx)                              │
│                                                                 │
│  3. C++ inference                                               │
│     output = session.run(tensor)                                │
│                                                                 │
│  4. Call Lua postprocess                                        │
│     result = model.postprocess(output, ctx)                     │
│     ctx.results.model1 = result                                 │
│                                                                 │
│  5. Return ctx.results                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Error Handling

| Error | Behavior |
|-------|----------|
| Preprocess returns nil | Skip inference, return empty result |
| Inference fails | Propagate error to caller |
| Postprocess fails | Propagate error to caller |

---

## 3.3 Serial Mode Implementation

### Execution Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│  Serial Mode (N models)                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Create context                                              │
│     ctx = { frame_id, image, results = {} }                     │
│                                                                 │
│  2. For i = 1 to N:                                             │
│                                                                 │
│     a. Call Lua preprocess                                      │
│        tensor = models[i].preprocess(ctx)                       │
│                                                                 │
│        -- ctx contains results from models 1..i-1               │
│        -- preprocess can access ctx.results.model1, etc.        │
│                                                                 │
│     b. If tensor is nil:                                        │
│        -- Skip this model (e.g., no detections)                 │
│        ctx.results.model[i] = nil                               │
│        continue                                                 │
│                                                                 │
│     c. C++ inference                                            │
│        output = sessions[i].run(tensor)                         │
│                                                                 │
│     d. Call Lua postprocess                                     │
│        result = models[i].postprocess(output, ctx)              │
│        ctx.results.model[i] = result                            │
│                                                                 │
│  3. Return ctx.results                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Dependency

```
Model 1                    Model 2                    Model 3
────────────────────────────────────────────────────────────────────
preprocess(ctx)            preprocess(ctx)            preprocess(ctx)
  │                          │                          │
  │ ctx.image              ctx.image                ctx.image
  │                          │                          │
  └─▶ tensor               ctx.results.model1       ctx.results.model1
                              │                       ctx.results.model2
                              └─▶ tensor                │
                                                        └─▶ tensor
```

### Skip Logic

```
Model 1 returns 0 detections:
  ctx.results.det = []

Model 2 preprocess checks:
  if #ctx.results.det == 0 then
    return nil  -- Signal to skip
  end

Pipeline sees nil:
  ctx.results.pose = nil  -- Mark as skipped
  continue to Model 3

Model 3 can also check:
  if ctx.results.det == nil or #ctx.results.det == 0 then
    return nil
  end
```

---

## 3.4 Parallel-Sync Mode Implementation

### Execution Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│  Parallel-Sync Mode                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Create context                                              │
│     ctx = { frame_id, image, results = {} }                     │
│                                                                 │
│  2. Preprocess all models (can be parallel or sequential)       │
│     tensors = []                                                │
│     for i = 1 to N:                                             │
│       tensors[i] = models[i].preprocess(ctx)                    │
│                                                                 │
│  3. Launch parallel inferences                                  │
│     futures = []                                                │
│     for i = 1 to N:                                             │
│       if tensors[i] != nil:                                     │
│         futures[i] = async { sessions[i].run(tensors[i]) }      │
│                                                                 │
│  4. Barrier: wait all                                           │
│     outputs = []                                                │
│     for i = 1 to N:                                             │
│       if futures[i] != nil:                                     │
│         outputs[i] = futures[i].get()                           │
│                                                                 │
│  5. Postprocess all models                                      │
│     for i = 1 to N:                                             │
│       if outputs[i] != nil:                                     │
│         ctx.results.model[i] = models[i].postprocess(outputs[i])│
│                                                                 │
│  6. Fusion (optional)                                           │
│     ctx.results.fused = fusion_func(ctx.results)                │
│                                                                 │
│  7. Return ctx.results                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Parallelism Options

| Option | Description | Use Case |
|--------|-------------|----------|
| **Multi-Thread** | std::async on CPU | CPU-only inference |
| **Multi-NPU** | Different NPU cores | NPU with multiple cores |
| **Multi-Device** | CPU + NPU | Hybrid execution |

### Barrier Semantics

```
                Model 1               Model 2
                ───────               ───────
   T=0ms        Start                 Start
   T=28ms       Complete              ...
   T=35ms       (wait)                Complete
   T=35ms       ─────────── BARRIER ───────────
   T=35ms       Fusion begins
```

### Fusion Design

Fusion is a Lua function that combines results from all models:

```
fusion(results) → combined_result

Input:  { model1: boxes, model2: mask }
Output: { instances: [{ box, mask }, ...] }
```

---

## 3.5 Parallel-Async Mode Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Parallel-Async Architecture                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────────────────────────────┐  │
│  Frame Input ────▶ │           Frame Dispatcher              │  │
│                    │  - Assigns frame_id                     │  │
│                    │  - Routes to model workers by interval   │ │
│                    └─────────────────────────────────────────┘  │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  Worker 1   │     │  Worker 2   │     │  Worker 3   │       │
│   │  interval=1 │     │  interval=10│     │  interval=30│       │
│   │             │     │             │     │             │       │
│   │  ┌───────┐  │     │  ┌───────┐  │     │  ┌───────┐  │       │
│   │  │Queue  │  │     │  │Queue  │  │     │  │Queue  │  │       │
│   │  └───┬───┘  │     │  └───┬───┘  │     │  └───┬───┘  │       │
│   │      │      │     │      │      │     │      │      │       │
│   │  Process    │     │  Process    │     │  Process    │       │
│   │      │      │     │      │      │     │      │      │       │
│   │      ▼      │     │      ▼      │     │      ▼      │       │
│   │  Result     │     │  Result     │     │  Result     │       │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│          │                   │                   │              │
│          └───────────────────┼───────────────────┘              │
│                              ▼                                  │
│                    ┌─────────────────────────────────────────┐  │
│                    │         Result Aggregator               │  │
│                    │  - Stores latest result per model       │  │
│                    │  - Combines by frame_id                 │  │
│                    └─────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                        Final Result                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Worker Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│  Worker Configuration                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model 1 (Detection):                                           │
│    frame_interval: 1      -- Process every frame                │
│    max_queue_size: 2      -- Buffer up to 2 frames              │
│    drop_on_overflow: true -- Drop oldest if queue full          │
│                                                                 │
│  Model 2 (Scene):                                               │
│    frame_interval: 10     -- Process every 10th frame           │
│    max_queue_size: 1      -- Only keep latest                   │
│    drop_on_overflow: true                                       │
│                                                                 │
│  Model 3 (Anomaly):                                             │
│    frame_interval: 30     -- Process every 30th frame           │
│    max_queue_size: 1                                            │
│    drop_on_overflow: true                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Result Aggregation

```
┌─────────────────────────────────────────────────────────────────┐
│  Result Aggregation Strategy                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame 0:                                                       │
│    det[0] ← Model 1 result (fresh)                              │
│    scene[0] ← Model 2 result (fresh)                            │
│    anomaly[0] ← Model 3 result (fresh)                          │
│                                                                 │
│  Frame 1-9:                                                     │
│    det[N] ← Model 1 result (fresh)                              │
│    scene[N] ← scene[0] (reuse)                                  │
│    anomaly[N] ← anomaly[0] (reuse)                              │
│                                                                 │
│  Frame 10:                                                      │
│    det[10] ← Model 1 result (fresh)                             │
│    scene[10] ← Model 2 result (fresh)                           │
│    anomaly[10] ← anomaly[0] (reuse)                             │
│                                                                 │
│  Frame 30:                                                      │
│    det[30] ← Model 1 result (fresh)                             │
│    scene[30] ← scene[30] or scene[20] (fresh or recent)         │
│    anomaly[30] ← Model 3 result (fresh)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Mode Configuration

### Lua Configuration

```
Pipeline Configuration (conceptual):

Single Mode:
  mode = "single"
  models = [{ name, path, preprocess, postprocess }]

Serial Mode:
  mode = "serial"
  models = [
    { name, path, preprocess, postprocess },
    { name, path, preprocess, postprocess },
    { name, path, preprocess, postprocess }
  ]

Parallel-Sync Mode:
  mode = "parallel_sync"
  models = [...]
  fusion = function(results) ... end

Parallel-Async Mode:
  mode = "parallel_async"
  models = [
    { name, path, ..., frame_interval = 1 },
    { name, path, ..., frame_interval = 10 },
    { name, path, ..., frame_interval = 30 }
  ]
  aggregator = function(results) ... end
```

---

## 3.7 Mode Comparison: Implementation Complexity

| Aspect | Single | Serial | Parallel-Sync | Parallel-Async |
|--------|--------|--------|---------------|----------------|
| Threading | None | None | Thread pool | Thread pool |
| Synchronization | None | None | Barrier | Queues |
| Data Sharing | N/A | Context | Context | Per-worker |
| Error Handling | Simple | Cascade | Fail-fast | Per-worker |
| Memory | 1 session | N sessions | N sessions | N sessions + queues |
| Latency | T_total | T_sum | T_max | T_main_model |

---

## 3.8 Extension: Frame-Level Pipeline (Reserved)

### Current 4 Modes vs Frame-Level Pipeline

```
Current 4 modes: Model-level parallelism (within same frame)
  Frame N: [Model1] → [Model2] → [Model3]
  Frame N+1:                                [Model1] → [Model2] → ...

Frame-level pipeline: Frame-level overlap (across frames)
  Frame N:   [pre] → [infer] → [post]
  Frame N+1:         [pre] → [infer] → [post]
                     ↑ Overlapping execution
```

### Reserved Interface

```
Pipeline {
    // Existing synchronous interface
    Results run(Image& frame);

    // Reserved async interface (Phase 2)
    void submit(Image& frame);       // Non-blocking submit
    Results try_get_result();        // Non-blocking get
    Results wait_result();           // Blocking wait
}
```

### Implementation Points (Phase 2)

```
Frame-level pipeline requires:
  1. Double/triple buffering: input_buffer, inference_buffer, output_buffer
  2. FrameContext Independent lifecycle management
  3. Thread division: preprocess_thread, inference_thread, postprocess_thread
  4. Synchronization primitives: condition_variable or lock-free queue

Not implemented currently, interfaces reserved for future extension
See 01-core-abstractions.md
```
