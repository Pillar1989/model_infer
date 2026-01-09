# 4. Pipeline Design

This document describes the C++ Pipeline class and Lua integration.

---

## 4.1 Pipeline Class Overview

### Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Pipeline** | Orchestrate model execution based on mode |
| **Session** | Single model inference (existing) |
| **Lua Script** | Preprocess, postprocess, fusion |

### Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline                                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Configuration:                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  mode: Mode (Single/Serial/ParallelSync/ParallelAsync)  │    │
│  │  max_models: 3 (compile-time constant)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Model Slots (max 3):                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  slot[0]: { session, preprocess_ref, postprocess_ref }  │    │
│  │  slot[1]: { session, preprocess_ref, postprocess_ref }  │    │
│  │  slot[2]: { session, preprocess_ref, postprocess_ref }  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Operations:                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  run(image) → results                                   │    │
│  │  run_async(image) → future<results>                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Execution Mode Enum

```
┌─────────────────────────────────────────────────────────────────┐
│  Mode Enum                                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum class Mode {                                              │
│      Single,        // 1 model                                  │
│      Serial,        // 2-3 models, sequential                   │
│      ParallelSync,  // 2-3 models, concurrent + barrier         │
│      ParallelAsync  // 2-3 models, independent rates            │
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.3 Model Slot Structure

Each model slot contains:

```
┌─────────────────────────────────────────────────────────────────┐
│  ModelSlot                                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  name: string              -- Model identifier (e.g., "det")    │
│  session: Session*         -- ONNX session for inference        │
│                                                                 │
│  preprocess: LuaRef        -- Lua function reference            │
│  postprocess: LuaRef       -- Lua function reference            │
│                                                                 │
│  -- Parallel-Async specific:                                    │
│  frame_interval: int       -- Process every N frames            │
│  queue_size: int           -- Input queue size                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.4 Pipeline Operations

### run() - Synchronous Execution

```
┌─────────────────────────────────────────────────────────────────┐
│  run(image) → results                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│    image: cv::Mat or Image object                               │
│                                                                 │
│  Processing:                                                    │
│    1. Create FrameContext                                       │
│    2. Execute based on mode:                                    │
│       - Single: run_single(ctx)                                 │
│       - Serial: run_serial(ctx)                                 │
│       - ParallelSync: run_parallel_sync(ctx)                    │
│       - ParallelAsync: submit_and_aggregate(ctx)                │
│    3. Return ctx.results                                        │
│                                                                 │
│  Output:                                                        │
│    Lua table with results from each model                       │
│    { model1: result1, model2: result2, ... }                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Internal Execution Methods

```
┌─────────────────────────────────────────────────────────────────┐
│  run_serial(ctx)                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for each model in slots:                                       │
│      tensor = call_lua(model.preprocess, ctx)                   │
│      if tensor is nil:                                          │
│          ctx.results[model.name] = nil                          │
│          continue                                               │
│      output = model.session.run(tensor)                         │
│      result = call_lua(model.postprocess, output, ctx)          │
│      ctx.results[model.name] = result                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  run_parallel_sync(ctx)                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  // Phase 1: Preprocess all                                     │
│  tensors = []                                                   │
│  for each model:                                                │
│      tensors.push(call_lua(model.preprocess, ctx))              │
│                                                                 │
│  // Phase 2: Parallel inference                                 │
│  futures = []                                                   │
│  for each model with non-nil tensor:                            │
│      futures.push(async { model.session.run(tensor) })          │
│                                                                 │
│  // Phase 3: Barrier + postprocess                              │
│  for each future:                                               │
│      output = future.get()  // Wait                             │
│      result = call_lua(model.postprocess, output, ctx)          │
│      ctx.results[model.name] = result                           │
│                                                                 │
│  // Phase 4: Fusion (if defined)                                │
│  if fusion_func:                                                │
│      ctx.results.fused = call_lua(fusion_func, ctx.results)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.5 Lua Integration

### Lua Function Signatures

```
┌─────────────────────────────────────────────────────────────────┐
│  Lua Function Contracts                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  preprocess(ctx) → tensor | nil                                 │
│  ──────────────────────────                                     │
│  Input:                                                         │
│    ctx.image         -- Original image                          │
│    ctx.results       -- Previous model results (Serial mode)    │
│    ctx.frame_id      -- Frame identifier                        │
│                                                                 │
│  Output:                                                        │
│    tensor            -- Input tensor for model                  │
│    nil               -- Skip this model                         │
│                                                                 │
│                                                                 │
│  postprocess(output, ctx) → result                              │
│  ─────────────────────────────────                              │
│  Input:                                                         │
│    output            -- Model output tensor                     │
│    ctx               -- Full context                            │
│                                                                 │
│  Output:                                                        │
│    result            -- Parsed result (any type)                │
│                                                                 │
│                                                                 │
│  fusion(results) → combined (ParallelSync only)                 │
│  ──────────────────────────────────────────────                 │
│  Input:                                                         │
│    results           -- { model1: r1, model2: r2, ... }         │
│                                                                 │
│  Output:                                                        │
│    combined          -- Fused result                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Calling Lua from C++

```
┌─────────────────────────────────────────────────────────────────┐
│  C++ → Lua Call Pattern                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  // Store Lua function reference during pipeline setup          │
│  LuaRef preprocess_func = lua_config["preprocess"];             │
│                                                                 │
│  // Call during execution                                       │
│  LuaRef result = preprocess_func.call(ctx_table);               │
│                                                                 │
│  // Handle nil return                                           │
│  if (result.isNil()) {                                          │
│      // Skip this model                                         │
│  } else {                                                       │
│      Tensor tensor = result.toValue<Tensor>();                  │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.6 Error Handling

### Error Types

| Error | Source | Handling |
|-------|--------|----------|
| Preprocess error | Lua exception | Propagate, abort pipeline |
| Inference error | C++ exception | Propagate, abort pipeline |
| Postprocess error | Lua exception | Propagate, abort pipeline |
| Skip signal | preprocess returns nil | Continue to next model |

### Error Propagation

```
┌─────────────────────────────────────────────────────────────────┐
│  Error Handling Strategy                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  try {                                                          │
│      // Lua preprocess                                          │
│      tensor = call_lua(preprocess, ctx)                         │
│  } catch (LuaException& e) {                                    │
│      throw PipelineError("preprocess failed: " + e.what())      │
│  }                                                              │
│                                                                 │
│  if (tensor.isNil()) {                                          │
│      // Not an error, just skip                                 │
│      return;                                                    │
│  }                                                              │
│                                                                 │
│  try {                                                          │
│      // C++ inference                                           │
│      output = session.run(tensor)                               │
│  } catch (std::exception& e) {                                  │
│      throw PipelineError("inference failed: " + e.what())       │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.7 Thread Safety

### Single & Serial Mode

- No threading, fully synchronous
- No thread safety concerns

### ParallelSync Mode

- Uses thread pool for parallel inference
- Context is read-only during parallel phase
- Results written after barrier (no race)

### ParallelAsync Mode

- Each worker has its own thread
- Queues protected by mutex
- Result aggregator uses atomic operations

---

## 4.8 Memory Management

### Session Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│  Session Lifecycle                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pipeline Creation:                                             │
│    - Load all sessions (eager) or defer (lazy)                  │
│    - Sessions owned by Pipeline                                 │
│                                                                 │
│  Pipeline Execution:                                            │
│    - Sessions remain loaded                                     │
│    - Tensors allocated per inference, freed after               │
│                                                                 │
│  Pipeline Destruction:                                          │
│    - All sessions unloaded                                      │
│    - All Lua references released                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Context Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│  Context Per Frame                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame arrives:                                                 │
│    ctx = new FrameContext(frame_id, image)                      │
│                                                                 │
│  During execution:                                              │
│    ctx.results populated by each model                          │
│                                                                 │
│  After execution:                                               │
│    results = ctx.results                                        │
│    ctx destroyed (image reference released)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.9 Design Constraints

### Hard Limits

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max models | 3 | Covers 99% scenarios, simplifies design |
| Max sessions | 3 | One per model slot |
| Max threads (ParallelSync) | 3 | One per model |

### Soft Limits (Configurable)

| Constraint | Default | Configurable |
|------------|---------|--------------|
| Queue size (Async) | 2 | Yes |
| Inference timeout | 5000ms | Yes |
| Memory limit | None | Optional |

---

## 4.10 Extension: Image/Tensor Interface Reservation

### Image Interface (Reserved for Plan B)

```
┌─────────────────────────────────────────────────────────────────┐
│  Image Device Specialization (Reserved)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Current:                                                       │
│    Image { cv::Mat data }                                       │
│                                                                 │
│  Reserved Extension:                                            │
│    Image {                                                      │
│      format: PixelFormat   -- RGB, BGR, NV12, etc.              │
│      stride: int           -- Row stride (for DMA alignment)    │
│      device: DeviceType    -- CPU, NPU_DMA, etc.                │
│      data: DeviceBuffer    -- Device-specific storage           │
│                                                                 │
│      // ROI operations                                          │
│      view(x, y, w, h) → Image   -- Zero-copy view               │
│      crop(x, y, w, h) → Image   -- Copy to new buffer           │
│    }                                                            │
│                                                                 │
│  See 01-core-abstractions.md                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tensor Interface (Reserved for Plan B)

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor Multi-Type Support (Reserved)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Current:                                                       │
│    Tensor { float32 only, NCHW layout }                         │
│                                                                 │
│  Reserved Extension:                                            │
│    Tensor {                                                     │
│      dtype: DType          -- float32, int8, uint8, float16     │
│      layout: Layout        -- NCHW, NHWC, NC1HWC2               │
│      quant_params: {       -- Quantization params (int8 only)   │
│        scale: float                                             │
│        zero_point: int                                          │
│      }                                                          │
│    }                                                            │
│                                                                 │
│  See 01-core-abstractions.md                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Async Interface (Reserved for Plan B)

```
┌─────────────────────────────────────────────────────────────────┐
│  Frame-Level Pipeline Interface (Reserved)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Current:                                                       │
│    Results run(Image& frame)        -- Sync interface           │
│                                                                 │
│  Reserved Extension:                                            │
│    void submit(Image& frame)        -- Non-blocking submit      │
│    Results try_get_result()         -- Non-blocking get         │
│    Results wait_result()            -- Blocking wait            │
│                                                                 │
│  Purpose: Frame-level pipeline (N+1 pre || N infer || N-1 post) │
│                                                                 │
│  See 03-execution-modes.md Section 3.8                          │
│        01-core-abstractions.md                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
