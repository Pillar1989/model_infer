# 5. Resource Management

This document describes session and memory management for multi-model pipelines.

---

## 5.1 Session Management

### SessionManager Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  SessionManager                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Responsibilities:                                              │
│  - Load and unload model sessions                               │
│  - Track resource usage                                         │
│  - Enforce 3-model limit                                        │
│                                                                 │
│  Session Pool (max 3):                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ Session 1   │ │ Session 2   │ │ Session 3   │                │
│  │ (det)       │ │ (pose)      │ │ (reid)      │                │
│  │ ~100MB      │ │ ~50MB       │ │ ~30MB       │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│                                                                 │
│  Total: 180MB                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Session States

```
┌─────────────────────────────────────────────────────────────────┐
│  Session State Machine                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌──────────────┐                                   │
│              │   Unloaded   │ ◀──────────────────┐              │
│              │   (0 MB)     │                    │              │
│              └──────┬───────┘                    │              │
│                     │ load()                     │ unload()     │
│                     ▼                            │              │
│              ┌──────────────┐                    │              │
│              │   Loaded     │ ───────────────────┘              │
│              │   (X MB)     │                                   │
│              └──────────────┘                                   │
│                                                                 │
│  Note: "Standby" state removed for simplicity                   │
│  (Memory savings minimal on embedded systems)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loading Policy

| Policy | Description | Use Case |
|--------|-------------|----------|
| **Eager** | Load all sessions at pipeline creation | Production, predictable latency |
| **Lazy** | Load on first use | Development, memory-constrained |

---

## 5.2 Memory Budget

### Typical Memory Allocation

```
┌─────────────────────────────────────────────────────────────────┐
│  Memory Budget (256MB total)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Weights:                                                 │
│    ├── Detection (YOLO):     ~100 MB                            │
│    ├── Pose:                  ~50 MB                            │
│    └── ReID:                  ~30 MB                            │
│    Subtotal:                 ~180 MB                            │
│                                                                 │
│  Runtime Buffers:                                               │
│    ├── Input tensors:         ~5 MB  (3 × 1.5MB)                │
│    ├── Output tensors:       ~10 MB  (varies)                   │
│    └── Intermediate:          ~5 MB                             │
│    Subtotal:                 ~20 MB                             │
│                                                                 │
│  Image/Frame Buffer:                                            │
│    └── Original image:        ~5 MB  (1080p × 3ch)              │
│                                                                 │
│  System Overhead:                                               │
│    └── ONNX Runtime, Lua:    ~40 MB                             │
│                                                                 │
│  Total:                      ~245 MB                            │
│  Headroom:                    ~11 MB                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Constraints

| Scenario | Max Models | Typical Memory |
|----------|------------|----------------|
| Lightweight | 3 × small | ~100 MB |
| Standard | 2 × medium + 1 × small | ~180 MB |
| Heavy | 1 × large + 2 × small | ~200 MB |

---

## 5.3 Buffer Reuse

### Buffer Pool

```
┌─────────────────────────────────────────────────────────────────┐
│  Buffer Pool Strategy                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Common Buffer Sizes (preallocated):                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Size Bucket        Count    Total                       │   │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  1.2 MB (640×640)   2        2.4 MB   (YOLO input)       │   │
│  │  2.8 MB (84×8400)   2        5.6 MB   (YOLO output)      │   │
│  │  0.4 MB (256×128)   16       6.4 MB   (ReID input×16)    │   │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Benefits:                                                      │
│  - Avoid malloc/free per frame                                  │
│  - Predictable memory usage                                     │
│  - Reduced fragmentation                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Buffer Lifecycle

```
Frame Processing:

  1. Acquire buffer from pool
     buffer = pool.acquire(size)

  2. Use for tensor storage
     tensor = Tensor(buffer, shape)

  3. Process through model
     output = session.run(tensor)

  4. After frame complete, release
     pool.release(buffer)
```

---

## 5.4 NPU Resource Management

### Multi-Core NPU

```
┌─────────────────────────────────────────────────────────────────┐
│  NPU Core Assignment (Dual-Core Example)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Serial Mode:                                                   │
│    Core 0: Model 1 → Model 2 → Model 3  (time-shared)           │
│    Core 1: (idle)                                               │
│                                                                 │
│  Parallel-Sync Mode:                                            │
│    Core 0: Model 1                                              │
│    Core 1: Model 2                                              │
│    (Model 3 waits for free core)                                │
│                                                                 │
│  Parallel-Async Mode:                                           │
│    Core 0: Model 1 (high-rate, dedicated)                       │
│    Core 1: Model 2/3 (low-rate, shared)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Device Assignment

| Mode | Strategy |
|------|----------|
| Serial | All models on same core (no contention) |
| Parallel-Sync | Spread across cores if available |
| Parallel-Async | Dedicate core to high-rate model |

---

## 5.5 Resource Limits

### Hard Limits

| Resource | Limit | Enforced By |
|----------|-------|-------------|
| Sessions | 3 max | Pipeline constructor |
| Memory | Platform-dependent | OS/Runtime |
| NPU Cores | Hardware-dependent | Driver |

### Soft Limits

| Resource | Default | Configurable |
|----------|---------|--------------|
| Input queue size | 2 | Yes |
| Tensor cache size | 4 | Yes |
| Timeout per model | 5000ms | Yes |

---

## 5.6 Resource Monitoring

### Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│  Runtime Metrics                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Memory:                                                        │
│    - total_allocated: bytes currently allocated                 │
│    - peak_usage: maximum bytes ever allocated                   │
│    - buffer_pool_hits: reuse count                              │
│    - buffer_pool_misses: new allocation count                   │
│                                                                 │
│  Timing (per model):                                            │
│    - preprocess_ms: average preprocess time                     │
│    - inference_ms: average inference time                       │
│    - postprocess_ms: average postprocess time                   │
│                                                                 │
│  Throughput:                                                    │
│    - frames_processed: total count                              │
│    - frames_per_second: current FPS                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Monitoring API

```
Pipeline provides:
  get_memory_usage() → bytes
  get_timing_stats() → { model1: stats, model2: stats, ... }
  get_throughput() → fps
```

---

## 5.7 Resource Recovery

### Cleanup on Error

```
┌─────────────────────────────────────────────────────────────────┐
│  Error Recovery                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  On inference error:                                            │
│    1. Release current frame's tensors                           │
│    2. Log error with context                                    │
│    3. Continue processing next frame                            │
│    (Sessions remain loaded)                                     │
│                                                                 │
│  On session error:                                              │
│    1. Attempt reload once                                       │
│    2. If reload fails, mark session as failed                   │
│    3. Pipeline can continue with remaining models               │
│                                                                 │
│  On memory exhaustion:                                          │
│    1. Clear buffer pool cache                                   │
│    2. Force garbage collection (Lua)                            │
│    3. Retry allocation                                          │
│    4. If still fails, abort pipeline                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Graceful Degradation

| Failure | Degradation |
|---------|-------------|
| Model 2/3 fails | Continue with Model 1 only |
| NPU fails | Fall back to CPU (if supported) |
| Memory pressure | Reduce batch size |

---

## 5.8 Extension: DeviceType Fine-Grained (Reserved for Plan B)

### Current DeviceType

```
┌─────────────────────────────────────────────────────────────────┐
│  DeviceType (Current)                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum DeviceType {                                              │
│      CPU,       -- Regular CPU memory                           │
│      NPU        -- NPU device memory (generic)                  │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Reserved Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  DeviceType (Reserved for Plan B)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum DeviceType {                                              │
│      CPU,                                                       │
│      CPU_PINNED,     -- Pinned memory (zero-copy DMA)           │
│      NPU,            -- Generic NPU                             │
│                                                                 │
│      // Platform-specific (Reserved)                            │
│      RK_DMA,         -- Rockchip DMA buffer                     │
│      RK_CMA,         -- Rockchip CMA buffer                     │
│      SG_ION,         -- Sophgo ION buffer                       │
│      HAILO_HOST,     -- Hailo host-side buffer                  │
│  }                                                              │
│                                                                 │
│  Extension timing: Add when platform-specific optimization neede│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5.9 Extension: DeviceBuffer Sync Mechanism (Reserved for Plan B)

### Current Design

- DeviceBuffer is an abstract interface, implemented by CpuMemory
- No explicit sync, assumes CPU access takes effect immediately

### Reserved Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  DeviceBuffer Sync Interface (Reserved)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  class DeviceBuffer {                                           │
│      virtual void sync(SyncDirection dir) = 0;                  │
│  };                                                             │
│                                                                 │
│  enum SyncDirection {                                           │
│      HOST_TO_DEVICE,   -- CPU → NPU                             │
│      DEVICE_TO_HOST,   -- NPU → CPU                             │
│      BIDIRECTIONAL     -- Bidirectional sync                    │
│  };                                                             │
│                                                                 │
│  Usage scenarios:                                               │
│    - Before NPU inference: sync(HOST_TO_DEVICE)                 │
│    - After NPU inference: sync(DEVICE_TO_HOST)                  │
│                                                                 │
│  Current implementation (CpuMemory):                            │
│    sync() { /* no-op */ }                                       │
│                                                                 │
│  Future implementation (NpuMemory):                             │
│    sync() { flush_cache(); invalidate_cache(); }                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5.10 NPU Structured Output (Reserved)

### Special Handling for NPUs like Hailo

```
┌─────────────────────────────────────────────────────────────────┐
│  Structured Output (Reserved for Plan B)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Some NPUs (e.g., Hailo) have built-in NMS, output is already   │
│  structured data:                                               │
│                                                                 │
│  NPU output: { boxes: [...], scores: [...], classes: [...] }    │
│                                                                 │
│  Rather than traditional:                                       │
│    Tensor [1, 84, 8400]                                         │
│                                                                 │
│  Handling approach:                                             │
│    1. Lua postprocess detects output type                       │
│    2. If already structured data, return directly               │
│    3. If Tensor, parse in traditional way                       │
│                                                                 │
│  This requires Lua interface to handle both return types        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5.11 Reference Documents

| Topic | Reference |
|-------|-----------|
| Image/Tensor Low-Level Design | [01. Core Abstractions](01-core-abstractions.md) |
| Frame-Level Pipeline Reservation | [03. Execution Modes](03-execution-modes.md) Section 3.8 |
| Pipeline Interface Reservation | [04. Pipeline Design](04-pipeline-design.md) Section 4.10 |
