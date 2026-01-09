# 8. Architecture Summary

---

## 8.1 Design Philosophy

### Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Lua for logic** | All pre/post processing in Lua scripts |
| **C++ for speed** | Model inference only in C++ |
| **Simplicity** | Max 3 models, 4 execution modes |
| **Embedded-first** | Memory and CPU efficiency |

### Responsibility Split

```
┌─────────────────────────────────────────────────────────────────┐
│  Responsibility Distribution                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Lua Script:                                                    │
│    ✓ Preprocess each model (resize, normalize, crop ROI)        │
│    ✓ Postprocess each model (parse output, NMS, etc.)           │
│    ✓ Configure pipeline (mode, model paths)                     │
│    ✓ Fusion logic (Parallel-Sync mode)                          │
│    ✓ Business logic (filtering, skip conditions)                │
│                                                                 │
│  C++ Pipeline:                                                  │
│    ✓ Load model sessions (max 3)                                │
│    ✓ Execute inference                                          │
│    ✓ Orchestrate by mode (serial, parallel, async)              │
│    ✓ Manage resources (memory, threads)                         │
│    ✗ NO business logic                                          │
│    ✗ NO data parsing                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8.2 Execution Mode Summary

### Four Modes

| Mode | Models | Sync | Use Case |
|------|--------|------|----------|
| **Single** | 1 | N/A | Detection, classification |
| **Serial** | 2-3 | Sequential | Det → Pose → ReID |
| **Parallel-Sync** | 2-3 | Barrier | Det ∥ Seg → Fusion |
| **Parallel-Async** | 2-3 | None | Det@30fps ∥ Scene@3fps |

### Mode Selection

```
                        1 model? ──────▶ Single
                           │
                           ▼
                     Dependent? ───yes──▶ Serial
                           │
                          no
                           │
                           ▼
                     Same rate? ───yes──▶ Parallel-Sync
                           │
                          no
                           │
                           ▼
                                        Parallel-Async
```

---

## 8.3 Component Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│  Component Architecture                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Lua Script Layer                                        │   │
│  │  - Pipeline configuration                                │   │
│  │  - Model pre/post functions                              │   │
│  │  - Fusion function (optional)                            │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Pipeline (C++)                                          │   │
│  │  - Mode executor (single/serial/parallel)                │   │
│  │  - Context management                                    │   │
│  │  - Lua function invocation                               │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SessionManager (C++)                                    │   │
│  │  - Session lifecycle (max 3)                             │   │
│  │  - Resource tracking                                     │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Session (existing)                                      │   │
│  │  - ONNX Runtime wrapper                                  │   │
│  │  - Tensor I/O                                            │   │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8.4 Constraints

### Hard Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max models | 3 | Covers 99% of scenarios |
| Max sessions | 3 | One per model slot |
| Execution modes | 4 | Sufficient for all patterns |

### Design Constraints

| Constraint | Description |
|------------|-------------|
| No execution graph | Too complex for embedded |
| No dynamic scheduling | Predictable resource usage |
| No conditional branching in C++ | Logic stays in Lua |

---

## 8.5 Performance Targets

### Timing by Mode

| Mode | Example | Target FPS |
|------|---------|------------|
| Single | YOLO detection | 30-60 FPS |
| Serial (2 models) | Det → Pose | 15-25 FPS |
| Serial (3 models) | Det → Pose → ReID | 10-15 FPS |
| Parallel-Sync | Det ∥ Seg | 20-30 FPS |

### Memory Targets

| Configuration | Memory |
|---------------|--------|
| 1 model (small) | ~50 MB |
| 2 models (medium) | ~100 MB |
| 3 models (mixed) | ~150 MB |
| 3 models (large) | ~200 MB |

---

## 8.6 Implementation Roadmap

### Phase 1: Foundation

| Component | Description |
|-----------|-------------|
| Pipeline class | Basic structure, Single mode |
| SessionManager | Load/unload up to 3 sessions |
| Lua integration | preprocess/postprocess callbacks |
| FrameContext | Data passing between models |

### Phase 2: Serial Mode

| Component | Description |
|-----------|-------------|
| Serial executor | Sequential model execution |
| Context chaining | Pass results between models |
| Skip logic | Handle nil returns from preprocess |
| Batch inference | Stack multiple ROIs |

### Phase 3: Parallel Modes

| Component | Description |
|-----------|-------------|
| Thread pool | For parallel execution |
| Parallel-Sync | Barrier synchronization |
| Fusion callback | Combine parallel results |
| Parallel-Async | Worker threads, queues |

### Phase 4: Optimization

| Component | Description |
|-----------|-------------|
| Buffer pooling | Reuse tensor buffers |
| NPU scheduling | Multi-core assignment |
| Profiling | Timing and memory metrics |

---

## 8.7 Comparison with Previous Design

### What Changed

| Previous Design | New Design |
|-----------------|------------|
| 5-level architecture | 2-level (Lua + C++) |
| Execution graph (ModelGraph) | Direct mode selection |
| Complex scheduling (ModelScheduler) | Simple mode-based execution |
| Many node types | Just models + fusion |
| Unlimited models | Max 3 models |

### Why Simplified

1. **Embedded constraints**: Complex abstractions have overhead
2. **Coverage**: 3 models cover 99% of use cases
3. **Maintainability**: Simpler code, fewer bugs
4. **Lua-centric**: Keep flexibility in scripts, not C++

---

## 8.8 Example Pipelines

### Covered Scenarios

| Scenario | Mode | Models |
|----------|------|--------|
| Pedestrian Analysis | Serial | Det → Pose → ReID |
| Instance Segmentation | Parallel-Sync | Det ∥ Seg |
| Face Recognition | Serial | Det → Align → Embed |
| Video Analytics | Parallel-Async | Det + Scene + Anomaly |
| Vehicle + Plate + OCR | Serial | VehicleDet → PlateDet → OCR |
| Object Tracking | Single | Detection (+ Lua tracker) |

### Beyond 3 Models

For scenarios needing 4+ models:
1. **Decompose**: Split into two pipelines
2. **Combine**: Merge models if possible
3. **Time-multiplex**: Swap models dynamically

---

## 8.9 Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Alignment as separate model? | Yes / Part of preprocess | TBD |
| NPU core assignment | Fixed / Dynamic | TBD |
| Parallel-Async result sync | Blocking / Callback | TBD |
| Error recovery strategy | Abort / Continue | TBD |
