# Future Architecture Documentation Index

## Overview

Multi-model inference pipeline architecture for embedded Linux systems.

---

## Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Lua for Pre/Post** | Each model has customized preprocessing and postprocessing in Lua |
| **Max 3 Models** | At most 3 models per pipeline, covering 99% of scenarios |
| **Serial/Parallel** | Lua configures execution mode |
| **C++ for Inference** | C++ only handles model loading and inference |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Lua Script                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model 1              Model 2              Model 3              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ preprocess  │     │ preprocess  │     │ preprocess  │        │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤        │
│  │  C++ infer  │     │  C++ infer  │     │  C++ infer  │        │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤        │
│  │ postprocess │     │ postprocess │     │ postprocess │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                 │
│  Config: mode = "serial" | "parallel"                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Document Structure

### Core Design

| Document | Content |
|----------|---------|
| [01. Core Abstractions](01-core-abstractions.md) | Image/Tensor/DeviceBuffer design, extension points |
| [02. Multi-Model Scenarios](02-multi-model-scenarios.md) | Typical 2-3 model combinations |
| [03. Execution Modes](03-execution-modes.md) | Serial and Parallel mode design |
| [04. Pipeline Design](04-pipeline-design.md) | C++ Pipeline class, Lua interface |
| [05. Resource Management](05-resource-management.md) | SessionManager, memory |
| [06. Lua Script Structure](06-lua-script-structure.md) | Script template, configuration |

### Complete Examples

| Document | Pattern | Models |
|----------|---------|--------|
| [07.1 Pedestrian Analysis](07.1-example-pedestrian.md) | Serial | Det → Pose → ReID |
| [07.2 Instance Segmentation](07.2-example-segmentation.md) | Parallel | Det ∥ Seg → Fusion |
| [07.3 Face Recognition](07.3-example-face.md) | Serial | Det → Align → Embed |

### Summary & Extensions

| Document | Content |
|----------|---------|
| [08. Architecture Summary](08-architecture-summary.md) | Design decisions, roadmap |

---

## Extension Points (Reserved for Plan B)

The current architecture follows Plan A (flat structure), with the following extension points reserved for future evolution to Plan B (layered architecture):

| Extension Point | Current | Reserved |
|-----------------|---------|----------|
| DeviceType | CPU, NPU | RK_DMA, RK_CMA, ... |
| Tensor DType | float32 | int8, uint8, float16 |
| Memory Layout | NCHW | NHWC, NC1HWC2 |
| Frame-level Pipeline | None | submit/try_get_result |
| Image Device Specialization | None | stride, align, format |

See [01. Core Abstractions](01-core-abstractions.md) for details

---

## Model Limit Rationale

| Models | Coverage | Typical Scenarios |
|--------|----------|-------------------|
| 1 | 60% | Detection, Classification, Segmentation |
| 2 | 35% | Det+Pose, Det+ReID, Det+Seg |
| 3 | 4% | Det+Pose+ReID, Det+Seg+Class |
| 4+ | <1% | Rare, decompose into stages |
