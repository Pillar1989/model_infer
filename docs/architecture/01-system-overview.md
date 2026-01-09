# 1. System Overview

## 1.1 Architecture Layers

The system implements a **3-layer hybrid C++/Lua architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Lua Application Scripts                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ YOLO11      │  │ YOLOv5       │  │ Custom       │      │
│  │ Detector    │  │ Pose         │  │ Pipelines    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          ↓ ↑ Lua C API (via LuaIntf)
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: C++ Lua Binding Modules                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ lua_cv      │  │ lua_nn       │  │ lua_utils    │      │
│  │ (Image,     │  │ (Session,    │  │ (NMS, box    │      │
│  │  OpenCV)    │  │  Tensor)     │  │  utils)      │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          ↓ ↑ Direct C++ calls
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: C++ Core Implementation                           │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ inference::      │  │ tensor::         │               │
│  │ OnnxSession      │  │ Tensor           │               │
│  │                  │  │ DeviceBuffer     │               │
│  │ (ONNX Runtime)   │  │ CpuMemory        │               │
│  └──────────────────┘  └──────────────────┘               │
│  ┌──────────────────────────────────────────┐             │
│  │ OpenCV 4.x (cv::Mat, VideoCapture)       │             │
│  └──────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Layer 1 (C++ Core)**:
- Performance-critical operations
- ONNX Runtime inference
- Tensor mathematics and operations
- OpenCV image processing

**Layer 2 (Binding Layer)**:
- Thin wrapper exposing C++ to Lua via LuaIntf
- Data conversion between C++ and Lua
- Minimal overhead

**Layer 3 (Lua Scripts)**:
- Business logic and control flow
- Preprocessing and postprocessing pipelines
- Model-specific configurations
- Runtime flexibility without recompilation

---

## 1.2 Directory Structure

```
src/
  ├── main.cpp                    # Application entry points
  ├── bindings/                   # Lua module registration
  ├── modules/
  │   ├── lua_cv/                 # Computer vision bindings
  │   ├── lua_nn/                 # Neural network bindings
  │   ├── lua_utils/              # Utility functions
  │   └── tensor/                 # Tensor implementation
  └── inference/                  # ONNX Runtime wrapper

scripts/                          # Lua model scripts
  ├── yolo11_*.lua
  ├── yolov5_*.lua
  └── lib/                        # Shared libraries

tests/                            # Test suites
models/                           # ONNX models
images/                           # Test images
docs/                             # Documentation
```

---

## 1.3 Core Abstractions

### Tensor
N-dimensional array with:
- Zero-copy view semantics (slice, transpose share storage)
- Device abstraction (CPU, NPU, TPU)
- Stride-based indexing for non-contiguous views

### Image
OpenCV wrapper providing:
- Image I/O and basic operations
- Conversion to Tensor (HWC → CHW)
- PreprocessRegistry for C++ preprocessing functions

### Session
ONNX Runtime wrapper providing:
- Model loading and metadata extraction
- Type-aware inference (float/float16 handling)
- Multi-output support

---

## 1.4 Design Philosophy

### Hybrid Language Design

| Aspect | C++ | Lua |
|--------|-----|-----|
| **Performance** | Critical path | Business logic |
| **Memory** | Explicit control | Automatic GC |
| **Compilation** | Requires rebuild | Runtime modification |
| **Use cases** | Inference, tensor ops, OpenCV | Preprocessing, postprocessing, control |

### Zero-Copy View Semantics

Operations like `slice()`, `transpose()`, `squeeze()` create views that share underlying storage:

```
Original Tensor:  [1, 84, 8400]  ← Storage allocated once
     │
     ├─→ slice():     [1, 4, 8400]   ← View (no copy)
     ├─→ slice():     [1, 80, 8400]  ← View (no copy)
     └─→ transpose(): [1, 8400, 84]  ← View (no copy)
```

### Device Abstraction

```
DeviceBuffer (interface)
    ├─ CpuMemory      (implemented)
    ├─ NpuMemory      (future)
    └─ TpuMemory      (future)
```

Enables:
- Adding new devices without changing Tensor API
- Device-specific optimizations
- Heterogeneous computing

---

## 1.5 Data Flow Overview

```
Image File
    │
    ├─→ OpenCV cv::Mat
    │     │
    │     └─→ lua_cv::Image
    │           │
    │           ├─→ Preprocessing (C++ or Lua)
    │           │     │
    │           │     └─→ Tensor (1, 3, H, W)
    │           │           │
    │           ├─→ Inference (ONNX Runtime)
    │           │     │
    │           │     └─→ Tensor (outputs)
    │           │           │
    │           └─→ Postprocessing (Lua + Tensor API)
    │                 │
    │                 └─→ Detections (Lua table)
    │
    └─→ Visualization/Save

Timing (YOLO11n, 640×640):
  Preprocess:   ~12ms  (6%)
  Inference:   ~100ms  (50%)
  Postprocess:  ~4.5ms (2.2%)
  Total:       ~200ms
```

---

## 1.6 Execution Modes

### Image Inference
```bash
./lua_runner <script.lua> <model.onnx> <image> [show] [save=path]
```

### Video Inference
```bash
./lua_runner <script.lua> <model.onnx> <video> [show] [save=path] [frames=N]
```

### Test Mode
```bash
./lua_runner <test_script.lua>
```

---

## Summary

The architecture emphasizes:

✅ **Hybrid design**: C++ performance + Lua flexibility
✅ **Clean layering**: Core → Bindings → Scripts
✅ **Zero-copy efficiency**: Tensor views share storage
✅ **Device abstraction**: Ready for NPU/TPU
✅ **Modular organization**: Clear responsibilities
✅ **Dual preprocessing**: C++ (fast) and Lua (flexible)
