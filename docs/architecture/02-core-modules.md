# 2. Core Modules

This document describes the core modules and their responsibilities.

---

## 2.1 Module: `lua_cv` (Computer Vision)

**Location**: `src/modules/lua_cv.h/cpp`

**Responsibility**: Expose OpenCV operations to Lua and provide C++ preprocessing functions

### 2.1.1 Image Class

```
┌─────────────────────────────────────────────────────────┐
│  lua_cv::Image                                          │
├─────────────────────────────────────────────────────────┤
│  Internal State:                                        │
│    cv::Mat mat_                                         │
│                                                         │
│  Operations:                                            │
│    - load(path) / save(path)                           │
│    - width() / height() / channels()                    │
│    - resize(w, h)                                       │
│    - pad(top, bottom, left, right, value)              │
│    - to_tensor(scale, mean, std)                       │
└─────────────────────────────────────────────────────────┘
```

**Design**: Thin wrapper around `cv::Mat`, delegates all operations to OpenCV.

### 2.1.2 Image → Tensor Conversion

```
Input: cv::Mat (H, W, C) in BGR format
   ↓
Color conversion (if needed): BGR → RGB
   ↓
Allocate: Tensor(1, C, H, W)
   ↓
Transform: HWC → CHW with normalization
   for c in [0, C):
     for h in [0, H):
       for w in [0, W):
         tensor[0,c,h,w] = (pixel[h,w,c]/scale - mean[c]) / std[c]
   ↓
Output: Tensor(1, C, H, W) on CPU
```

### 2.1.3 PreprocessRegistry

**Purpose**: Registry for C++ preprocessing functions

```
┌──────────────────────────────────────────────────────┐
│  PreprocessRegistry (Singleton)                      │
├──────────────────────────────────────────────────────┤
│  map<string, PreprocessFunc>                         │
│       │                                              │
│       ├─→ "letterbox" → preprocess_letterbox()      │
│       └─→ "resize_center_crop" → ...                │
│                                                      │
│  API:                                                │
│    run(type, img, L, config) → PreprocessResult     │
│    has(type) → bool                                  │
└──────────────────────────────────────────────────────┘

PreprocessResult:
  {
    tensor: Tensor        (preprocessed input)
    meta: LuaRef          (metadata for postprocessing)
  }
```

**Example**: letterbox preprocessing
1. Calculate scale ratio (keep aspect ratio)
2. Resize image
3. Calculate padding (aligned to stride)
4. Add padding
5. Convert to tensor (HWC → CHW, normalize)
6. Return tensor + metadata

---

## 2.2 Module: `tensor::Tensor` (N-Dimensional Arrays)

**Location**: `src/modules/tensor/`

**Responsibility**: N-dimensional array with device abstraction and zero-copy semantics

### 2.2.1 Tensor Structure

```
┌────────────────────────────────────────────────────────────────┐
│  tensor::Tensor                                                 │
├────────────────────────────────────────────────────────────────┤
│  Metadata (per-tensor):                                        │
│    std::vector<int64_t> shape_                                 │
│    std::vector<int64_t> strides_                               │
│    int64_t offset_                                             │
│    bool contiguous_                                            │
│    DeviceType device_type_                                     │
│                                                                 │
│  Storage (shared across views):                                │
│    std::shared_ptr<DeviceBuffer> storage_                      │
│         │                                                       │
│         └─→ DeviceBuffer (abstract interface)                  │
│               │                                                 │
│               └─→ CpuMemory (malloc/free implementation)       │
└────────────────────────────────────────────────────────────────┘
```

### 2.2.2 Device Abstraction

```
DeviceType enum:
  ├─ CPU
  ├─ NPU
  └─ TPU

DeviceBuffer (interface):
  ├─ data() → void*
  ├─ size() → size_t
  ├─ device() → DeviceType
  ├─ copy_to_async(...)
  └─ sync()

CpuMemory (implementation):
  ├─ allocate() → malloc()
  ├─ free() → ::free()
  └─ from_external() for zero-copy wrapping
```

### 2.2.3 Zero-Copy Views

```
Original: Tensor A
  shape:     [4, 8400]
  strides:   [8400, 1]
  offset:    0
  storage:   shared_ptr [refcount=1] ──→ malloc(...)
                 │
                 │ (same storage shared)
                 │
View B = A.slice(0, 0, 1)
  shape:     [8400]
  strides:   [1]
  offset:    0
  storage:   shared_ptr [refcount=2] ──→ (same memory)
                 │
                 │
View C = A.transpose(0, 1)
  shape:     [8400, 4]
  strides:   [1, 8400]  ← Non-contiguous!
  offset:    0
  storage:   shared_ptr [refcount=3] ──→ (same memory)
```

**View operations** (zero-copy):
- `slice(dim, start, end)`
- `transpose(dim0, dim1)`
- `squeeze(dim)` / `unsqueeze(dim)`
- `view(shape)` (requires contiguous)

**Copy operations**:
- `contiguous()` - Make contiguous copy if needed
- `to(device)` - Copy to different device
- `clone()` - Deep copy

### 2.2.4 Operation Categories

The tensor implementation is split across multiple files by functionality:

**Shape operations**:
- slice, transpose, squeeze, unsqueeze
- view, reshape, permute

**Mathematical operations**:
- Element-wise: add, sub, mul, div (with in-place variants)
- Scalar operations: add_scalar, mul_scalar

**Activation functions**:
- sigmoid, sigmoid_ (in-place)

**Reduction operations**:
- max, argmax, max_with_argmax (fused)
- sum

**Selection operations**:
- where_indices (conditional filtering)
- index_select (gather by indices)
- extract_columns (optimized column extraction)

**Advanced operations**:
- gather, concat, split

### 2.2.5 Stride-Based Indexing

```
Element access:
  data()[offset_ + i₀*strides_[0] + i₁*strides_[1] + ...]

Example (contiguous):
  shape:    [3, 4]
  strides:  [4, 1]
  Element [1, 2] = data()[0 + 1*4 + 2*1] = data()[6]

Example (transposed, non-contiguous):
  shape:    [4, 3]
  strides:  [1, 4]
  Element [2, 1] = data()[0 + 2*1 + 1*4] = data()[6]
  (same memory location!)
```

---

## 2.3 Module: `inference::OnnxSession` (Inference Engine)

**Location**: `src/inference/inference.h/cpp`

**Responsibility**: ONNX Runtime wrapper

### 2.3.1 OnnxSession Structure

```
┌──────────────────────────────────────────────────────────┐
│  inference::OnnxSession                                   │
├──────────────────────────────────────────────────────────┤
│  ONNX Runtime objects:                                   │
│    Ort::Env env_                                         │
│    Ort::Session session_                                 │
│    Ort::AllocatorWithDefaultOptions allocator_           │
│                                                           │
│  Model metadata (extracted at construction):             │
│    std::vector<std::string> input_names_                 │
│    std::vector<std::string> output_names_                │
│    std::vector<std::vector<int64_t>> input_shapes_       │
│    std::vector<std::vector<int64_t>> output_shapes_      │
│    std::vector<ONNXTensorElementDataType> types_         │
└──────────────────────────────────────────────────────────┘
```

### 2.3.2 Inference Flow

```
Input: Tensor (float data)
   │
   ├─→ [Model expects FLOAT]
   │     └─→ Zero-copy wrap in Ort::Value
   │           │
   │           └─→ session_.Run()
   │                 │
   │                 └─→ Extract output
   │
   └─→ [Model expects FLOAT16]
         ├─→ Allocate temp float16 buffer
         ├─→ Convert float → float16
         ├─→ Wrap in Ort::Value
         ├─→ session_.Run()
         ├─→ Extract output (float16)
         └─→ Convert float16 → float

Output: Wrapped in Tensor, returned as Lua table
        {output0 = Tensor, output1 = Tensor, ...}
```

### 2.3.3 Initialization

```
Construction:
  1. Create Ort::Env
  2. Configure SessionOptions:
     - Thread count (intra-op, inter-op)
     - Optimization level
  3. Load model from file
  4. Extract metadata:
     - Input/output names
     - Shapes
     - Data types
  5. Store for runtime validation
```

---

## 2.4 Module: `lua_nn` (Neural Network Bindings)

**Location**: `src/modules/lua_nn.h/cpp`

**Responsibility**: Expose inference and tensor to Lua

### 2.4.1 Session Binding

```lua
-- Lua API:
session = nn.Session.new("model.onnx")

-- Query metadata:
names = session:input_names()     → {"input"}
shapes = session:input_shapes()   → {{1, 3, 640, 640}}

-- Run inference:
outputs = session:run(tensor)
-- Returns: {output0 = Tensor, ...}
```

**Implementation**:
- Wraps `inference::OnnxSession`
- Accepts `tensor::Tensor` as input
- Returns Lua table mapping output names to Tensors

### 2.4.2 Tensor Binding

```cpp
// Tensor is typedef:
namespace lua_nn {
    using Tensor = tensor::Tensor;
}
```

All tensor operations exposed via LuaIntf:
```cpp
LuaBinding(L).beginClass<Tensor>("Tensor")
    .addFunction("slice", &Tensor::slice)
    .addFunction("transpose", &Tensor::transpose)
    // ... many more
.endClass();
```

---

## 2.5 Module: `lua_utils` (Utility Functions)

**Location**: `src/modules/lua_utils.h/cpp`

**Responsibility**: Postprocessing utilities

**Exposed functions**:

```lua
-- Non-Maximum Suppression
keep = utils.nms(proposals, iou_threshold)

-- Box format conversion
xyxy = utils.xywh2xyxy(xywh_boxes)
xywh = utils.xyxy2xywh(xyxy_boxes)

-- Coordinate scaling
scaled = utils.scale_boxes(boxes, orig_shape, target_shape)
```

**Implementation**: Pure C++ functions accepting/returning Lua tables.

---

## Module Interaction Diagram

```
┌─────────────┐
│ Lua Script  │
└─────────────┘
      │ ↓ ↑
      │ Uses
      ↓
┌──────────────────────────────────────────────┐
│ lua_cv        lua_nn        lua_utils        │
│  │              │               │            │
│  Image          Session         nms()        │
│  │              │               │            │
└──┼──────────────┼───────────────┼────────────┘
   │              │               │
   │ Calls        │ Calls         │ Pure Lua/C++
   ↓              ↓
┌──────────────────────────────────────────────┐
│ OpenCV        inference::       (no deps)    │
│  cv::Mat      OnnxSession                    │
│               ├─ ONNX Runtime                │
│               └─ tensor::Tensor              │
└──────────────────────────────────────────────┘
```

---

## Summary

The module design emphasizes:

✅ **Clear responsibilities**: Each module has a focused purpose
✅ **Thin bindings**: Minimal wrapper overhead
✅ **Device abstraction**: DeviceBuffer enables future extensions
✅ **Zero-copy semantics**: Tensor views share storage
✅ **Modular implementation**: Split by functionality for maintainability
