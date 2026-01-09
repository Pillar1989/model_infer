# 1. Core Abstractions

This document supplements the low-level abstraction design for Image, Tensor, and DeviceBuffer, addressing issues raised by reviewers.

---

## 1.1 Design Goals

| Goal | Description |
|------|-------------|
| **Zero-copy** | Image ↔ Tensor conversion with minimal copying |
| **Device Specialization** | Support stride/align for platforms like RK |
| **Quantization Support** | Support int8/uint8, pre/post processing can access quantization parameters |
| **Reserved Extensions** | Extension points reserved for Plan B layered architecture |

---

## 1.2 DeviceBuffer Fine-Grained Design

### Current Issues

```
Current DeviceType:
  CPU, NPU, TPU  ← Too coarse-grained

Actual requirements (RK platform):
  - RK VPU decode → DMA buffer (can go directly to NPU)
  - RK NPU input → CMA buffer
  - Regular malloc → Needs copy to CMA
```

### DeviceType Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  DeviceType (Phase 1: Basic)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum class DeviceType {                                        │
│      CPU,              // Regular malloc                        │
│      CPU_PINNED,       // Pinned memory (DMA accessible)        │
│      NPU,              // NPU generic                           │
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DeviceType (Phase 2: Platform Specialization, Reserved)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum class DeviceType {                                        │
│      CPU,                                                       │
│      CPU_PINNED,                                                │
│                                                                 │
│      // RK Platform Specialization                              │
│      RK_DMA,           // VPU decode output, fd-based           │
│      RK_CMA,           // NPU input, contiguous phys memory     │
│      RK_NPU_CORE0,     // NPU Core 0 dedicated                  │
│      RK_NPU_CORE1,     // NPU Core 1 dedicated                  │
│                                                                 │
│      // Other Platforms (Reserved)                              │
│      // HAILO_DDR,                                              │
│      // ...                                                     │
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### DeviceBuffer Interface Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  DeviceBuffer Interface                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  class DeviceBuffer {                                           │
│  public:                                                        │
│      // Basic Interface (Existing)                              │
│      virtual void* data() = 0;                                  │
│      virtual size_t size() const = 0;                           │
│      virtual DeviceType device() const = 0;                     │
│                                                                 │
│      // Extended Interface (New)                                │
│      virtual int fd() const { return -1; }  // DMA fd           │
│      virtual void sync() {}                 // Device sync      │
│                                             // (eg rknn_mem_sync│
│                                                                 │
│      // Platform-Specific Attributes (Reserved)                 │
│      struct PlatformAttrs {                                     │
│          int dma_fd = -1;                                       │
│          size_t phy_addr = 0;                                   │
│          // Other platform attributes...                        │
│      };                                                         │
│      virtual PlatformAttrs platform_attrs() const { return {}; }│
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Image Device Specialization

### Current Issues

```
Frames decoded on RK platform:
  - May be in NV12/NV21 format
  - Has stride (may != width * channels)
  - Has alignment requirements
  - Data in DMA buffer

Using cv::Mat directly causes:
  - Format conversion overhead
  - Memory copy overhead
```

### Image Structure Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  Image Structure (Phase 1)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  class Image {                                                  │
│  public:                                                        │
│      // Basic attributes                                        │
│      int width() const;                                         │
│      int height() const;                                        │
│      int channels() const;                                      │
│                                                                 │
│      // Format (New)                                            │
│      enum class Format {                                        │
│          BGR,       // OpenCV default                           │
│          RGB,       // Most model inputs                        │
│          NV12,      // YUV 4:2:0 (common in decode)             │
│          NV21,      // YUV 4:2:0                                │
│          GRAY,      // Grayscale                                │
│      };                                                         │
│      Format format() const;                                     │
│                                                                 │
│      // Stride (New)                                            │
│      int stride() const;        // Bytes per row, may>width*ch  │
│      bool is_contiguous() const; // stride == width * channels  │
│                                                                 │
│      // ROI Operations (Explicit Interface)                     │
│      Image crop(int x, int y, int w, int h) const;  // Copy     │
│      Image view(int x, int y, int w, int h) const;  // Zero-copy│
│                                                                 │
│      // Device info (New)                                       │
│      DeviceType device() const;                                 │
│      std::shared_ptr<DeviceBuffer> buffer() const;              │
│                                                                 │
│      // Conversion                                              │
│      Image to_device(DeviceType target) const; // Cross-device  │
│      Image to_format(Format target) const;     // Format change │
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ROI Zero-Copy vs Copy

```
┌─────────────────────────────────────────────────────────────────┐
│  ROI Operation Semantics                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  view(x, y, w, h):                                              │
│    - Returns view, shares underlying buffer                     │
│    - Modifying view affects original image                      │
│    - Zero-copy, fast                                            │
│    - Limitation: must be stride-aligned                         │
│                                                                 │
│  crop(x, y, w, h):                                              │
│    - Returns independent copy                                   │
│    - Modifications don't affect original image                  │
│    - Has copy overhead                                          │
│    - No limitations                                             │
│                                                                 │
│  Selection strategy:                                            │
│    - Read-only access → view()                                  │
│    - Need independent modification → crop()                     │
│    - Send to model (needs resize) → crop() + resize()           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.4 Tensor Multi-Type and Layout

### Current Issues

```
1. Only supports float32, quantized models (int8/uint8) cannot use zero-copy
2. Layout implicit in shape/strides, not explicit enough
3. Quantization parameters (scale, zero_point) are not stored
```

### Tensor Extension

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor Data Type (New)                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum class DType {                                             │
│      Float32,     // Current default                            │
│      Float16,     // Half precision                             │
│      Int8,        // Quantization                               │
│      UInt8,       // Quantization (common)                      │
│      Int32,       // Index, label                               │
│  };                                                             │
│                                                                 │
│  Tensor {                                                       │
│      DType dtype() const;                                       │
│      size_t element_size() const; // sizeof(dtype)              │
│  };                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Tensor Layout (New)                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enum class Layout {                                            │
│      NCHW,        // ONNX default                               │
│      NHWC,        // TFLite, some platforms prefer              │
│      NC1HWC2,     // RKNN some operators                        │
│  };                                                             │
│                                                                 │
│  Tensor {                                                       │
│      Layout layout() const;                                     │
│      Tensor to_layout(Layout target) const; // Convert          │
│  };                                                             │
│                                                                 │
│  Note: Layout can also be expressed implicitly via strides,     │
│        explicit enum easier for debugging                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Tensor Quantization Parameters (New)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  struct QuantParams {                                           │
│      float scale = 1.0f;                                        │
│      int32_t zero_point = 0;                                    │
│      // Per-channel quantization (reserved)                     │
│      // std::vector<float> scales;                              │
│      // std::vector<int32_t> zero_points;                       │
│  };                                                             │
│                                                                 │
│  Tensor {                                                       │
│      bool is_quantized() const;                                 │
│      QuantParams quant_params() const;                          │
│                                                                 │
│      // Quantize/dequantize                                     │
│      Tensor quantize(DType target, QuantParams params) const;   │
│      Tensor dequantize() const; // → Float32                    │
│  };                                                             │
│                                                                 │
│  Zero-copy scenarios:                                           │
│    - Pre-processing outputs int8 tensor (fused quantization)    │
│    - Post-processing reads int8 tensor (conditional dequant)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.5 Image ↔ Tensor Boundary

### Conversion Rules

```
┌─────────────────────────────────────────────────────────────────┐
│  Image → Tensor Conversion                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image.to_tensor(options) → Tensor                              │
│                                                                 │
│  options:                                                       │
│    - normalize: bool          // Whether to /255                │
│    - mean, std: float[3]      // Normalization parameters       │
│    - layout: Layout           // NCHW or NHWC                   │
│    - dtype: DType             // Float32 or Int8 (fused quant)  │
│    - quant_params: QuantParams // If dtype=Int8                 │
│                                                                 │
│  Zero-copy conditions:                                          │
│    - image.format == RGB/BGR                                    │
│    - image.is_contiguous()                                      │
│    - image.device == target device                              │
│    - No normalize/mean/std (or hardware support)                │
│    - Layout matches (NHWC image → NHWC tensor)                  │
│                                                                 │
│  Otherwise copy + conversion needed                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Tensor → Image Conversion (less common)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tensor.to_image(options) → Image                               │
│                                                                 │
│  Purpose:                                                       │
│    - Extract masked region from tensor and send to next model   │
│    - Visualize intermediate results                             │
│                                                                 │
│  options:                                                       │
│    - denormalize: bool        // Denormalize                    │
│    - format: Image::Format    // Output format                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Masked Image Scenario

```
┌─────────────────────────────────────────────────────────────────┐
│  Extract Masked Image from Tensor                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario: Segmentation outputs mask, extract masked region     │
│                                                                 │
│  Option 1: Tensor operations (recommended)                      │
│    mask_tensor = seg_output.argmax(0)                           │
│    masked_input = input_tensor * mask_tensor.unsqueeze(0)       │
│    → All Tensor, no Image conversion                            │
│                                                                 │
│  Option 2: Convert to Image (when visualization needed)         │
│    mask = seg_output.argmax(0).to_image()                       │
│    masked_image = original_image.apply_mask(mask)               │
│    → Has conversion overhead, but intuitive                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.6 Structured Output Support (Reserved)

### Issues

```
Platforms like Hailo:
  - NMS embedded in model
  - Output is not Tensor, but structured data (boxes, scores, classes)

Current Tensor cannot express this
```

### Reserved Solutions

```
┌─────────────────────────────────────────────────────────────────┐
│  Structured Output (Reserved, not implemented yet)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Option A: Carry in Tensor, agreed format                       │
│    output[0] = boxes tensor [N, 4]                              │
│    output[1] = scores tensor [N]                                │
│    output[2] = classes tensor [N]                               │
│    output[3] = num_detections tensor [1]                        │
│                                                                 │
│  Option B: New type StructuredOutput (clearer)                  │
│    class StructuredOutput {                                     │
│        std::variant<Tensor, DetectionResult, ...> data;         │
│    };                                                           │
│                                                                 │
│  Current: Option A, express using multiple Tensor outputs       │
│  Future: Can extend to Option B if needed                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.7 Frame-Level Pipeline Reservation

### Current Status

```
Current 4 modes are model-level parallelism:
  - Serial: Model1 → Model2 → Model3
  - Parallel-Sync: Model1 ∥ Model2 → Fusion
  - Parallel-Async: Model1 @ rate1 ∥ Model2 @ rate2
```

### Frame-Level Pipeline (Reserved)

```
┌─────────────────────────────────────────────────────────────────┐
│  Frame-Level Pipeline (Reserved for Phase 2)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Goal:                                                          │
│    Frame N:   [preprocess] → [inference] → [postprocess]        │
│    Frame N+1:                [preprocess] → [inference] → ...   │
│                              ↑ Overlapping execution            │
│                                                                 │
│  Implementation reservation:                                    │
│    - FrameContext independent lifecycle management              │
│    - Pipeline optional frame_pipeline mode                      │
│    - Double/triple buffering: input/inference/output buffers    │
│                                                                 │
│  Not implemented currently, but interfaces reserved:            │
│    Pipeline {                                                   │
│        // Existing synchronous interface                        │
│        Results run(Image& frame);                               │
│                                                                 │
│        // Reserved async interface                              │
│        void submit(Image& frame);    // Non-blocking submit     │
│        Results try_get_result();     // Non-blocking get        │
│    }                                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.8 Concurrent Synchronization

### DeviceBuffer Sync

```
┌─────────────────────────────────────────────────────────────────┐
│  Device Sync (Phase 1)                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Issue: RKNN requires rknn_mem_sync to ensure CPU/NPU data      │
│         consistency                                             │
│                                                                 │
│  DeviceBuffer Interface:                                        │
│    virtual void sync(SyncDirection dir) {                       │
│        // Default empty impl, platform-specific classes override│
│    }                                                            │
│                                                                 │
│    enum class SyncDirection {                                   │
│        HostToDevice,   // CPU write complete, sync to NPU       │
│        DeviceToHost,   // NPU write complete, sync to CPU       │
│        Bidirectional,  // Bidirectional sync                    │
│    };                                                           │
│                                                                 │
│  Usage timing:                                                  │
│    - After CPU preprocess, before NPU: HostToDevice             │
│    - After NPU inference, before CPU postprocess: DeviceToHost  │
│                                                                 │
│  Pipeline calls automatically, transparent to user              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.9 Extension Path Summary

| Feature | Phase 1 (Current) | Phase 2 (Reserved) |
|---------|------------------|-------------------|
| DeviceType | CPU, CPU_PINNED, NPU | RK_DMA, RK_CMA, ... |
| Image Format | BGR, RGB | NV12, NV21 |
| Image ROI | crop(), view() | Hardware-accelerated crop |
| Tensor DType | Float32 | Int8, UInt8, Float16 |
| Tensor Layout | NCHW (default) | NHWC, NC1HWC2 |
| Quant Params | None | scale, zero_point |
| Structured Output | Multi-tensor convention | StructuredOutput class |
| Frame Pipeline | None | submit/try_get_result |
| Device Sync | Manual | Pipeline auto-calls |

---

## 1.10 Relationship with Existing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Architecture Layers (Plan A: Flat, with reserved extensions)   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Lua Scripts (pre/post processing)                       │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Pipeline (4 modes: Single/Serial/Parallel-Sync/Async)   │   │
│  │  [Reserved: submit/try_get_result for frame pipeline]    │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Session / SessionManager                                │   │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Core Abstractions (this document)                       │   │
│  │  - Image: format, stride, ROI, device                    │   │
│  │  - Tensor: dtype, layout, quant_params                   │   │
│  │  - DeviceBuffer: fine-grained DeviceType, sync()         │   │
│  │  [Reserved: can be separated as Level 1 layer]           │   │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Future extension to Plan B:
  - Core Abstractions → Level 1
  - Add Frame Pipeline → Level 2
  - Pipeline (4 modes) → Level 3
```
