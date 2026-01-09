# 4. Memory Management

This document describes memory management strategies and lifecycle patterns.

---

## 4.1 Shared Ownership Model

The system uses `std::shared_ptr<DeviceBuffer>` for tensor storage:

### Reference Counting

```
Tensor A created:
  storage_ = make_shared<CpuMemory>(size)
  refcount = 1
      │
      └─→ malloc(size * sizeof(float))

View B = A.slice(...):
  storage_ = A.storage_  (copy shared_ptr)
  refcount = 2
      │
      └─→ (same memory)

View C = A.transpose(...):
  storage_ = A.storage_  (copy shared_ptr)
  refcount = 3
      │
      └─→ (same memory)

A goes out of scope:
  refcount → 2

B goes out of scope:
  refcount → 1

C goes out of scope:
  refcount → 0
  └─→ ~CpuMemory() destructor
      └─→ free(ptr_)
```

### Implications

**Advantages**:
- ✅ Safe: Multiple tensors can reference same data
- ✅ Automatic: Memory freed when last reference dies
- ✅ Simple: No manual tracking needed

**Considerations**:
- ⚠️ Shared mutation: Modifying data through one tensor affects all views
- ⚠️ Atomic overhead: `shared_ptr` copy = atomic increment/decrement
- ⚠️ No COW: Copy-on-write not implemented

---

## 4.2 CpuMemory Lifecycle

### Allocation

```cpp
CpuMemory::allocate(size):
  1. ptr_ = malloc(size)
  2. if (ptr_ == nullptr)
       throw std::runtime_error("Out of memory")
  3. memset(ptr_, 0, size)  // Zero-initialize
  4. size_ = size
```

**Note**: Zero-initialization can be skipped for performance if data will be immediately overwritten.

### Deallocation

```cpp
CpuMemory::~CpuMemory():
  1. if (deleter_)
       deleter_(ptr_)  // Custom deleter
     else
       free(ptr_)      // Default free
  2. ptr_ = nullptr
```

### External Data Wrapping

```cpp
CpuMemory::from_external(ptr, size, deleter):
  1. Store ptr_, size_
  2. Store custom deleter function
  3. When destroyed, call deleter(ptr_)

// Example:
auto buffer = CpuMemory::from_external(
    onnx_data_ptr,
    size,
    [onnx_tensor = std::move(onnx_tensor)](void*) mutable {
        // ONNX tensor destroyed here
        // Releases onnx_data_ptr
    }
);

auto tensor = Tensor::from_device_buffer(buffer, shape, strides);
// tensor now owns onnx_tensor through lambda capture
```

**Use case**: Zero-copy wrapping of external buffers (e.g., ONNX Runtime output, cv::Mat data).

**Current status**: Implemented but not used in inference path (outputs are copied instead).

---

## 4.3 Tensor Storage Sharing

### View Creation (Zero-Copy)

```
Original tensor (contiguous):
  shape:    [1, 84, 8400]
  strides:  [705600, 8400, 1]
  offset:   0
  storage:  shared_ptr<DeviceBuffer> [refcount=1]
                │
                └─→ malloc(1 * 84 * 8400 * 4 bytes) = 2.82 MB

boxes = output:slice(1, 0, 4)         [Zero-copy view]
  shape:    [1, 4, 8400]
  strides:  [705600, 8400, 1]
  offset:   0
  storage:  (same shared_ptr) [refcount=2]

scores = output:slice(1, 4, 84)       [Zero-copy view]
  shape:    [1, 80, 8400]
  strides:  [705600, 8400, 1]
  offset:   4 * 8400 * 4 bytes = 134400 bytes
  storage:  (same shared_ptr) [refcount=3]

transposed = scores:transpose(1, 2)   [Zero-copy view]
  shape:    [1, 8400, 80]
  strides:  [705600, 1, 8400]  ← Non-contiguous!
  offset:   134400
  storage:  (same shared_ptr) [refcount=4]

Total memory: 2.82 MB (shared across all views)
```

### Contiguous Copy (Explicit Copy)

```
scores = output:slice(1, 4, 84):transpose(1, 2)
  shape:     [1, 8400, 80]
  strides:   [705600, 1, 8400]  ← Non-contiguous
  storage:   (shared with output)

contig = scores:contiguous()          [Triggers copy]
  shape:     [1, 8400, 80]
  strides:   [672000, 80, 1]  ← Contiguous
  storage:   NEW shared_ptr [refcount=1]
                │
                └─→ malloc(1 * 8400 * 80 * 4) = 2.69 MB

Memory after contiguous():
  - Original storage: 2.82 MB (still referenced by output, boxes)
  - New storage: 2.69 MB (contig owns this)
  - Total: 5.51 MB (temporary peak)
```

**Why contiguous matters**:
- Some operations require contiguous memory (e.g., wrap in ONNX tensor)
- Contiguous memory enables faster operations (better cache locality)
- Non-contiguous views use stride-based access (more computations)

---

## 4.4 Memory Patterns in Inference

### Typical Memory Usage Pattern

```
Frame N processing:

1. Load image (~2 MB for 640×640 RGB)
     cv::Mat allocation

2. Preprocessing (~2 MB)
     Tensor(1, 3, 640, 640) allocation

3. Inference (~20 MB peak)
     - ONNX Runtime internal buffers
     - Model weights (loaded once, shared)
     - Output tensor (~3 MB)

4. Postprocessing views (0 MB new)
     - slice(), transpose() = zero-copy views
     - contiguous() = ~3 MB temporary copy

5. Lua detections (~0.1 MB)
     - Small Lua table (typically <100 detections)

6. Cleanup (automatic)
     - Tensors go out of scope → refcount → 0 → free
     - cv::Mat destroyed
     - Lua GC eventually frees Lua objects

Peak: ~27 MB (during inference + postprocessing)
Steady: ~0 MB (after cleanup, before next frame)
```

### Video Processing Memory Pattern

```
Initialization:
  ├─ Model weights: ~4 MB (persistent)
  ├─ ONNX Runtime: ~5 MB (persistent)
  └─ Lua state: ~1 MB (persistent)
  Total: ~10 MB baseline

Per-frame cycle:
  ├─ Load: +2 MB (frame buffer)
  ├─ Preprocess: +2 MB (tensor)
  ├─ Inference: +20 MB (peak)
  ├─ Postprocess: +3 MB (contiguous copy)
  └─ Cleanup: -27 MB (all freed)

Expected memory growth: ~0-1 KB/frame
  (Lua GC, allocation fragmentation)

Leak indicators:
  ⚠️  >10 KB/frame = likely leak
  ❌ >100 KB/frame = definite leak
```

---

## 4.5 Lua Memory Management

### Garbage Collection

Lua uses **automatic garbage collection**:

```
Lua objects:
  - Tables (detections, metadata)
  
  - Closures (preprocess, postprocess functions)
  - Userdata (wrapped C++ objects: Image, Tensor, Session)

GC triggers:
  - Periodically during script execution
  - When memory pressure increases
  - Can be forced: collectgarbage("collect")
```

### C++ Object Lifetime in Lua

```
C++ object lifecycle:

1. Creation in C++:
     tensor = Tensor(shape)

2. Passed to Lua:
     LuaRef::fromValue(L, tensor)
     ↓
     Lua receives userdata wrapping Tensor*

3. Lua holds reference:
     local t = some_function_returning_tensor()
     - Userdata kept alive
     - C++ Tensor destructor NOT called

4. Lua releases reference:
     t = nil  (or goes out of scope)
     collectgarbage("collect")
     ↓
     Userdata finalized
     ↓
     C++ Tensor destructor called
     ↓
     shared_ptr<DeviceBuffer> refcount decremented
```

**Important**: C++ objects wrapped in Lua are kept alive by Lua GC, not C++ scope.

---

## 4.6 Memory Optimization Strategies

### Current Optimizations

1. **Zero-copy views**: slice(), transpose(), squeeze() don't allocate
2. **Lazy contiguous()**: Only copy when needed
3. **Small Lua tables**: Convert tensor → Lua only after filtering
4. **Shared model weights**: ONNX session shared across frames

### Potential Optimizations (Not Implemented)

1. **Buffer pooling**: Reuse tensor storage across frames
2. **Copy-on-write**: Defer actual copy until modification
3. **Arena allocation**: Batch allocate/free for frame
4. **Memory mapping**: Map model weights instead of loading

---

## 4.7 Memory Leak Detection

### Video Mode Monitoring

```
Frame processing loop:

mem_start = get_memory_usage()

for each frame:
    process_frame()
    mem_current = get_memory_usage()

    if mem_current > mem_peak:
        mem_peak = mem_current

mem_increase = mem_current - mem_start
leak_per_frame = mem_increase / num_frames

if leak_per_frame > 10 KB:
    warn("Potential memory leak")
```

### Diagnostic Output

```
=== Memory Summary ===
Initial:  50.2 MB
Final:    52.1 MB
Peak:     54.3 MB
Increase: 1.9 MB
Per frame: 0.02 KB  ← Acceptable (fragmentation, Lua GC)

vs.

Initial:  50.2 MB
Final:    150.8 MB
Peak:     150.8 MB
Increase: 100.6 MB
Per frame: 1.01 MB  ← ❌ Memory leak!
```

---

## Summary

Memory management characteristics:

✅ **Shared ownership**: `shared_ptr<DeviceBuffer>` for safety
✅ **Zero-copy views**: Efficient tensor slicing
✅ **Automatic cleanup**: Destructors free memory
✅ **Leak detection**: Built-in monitoring for video mode

⚠️ **Considerations**:
- Shared views can lead to unintended mutation
- Contiguous copies create temporary peaks
- Lua GC delays object destruction
- No explicit buffer pooling (allocate/free per frame)
