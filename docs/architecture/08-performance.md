# 8. Performance Characteristics

This document analyzes performance characteristics and benchmarks.

---

## 8.1 Operation Cost Analysis

### Zero-Copy Operations (Sub-Microsecond)

Operations that only modify metadata:

```
Operation               Cost        Notes
────────────────────────────────────────────────────────────
slice(dim, start, end)  ~0.001ms    Copy metadata only
transpose(dim0, dim1)   ~0.001ms    Swap stride values
squeeze(dim)            ~0.001ms    Remove dimension from shape
unsqueeze(dim)          ~0.001ms    Add dimension to shape
view(shape)             ~0.001ms    Reshape (requires contiguous)
```

**Implementation**: These operations create new Tensor objects with different `shape_`, `strides_`, `offset_` but same `storage_`.

### Memory-Bound Operations (Millisecond Range)

Operations that copy or reorganize data:

```
Operation               Cost        Notes
────────────────────────────────────────────────────────────
contiguous()            0.02-4ms    Depends on size
                                    (672K elements = ~4ms)

to(device)              Varies      Future: CPU↔NPU transfer
                                    Current: CPU only

clone()                 Similar     Deep copy of entire tensor
                        to size
```

**Example**: `contiguous()` on `(8400, 80)` tensor:
- Elements: 8400 × 80 = 672,000
- Bytes: 672,000 × 4 = 2.69 MB
- Time: ~3.77 ms (RISC-V C906)
- Throughput: ~713 MB/s

### Compute-Bound Operations (Millisecond Range)

Operations that perform calculations:

```
Operation               Cost        Notes
────────────────────────────────────────────────────────────
add/sub/mul/div         0.1-2ms     Element-wise arithmetic
sigmoid()               0.2-3ms     exp() per element
max_with_argmax()       0.5-1ms     Single-pass reduction
where_indices()         0.05-0.1ms  Conditional scan
extract_columns()       0.02ms      Direct column extraction
```

---

## 8.2 YOLO11n Benchmark

### Test Configuration

```
Model:       YOLO11n (nano)
Input:       640×640 RGB image
Platform:    SG2002 (RISC-V C906, ~1GHz)
Target:      Embedded Linux (<256MB RAM)
```

### Time Breakdown

```
Component                          Time      Percentage
─────────────────────────────────────────────────────────
Model loading (one-time)          ~80ms     N/A
Image load + decode               ~3ms      1.5%
Resize + preprocess (OpenCV)      ~12ms     6.0%
HWC→CHW conversion + normalize    (included in preprocess)
ONNX inference                    ~100ms    50.0%
Postprocess (contiguous)          ~4ms      2.0%
Postprocess (other tensor ops)    ~0.5ms    0.25%
Visualization + save              ~5ms      2.5%
─────────────────────────────────────────────────────────
Total (excluding model load)      ~200ms    100%
```

### Postprocessing Detail

```
Operation                           Time
──────────────────────────────────────────────
boxes = output:slice(...):contiguous()
  - Slice [1, 4, 8400]              ~0.001ms (zero-copy)
  - contiguous()                    ~0.5ms   (33.6K elements)

scores = output:slice(...):contiguous()
  - Slice [1, 80, 8400]             ~0.001ms (zero-copy)
  - contiguous()                    ~3.77ms  (672K elements)

max_with_argmax(scores, dim=0)      ~0.57ms  (fused reduction)
where_indices(threshold)            ~0.07ms  (scan 8400 elements)
extract_columns(indices)            ~0.02ms  (typically ~50 detections)
Lua loop (proposals building)       ~0.10ms
Coordinate scaling                  ~0.02ms
NMS (utils.nms)                     ~0.05ms  (C++ implementation)
──────────────────────────────────────────────
Total postprocess                   ~4.5ms
```

### Bottleneck Analysis

```
Bottleneck ranking:
  1. ONNX inference (~100ms)        ← Model complexity limited
  2. Preprocessing (~12ms)          ← OpenCV operations
  3. Postprocessing (~4.5ms)        ← Mostly contiguous()

Optimization potential:
  ✓ ONNX inference: Limited (model-dependent)
  ✓ Preprocessing: Could use C++ path (currently Lua)
  ✓ Postprocessing: Already optimized (Tensor API)
```

---

## 8.3 Comparison: C++ vs Lua Preprocessing

### C++ Preprocessing (PreprocessRegistry)

```
Flow:
  Image → preprocess_letterbox() (C++)
    ├─ cv::resize
    ├─ cv::copyMakeBorder
    └─ to_tensor (HWC→CHW)

Time: ~10-12ms
```

### Lua Preprocessing

```
Flow:
  Image → Model.preprocess() (Lua)
    ├─ img:resize()      [calls C++]
    ├─ img:pad()         [calls C++]
    └─ img:to_tensor()   [calls C++]

Time: ~12-15ms
```

**Difference**: ~2-3ms overhead due to Lua/C++ boundary crossings.

**Recommendation**: Use C++ path (`preprocess_config`) for production.

---

## 8.4 Memory Usage Patterns

### Per-Frame Memory

```
Component                     Memory
────────────────────────────────────────────
Image buffer (640×640 RGB)    ~1.2 MB
Preprocessed tensor           ~1.2 MB (3×640×640×4 bytes)
ONNX internal buffers         ~15-20 MB (peak during inference)
Output tensor (1, 84, 8400)   ~2.82 MB
Postprocess views             ~0 MB (zero-copy)
Contiguous copy               ~2.69 MB (temporary)
Lua detections                ~0.1 MB (typically <100 boxes)
────────────────────────────────────────────
Peak during inference         ~25-30 MB
After cleanup                 ~0 MB (freed)
```

### Video Processing Memory Growth

```
Expected growth: 0-1 KB/frame
  - Lua GC allocation variance
  - Memory allocator fragmentation

Leak indicators:
  ⚠️  1-10 KB/frame    = Minor issue (check Lua refs)
  ❌ >10 KB/frame      = Likely leak
  ❌ >100 KB/frame     = Definite leak
```

---

## 8.5 Performance Optimization Strategies

### Implemented Optimizations

1. **Zero-copy tensor views**
   - slice(), transpose(), squeeze() don't allocate
   - Saves memory and time

2. **Fused operations**
   - `max_with_argmax()` = single pass (not max + argmax separately)
   - Reduces cache misses

3. **Early filtering**
   - `where_indices()` in C++ before converting to Lua
   - Avoids large Lua table allocations

4. **Batch memcpy in contiguous()**
   - Optimized copy for contiguous blocks
   - Better than element-by-element

5. **C++ preprocessing path**
   - Reduces Lua/C++ boundary crossings
   - ~2-3ms faster than Lua path

6. **Inline hot-path functions**
   - `at()`, `data()`, `raw_data()` inlined
   - Reduces function call overhead

### Not Implemented (Future)

1. **Buffer pooling**
   - Reuse tensor storage across frames
   - Avoid malloc/free overhead

2. **SIMD vectorization**
   - Use NEON (ARM) or RVV (RISC-V) for element-wise ops
   - Potential 4x speedup

3. **NPU offloading**
   - Preprocess on NPU (e.g., RK RGA)
   - Inference on NPU
   - Reduces CPU load

4. **Multi-threading**
   - Frame N+1 preprocess || Frame N inference
   - Requires task scheduler and buffer management

---

## 8.6 Profiling and Measurement

### Timing Utilities

```lua
-- Lua timing
local start = os.clock()
-- ... operation ...
local elapsed = os.clock() - start
print(string.format("Time: %.2f ms", elapsed * 1000))
```

```cpp
// C++ timing
auto start = std::chrono::high_resolution_clock::now();
// ... operation ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Time: " << duration.count() << " ms\n";
```

### Memory Profiling

```cpp
// In main.cpp, video mode
MemoryInfo mem_current;
mem_current.update();  // Read from /proc/self/status

std::cout << "RSS: " << mem_current.vm_rss_kb / 1024.0 << " MB\n";
```

### perf Tools (Linux)

```bash
# CPU profiling
perf record -g ./build/lua_runner script.lua model.onnx input.jpg
perf report

# Cache miss analysis
perf stat -e cache-references,cache-misses ./build/lua_runner ...

# Hotspot analysis
perf record -F 99 -g ./build/lua_runner ...
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

---

## 8.7 Performance Comparison Table

### Tensor API vs Legacy Methods

```
Operation                    Legacy        Tensor API   Speedup
────────────────────────────────────────────────────────────────
Filter detections            ~5ms          ~4.5ms       1.1x
  (YOLO postprocess)

Reason: Optimized operations:
  - max_with_argmax (fused)
  - where_indices (vectorized)
  - extract_columns (direct Lua table output)
```

### C++ vs Lua Preprocessing

```
Method                       Time          Notes
────────────────────────────────────────────────────────────────
C++ (PreprocessRegistry)     ~10-12ms      Direct OpenCV calls
Lua (Model.preprocess)       ~12-15ms      Boundary crossing overhead
```

---

## 8.8 Scalability Analysis

### Input Size Scaling

```
Input Size     Preprocess   Inference   Postprocess   Total
─────────────────────────────────────────────────────────────
320×320        ~5ms         ~25ms       ~1ms          ~31ms
640×640        ~12ms        ~100ms      ~4.5ms        ~117ms
1280×1280      ~45ms        ~400ms      ~18ms         ~463ms

Scaling: roughly O(n²) with image dimensions
```

### Batch Size (Not Currently Supported)

```
Current: Single-image inference
Future: Batch inference could amortize ONNX overhead

Batch=1:   100ms/image
Batch=4:   ~300ms/4 images = 75ms/image (estimated)
Batch=8:   ~500ms/8 images = 62ms/image (estimated)
```

---

## Summary

Performance characteristics:

✅ **Efficient postprocessing**: ~4.5ms (2.2% of total time)
✅ **Zero-copy views**: Minimal overhead for slicing
✅ **Optimized operations**: Fused ops, vectorized filtering
✅ **Predictable**: Consistent timing across frames

🎯 **Bottleneck**: ONNX inference (~100ms, 50% of time)
  - Model-dependent, limited optimization opportunity
  - Could use lighter model (YOLO11n → YOLO11t)
  - Or hardware acceleration (NPU)

⚡ **Optimization priorities**:
  1. Use NPU for inference (when available)
  2. Use C++ preprocessing path
  3. Consider buffer pooling for video
  4. Multi-threading for pipeline overlap
