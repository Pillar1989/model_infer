# 9. Configuration and Settings

This document describes configuration options for ONNX Runtime, OpenCV, and runtime behavior.

---

## 9.1 ONNX Runtime Configuration

### Session Options

Configured in `inference::OnnxSession` constructor:

```cpp
Ort::SessionOptions session_options;

// Thread configuration
session_options.SetIntraOpNumThreads(4);
session_options.SetInterOpNumThreads(4);

// Optimization level
session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL
);

// Memory pattern (optional)
session_options.EnableMemPattern();

// CPU memory arena (optional)
session_options.EnableCpuMemArena();
```

### Thread Configuration

**Intra-op threads**: Parallelism within a single operator

```cpp
SetIntraOpNumThreads(N)

Recommendations:
  - Desktop: N = CPU cores / 2
  - Embedded: N = 2-4 (avoid oversubscription)
  - Single-core: N = 1
```

**Inter-op threads**: Parallelism between operators

```cpp
SetInterOpNumThreads(N)

Recommendations:
  - Usually same as intra-op
  - Or 1 (disable inter-op parallelism)
```

**Current setting**: Both set to 4 (suitable for quad-core embedded CPUs)

### Graph Optimization Levels

```cpp
ORT_DISABLE_ALL         // No optimization
ORT_ENABLE_BASIC        // Basic optimizations (constant folding)
ORT_ENABLE_EXTENDED     // Extended optimizations (operator fusion)
ORT_ENABLE_ALL          // All optimizations (current)

Recommendation: Always use ORT_ENABLE_ALL unless debugging
```

**Enabled optimizations** (ORT_ENABLE_ALL):
- Constant folding
- Redundant node elimination
- Operator fusion (e.g., Conv+BN+ReLU → single op)
- Layout optimization
- Memory planning

### Execution Providers

```cpp
// Default: CPU Execution Provider
// (automatically selected)

// Future: Add execution providers
// session_options.AppendExecutionProvider_CUDA(cuda_options);
// session_options.AppendExecutionProvider_TensorRT(trt_options);
// session_options.AppendExecutionProvider_OpenVINO(ov_options);
```

**Available providers** (platform-dependent):
- CPU: Always available
- CUDA: NVIDIA GPUs
- TensorRT: NVIDIA GPUs (optimized)
- OpenVINO: Intel CPUs/GPUs
- NNAPI: Android devices
- CoreML: Apple devices
- RKNPU: Rockchip NPUs

**Current status**: CPU only (NPU support planned)

---

## 9.2 OpenCV Configuration

### Image I/O

**imread flags**:

```cpp
cv::imread(path, cv::IMREAD_COLOR)

Flags:
  IMREAD_COLOR       = Load as BGR (default)
  IMREAD_GRAYSCALE   = Load as single channel
  IMREAD_UNCHANGED   = Load as-is (preserves alpha)
```

**imwrite options**:

```cpp
cv::imwrite(path, img, params)

Examples:
  // JPEG quality (0-100, default 95)
  {cv::IMWRITE_JPEG_QUALITY, 90}

  // PNG compression (0-9, default 3)
  {cv::IMWRITE_PNG_COMPRESSION, 5}
```

**Current usage**: Defaults (IMREAD_COLOR, JPEG quality 95)

### Video Capture

```cpp
cv::VideoCapture cap(path);

// Reduce buffering (lower latency)
cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

// Backend selection (optional)
cv::VideoCapture cap(path, cv::CAP_FFMPEG);  // Use FFmpeg
```

**Properties**:
```cpp
CAP_PROP_FRAME_WIDTH   = Frame width
CAP_PROP_FRAME_HEIGHT  = Frame height
CAP_PROP_FPS           = Frames per second
CAP_PROP_FRAME_COUNT   = Total frames
CAP_PROP_BUFFERSIZE    = Internal buffer size
```

### Video Writer

```cpp
cv::VideoWriter writer;

int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
writer.open(save_path, fourcc, fps, size);

// Common codecs:
// 'M','J','P','G' = MJPEG
// 'm','p','4','v' = MPEG-4
// 'X','V','I','D' = XVID
// 'H','2','6','4' = H.264 (requires ffmpeg with h264 support)
```

**Current usage**: MPEG-4 (`mp4v`)

### Image Processing

**Resize interpolation**:

```cpp
cv::resize(src, dst, size, 0, 0, interpolation)

Interpolation methods:
  INTER_NEAREST  = Fastest, lowest quality
  INTER_LINEAR   = Fast, good quality (default)
  INTER_CUBIC    = Slower, better quality
  INTER_AREA     = Best for downscaling
```

**Current usage**: `INTER_LINEAR` (OpenCV default)

**Border padding**:

```cpp
cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value)

Border types:
  BORDER_CONSTANT = Constant color fill (used for YOLO)
  BORDER_REPLICATE = Edge pixel replication
  BORDER_REFLECT = Mirror reflection
```

**Current usage**: `BORDER_CONSTANT` with value=114 (YOLO standard)

---

## 9.3 Lua Configuration

### Lua State Options

```cpp
lua_State* L = luaL_newstate();
luaL_openlibs(L);  // Load standard libraries

// Standard libraries loaded:
// - base (print, type, pairs, etc.)
// - table (table.insert, table.sort, etc.)
// - string (string.format, string.sub, etc.)
// - math (math.sin, math.sqrt, etc.)
// - io (io.open, io.read, etc.)
// - os (os.clock, os.time, etc.)
```

### Garbage Collection

```lua
-- Manual GC control (rarely needed)
collectgarbage("collect")  -- Force full collection
collectgarbage("step", 100) -- Incremental step

-- GC tuning (advanced)
collectgarbage("setpause", 200)     -- GC pause (% of memory)
collectgarbage("setstepmul", 200)   -- GC step multiplier
```

**Current usage**: Automatic (default Lua GC settings)

**Recommendation**: Leave GC on automatic unless profiling shows issues

---

## 9.4 Model Configuration

### Lua Model Script Structure

Every model script returns a `Model` table:

```lua
local Model = {}

-- Model-specific settings
Model.config = {
    input_size = {640, 640},     -- Model input dimensions
    conf_thres = 0.25,            -- Confidence threshold
    iou_thres = 0.45,             -- NMS IoU threshold
    stride = 32,                  -- Preprocessing alignment
    labels = coco_labels          -- Class labels
}

-- C++ preprocessing configuration (optional, recommended)
Model.preprocess_config = {
    type = "letterbox",           -- Preprocessing type
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- Lua preprocessing (optional, fallback)
Model.preprocess = function(img)
    -- Lua implementation
    return tensor, meta
end

-- Postprocessing (required)
Model.postprocess = function(outputs, meta)
    -- Extract detections from outputs
    return detections
end

return Model
```

### Preprocessing Configuration

**Letterbox** (YOLO standard):

```lua
preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,          -- Padding alignment
    fill_value = 114      -- Gray padding
}
```

**Resize + Center Crop**:

```lua
preprocess_config = {
    type = "resize_center_crop",
    input_size = {224, 224},
    scale = 1.0 / 255.0,
    mean = {0.485, 0.456, 0.406},
    std = {0.229, 0.224, 0.225}
}
```

---

## 9.5 Runtime Configuration

### Command-Line Options

```bash
# Image inference
./lua_runner <script> <model> <image> [options]

Options:
  show          - Display result window
  save=PATH     - Save output to file

# Video inference
./lua_runner <script> <model> <video> [options]

Options:
  show          - Display video window
  save=PATH     - Save output video
  frames=N      - Process only N frames
  skip=N        - Process every Nth frame (default: 1)
```

**Examples**:

```bash
# Display result
./lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx images/test.jpg show

# Save result
./lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx images/test.jpg save=output.jpg

# Video with frame limit
./lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx video.mp4 frames=100

# Process every 3rd frame (faster, lower FPS output)
./lua_runner scripts/yolo11_detector.lua models/yolo11n.onnx video.mp4 skip=3
```

### Memory Monitoring (Video Mode)

Automatically enabled in video mode:

```
Tracks:
  - Initial memory (RSS)
  - Current memory per frame
  - Peak memory
  - Memory growth rate

Warnings:
  ⚠️  >10 KB/frame growth = Potential leak
  ❌ >100 KB/frame growth = Definite leak
```

**Disable** (not currently configurable, always on)

---

## 9.6 Platform-Specific Settings

### Embedded Linux (RISC-V/ARM)

**Recommended ONNX Runtime settings**:

```cpp
session_options.SetIntraOpNumThreads(2);   // Fewer threads
session_options.SetInterOpNumThreads(1);   // Disable inter-op
session_options.EnableCpuMemArena();       // Reduce fragmentation
```

**Recommended OpenCV settings**:

```cpp
// Use INTER_NEAREST for faster preprocessing (at cost of quality)
cv::resize(src, dst, size, 0, 0, cv::INTER_NEAREST);

// Reduce video buffer
cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
```

### x86_64 Desktop

**Recommended ONNX Runtime settings**:

```cpp
session_options.SetIntraOpNumThreads(8);   // More threads
session_options.SetInterOpNumThreads(4);
```

**Optional optimizations**:

```cpp
// Enable AVX2 (compile flag)
-mavx2

// Enable MKL (if available)
session_options.AppendExecutionProvider_Dnnl();
```

---

## 9.7 Environment Variables

### ONNX Runtime

```bash
# Logging level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
export ORT_LOGGING_LEVEL=2

# Enable profiling
export ORT_ENABLE_PROFILING=1
```

### OpenCV

```bash
# Number of threads for OpenCV operations
export OPENCV_THREAD_COUNT=4

# Disable OpenCL (if causing issues)
export OPENCV_OPENCL_RUNTIME=""
```

### Lua

```bash
# Lua module search path
export LUA_PATH="./scripts/?.lua;./scripts/?/init.lua;;"

# Lua C module search path
export LUA_CPATH="./?.so;./lib/?.so;;"
```

---

## 9.8 Configuration Best Practices

### Development

```
ONNX Runtime:
  - Optimization: ORT_ENABLE_ALL
  - Threads: 4 (or CPU cores / 2)
  - Logging: Level 2 (Warning)

OpenCV:
  - Interpolation: INTER_LINEAR
  - Quality: Default

Lua:
  - GC: Automatic
```

### Production (Embedded)

```
ONNX Runtime:
  - Optimization: ORT_ENABLE_ALL
  - Threads: 2-4 (avoid oversubscription)
  - Memory arena: Enabled
  - Logging: Level 3 (Error only)

OpenCV:
  - Interpolation: INTER_NEAREST (faster) or INTER_LINEAR (better quality)
  - Buffer: Minimal (CAP_PROP_BUFFERSIZE = 1)

Lua:
  - GC: Automatic (monitor for leaks)
```

### Testing

```
ONNX Runtime:
  - Profiling: Enabled
  - Logging: Level 1 (Info)

Video mode:
  - Memory monitoring: Enabled
  - Frame limit: Set (e.g., 100 frames)
```

---

## Summary

Configuration aspects:

✅ **ONNX Runtime**: Thread count, optimization level, execution providers
✅ **OpenCV**: Image I/O, video capture, processing parameters
✅ **Lua**: Standard libraries, GC settings
✅ **Model scripts**: Preprocessing config, thresholds
✅ **Runtime**: Command-line options, monitoring

**Key recommendations**:
- Use ORT_ENABLE_ALL optimization
- Tune thread count for platform (2-4 for embedded, 4-8 for desktop)
- Use C++ preprocessing (preprocess_config) for performance
- Enable memory monitoring for video (automatic)
- Adjust thresholds (conf_thres, iou_thres) per use case
