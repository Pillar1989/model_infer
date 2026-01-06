# Model Inference Engine

A high-performance, flexible model inference engine combining the speed of C++ with the scripting flexibility of Lua. Built with **ONNX Runtime**, **OpenCV**, and **LuaIntf**.

## 🚀 Features

- **Hybrid Architecture**: Core inference and heavy lifting in C++, business logic and pre/post-processing in Lua.
- **High Performance**: Zero-copy tensor passing where possible, optimized OpenCV operations.
- **Flexible**: Modify inference pipelines (preprocessing, NMS, etc.) without recompiling C++.
- **Robust**: Handles dynamic input shapes (Auto-padding), Float16/Float32 type mismatch automatically.
- **Comparison Tool**: Includes a pure C++ implementation (`cpp_infer`) to benchmark against the Lua-scripted engine.

## 🛠️ Prerequisites

- **C++ Compiler**: Supporting C++20.
- **CMake**: Version 3.18 or higher.
- **OpenCV**: Version 4.x installed on the system.
- **ONNX Runtime**: Prebuilt binaries (see below).

## 📦 Setup & Build

### 1. Prepare ONNX Runtime
**⚠️ Important**: This project requires ONNX Runtime prebuilt binaries.
You must manually ensure the `onnxruntime-prebuilt` directory contains the `include` and `lib` folders.

1. Download the ONNX Runtime (Linux x64) release (compatible with your system).
2. Extract it into the project root and rename/link it to `onnxruntime-prebuilt`.

Structure should look like:
```
project_root/
├── onnxruntime-prebuilt/
│   ├── include/
│   └── lib/
│       └── libonnxruntime.so
├── src/
├── scripts/
...
```

### 2. Build
```bash
mkdir build && cd build
cmake ..
make
```

## 🏃 Usage

### Lua Inference Engine (Recommended)
The main engine uses Lua scripts to define the pipeline. Supports both **images** and **videos** with automatic detection based on file extension.

#### Image Inference
```bash
# Syntax
./model_infer <script_path> <model_path> <image_path> [options]

# Basic usage
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg

# With visualization window
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg show

# Save result to file
./model_infer ../scripts/yolo11_seg.lua ../models/yolo11n-seg.onnx ../images/zidane.jpg save=output.jpg
```

#### Video Inference
Automatically processes video files (.mp4, .avi, .mov, etc.) with real-time memory monitoring.

```bash
# Basic usage (process all frames)
./model_infer ../scripts/yolo11_detector.lua ../models/yolo11n.onnx ../images/video.mp4

# With real-time display
./model_infer ../scripts/yolo11_seg.lua ../models/yolo11n-seg.onnx ../images/video.mp4 show

# Save output video
./model_infer ../scripts/yolo11_seg.lua ../models/yolo11n-seg.onnx ../images/video.mp4 save=output.mp4

# Process only first 100 frames (for testing)
./model_infer ../scripts/yolo11_detector.lua ../models/yolo11n.onnx ../images/video.mp4 frames=100

# Skip frames (process every 2nd frame for speed)
./model_infer ../scripts/yolo11_detector.lua ../models/yolo11n.onnx ../images/video.mp4 skip=2

# Combine options
./model_infer ../scripts/yolo11_seg.lua ../models/yolo11n-seg.onnx ../images/video.mp4 show save=out.mp4 frames=200 skip=2
```

**Video Features:**
- ✅ Real-time FPS and detection count display
- ✅ Memory leak detection (warns if >10 KB/frame growth)
- ✅ Progress tracking with memory usage monitoring
- ✅ Support for all common video formats

### Pure C++ Inference (Benchmark)
A hardcoded C++ implementation is provided for performance comparison.

```bash
# Syntax
./cpp_infer <model_path> <image_path> [show]

# Example
./cpp_infer ../models/yolov5n.onnx ../images/zidane.jpg
```

## 📊 Benchmark Results

Comparison between different implementations on YOLO11n detection task.

| Implementation | Time (ms) | vs C++ | Memory | Notes |
|:--------------|----------:|:------:|:------:|:------|
| **C++ (`cpp_infer`)** | 180 | - | ~145 MB | Baseline reference |
| **Lua + Vectorized Tensor API** | 190 | +5.6% | ~150 MB | **Recommended** - General-purpose API |
| **Lua + `filter_yolo`** | 290 | +61% | ~150 MB | Legacy specialized API |
| **Lua + Naive Tensor API** | 515 | +186% | ~160 MB | Poor performance (avoid) |

**Key Takeaway**: The new vectorized Tensor API achieves **near-C++ performance** (only 10ms slower) while maintaining full generality. This proves that **generic APIs can match specialized implementations** with proper design.

*Note: Tested on Linux x64 AMD Ryzen 9 3900X 12-Core Processor. "Inference Time" includes initialization, image loading, preprocessing, inference, and postprocessing.*

## �📂 Project Structure

- `src/`: C++ source code.
  - `modules/`: Lua bindings for CV and NN operations.
  - `bindings/`: Module registration and Lua context setup.
  - `main.cpp`: Lua engine entry point.
  - `cpp_main.cpp`: Pure C++ implementation.
  - `test_main.cpp`: Tensor API test program.
- `scripts/`: Lua scripts defining inference logic.
  - `yolo11_*.lua`: YOLO11 inference scripts (detection, segmentation, pose).
  - `yolov5_*.lua`: YOLOv5 inference scripts.
  - `*_tensor_*.lua`: Tensor API-based implementations.
  - `test_tensor*.lua`: Tensor API test scripts.
- `lua/`: Lua 5.5 source code (compiled as C++).
- `lua-intf-ex/`: C++/Lua binding library (LuaIntf).
- `onnxruntime-prebuilt/`: ONNX Runtime headers and libraries.
- `models/`: ONNX model files.
- `images/`: Test images.

## 📝 Writing High-Performance Lua Inference Scripts

### Architecture Overview

The inference pipeline consists of three stages:
1. **Preprocessing** (Lua + OpenCV): Load and prepare image data
2. **Inference** (C++ + ONNX Runtime): Execute the neural network
3. **Postprocessing** (Lua): Parse outputs and extract results

### Available Lua Modules

#### `lua_cv` - Computer Vision Operations
```lua
local cv = require "lua_cv"

-- Image operations
local img = cv.Image.load(image_path)
local resized = img:resize(640, 640)
local padded = img:letterbox(640, 640)  -- Preserve aspect ratio
local rgb = img:cvt_color(cv.COLOR_BGR2RGB)

-- Preprocessing for inference
local blob = img:to_blob()  -- Convert to CHW float32 tensor [C,H,W]
```

#### `lua_nn` - Neural Network & Tensor Operations
```lua
local nn = require "lua_nn"

-- Create inference session
local session = nn.InferenceSession.new(model_path)
session:warm_up(3)  -- Optional warmup

-- Run inference
local outputs = session:infer({input_blob})
local output = outputs[1]  -- Get first output tensor

-- Tensor operations (Level 1-3)
-- Level 1: Shape manipulation
local sliced = output:slice(0, 0, 1)      -- [1,N,M] -> [1,N,M]
local squeezed = output:squeeze(0)        -- [1,N,M] -> [N,M]
local reshaped = output:reshape({-1, 85}) -- Flatten to [N,85]
local transposed = output:transpose(0, 1) -- Swap dimensions

-- Level 2: Math operations
local added = tensor:add(5)               -- Element-wise add
local scaled = tensor:mul(2.0)            -- Element-wise multiply
local sum = tensor:sum()                  -- Reduce to scalar
local max_val = tensor:max()              -- Find maximum value
local argmax_idx = tensor:argmax()        -- Index of maximum

-- Level 3: Advanced operations
local activated = tensor:sigmoid()        -- Apply sigmoid
local normalized = tensor:softmax()       -- Apply softmax
local top_k = tensor:topk_lua(5)         -- Get top 5 values and indices
local table_data = tensor:to_table()     -- Convert to Lua table (SLOW!)

-- ⚡ NEW: Vectorized Filtering Operations (High Performance)
-- These methods enable near-C++ performance with generic Tensor API
local indices = tensor:where_indices(threshold, "ge")  -- Get indices where tensor >= threshold
local filtered = tensor:index_select(dim, indices)     -- Select elements at specific indices
local columns = tensor:extract_columns({1, 5, 10})     -- Extract multiple columns (optimized for [C,N])
local nz_indices = tensor:nonzero()                     -- Get indices of non-zero elements
```

#### `lua_utils` - Utility Functions
```lua
local utils = require "lua_utils"

-- Non-Maximum Suppression
local filtered = utils.nms(boxes, iou_threshold)

-- Box format conversion
local xyxy_box = utils.xywh2xyxy(xywh_box)
local xywh_box = utils.xyxy2xywh(xyxy_box)

-- Coordinate scaling
local scaled_box = utils.scale_boxes(box, orig_shape, new_shape)
```

### Performance Best Practices

#### ⚡ DO: Use Vectorized Filtering (BEST - Near C++ Speed!)
```lua
-- RECOMMENDED: Filter in C++, convert only small result set
-- Example: YOLO detection with 8,400 boxes -> ~50 valid boxes
local max_scores = scores:max(0, false)  -- [8400]
local class_ids = scores:argmax(0)       -- Lua table [8400]

-- 🚀 KEY: Get valid indices in C++ (fast)
local valid_indices = max_scores:where_indices(conf_thres, "ge")  -- Returns ~50 indices

-- 🚀 Extract only valid data (batch operation in C++)
local filtered_boxes = boxes:extract_columns(valid_indices)  -- {{cx,cy,w,h}, ...}
local filtered_scores = max_scores:index_select(0, valid_indices):to_table()  -- Only 50 elements

-- Now loop over small filtered set (~50 iterations instead of 8,400!)
for i = 1, #valid_indices do
    local box_data = filtered_boxes[i]
    local score = filtered_scores[i]
    -- Process...
end
```

#### ✅ DO: Use Tensor Operations Directly
```lua
-- GOOD: Fast tensor operations (< 3 μs)
local boxes = output:slice(1, 0, 4)      -- Extract box coordinates
local scores = output:slice(1, 4, 85)    -- Extract class scores
local max_scores = scores:max(1)         -- Get max score per box
local class_ids = scores:argmax(1)       -- Get class IDs
```

#### 🐌 DON'T: Convert Full Tensors to Tables
```lua
-- BAD: to_table() is extremely slow for large tensors
-- Converting [8400, 4] takes ~230ms (massive overhead!)
local boxes_table = boxes:to_table()
for i = 1, 8400 do
    -- Process all boxes in Lua loop... SLOW!
end

-- GOOD: Filter first, then convert
local valid_indices = scores:where_indices(threshold, "ge")  -- Returns ~50 indices
local filtered = boxes:extract_columns(valid_indices)  -- Only 50 boxes, fast!
```

#### 🎯 Migration Guide: Old API → Vectorized API

**Before (Slow - 515ms):**
```lua
local boxes_table = boxes:to_table()      -- 8,400 boxes converted
local scores_table = scores:to_table()    -- 8,400 scores converted
for i = 1, 8400 do
    if scores_table[i] >= threshold then  -- Lua loop checking 8,400 times
        -- Process box...
    end
end
```

**After (Fast - 190ms):**
```lua
local valid_indices = scores:where_indices(threshold, "ge")  -- C++ filters to ~50 indices
local filtered_boxes = boxes:extract_columns(valid_indices)   -- Extract 50 boxes
local filtered_scores = scores:index_select(0, valid_indices):to_table()  -- Convert 50 scores
for i = 1, #valid_indices do  -- Lua loop only 50 times!
    -- Process box...
end
```

### Script Template

Here's a minimal template for writing inference scripts:

```lua
local cv = require "lua_cv"
local nn = require "lua_nn"
local utils = require "lua_utils"

-- Parse arguments
if #arg < 2 then
    print("Usage: script.lua <model_path> <image_path>")
    os.exit(1)
end
local model_path = arg[1]
local image_path = arg[2]

-- 1. Preprocessing
local img = cv.Image.load(image_path)
local orig_h, orig_w = img:height(), img:width()
local input_img = img:letterbox(640, 640):cvt_color(cv.COLOR_BGR2RGB)
local blob = input_img:to_blob()

-- 2. Inference
local session = nn.InferenceSession.new(model_path)
session:warm_up(3)
local outputs = session:infer({blob})

-- 3. Postprocessing
local output = outputs[1]:squeeze(0)  -- Remove batch dimension

-- Use tensor operations for high-performance processing
local boxes = output:slice(1, 0, 4)
local obj_scores = output:slice(1, 4, 5)
local class_scores = output:slice(1, 5, output:size(1))

-- Apply confidence threshold
local max_class_scores = class_scores:max(1)
local class_ids = class_scores:argmax(1)
local final_scores = obj_scores:mul(max_class_scores)

-- For demonstration, convert to table (in production, use filter_yolo)
local results = {}
local score_table = final_scores:to_table()
for i, score in ipairs(score_table) do
    if score > 0.25 then
        table.insert(results, {
            box = boxes:slice(0, i-1, i):to_table()[1],
            score = score,
            class_id = class_ids:to_table()[i]
        })
    end
end

print(string.format("Found %d objects", #results))
```

### Example Scripts

- **YOLO11 Detection**: [scripts/yolo11_detector.lua](scripts/yolo11_detector.lua) - Uses `filter_yolo` API (fast)
- **YOLO11 Tensor Version**: [scripts/yolo11_tensor_detector.lua](scripts/yolo11_tensor_detector.lua) - Pure tensor operations
- **YOLO11 Segmentation**: [scripts/yolo11_seg.lua](scripts/yolo11_seg.lua) - Instance segmentation with masks
- **YOLO11 Pose**: [scripts/yolo11_pose.lua](scripts/yolo11_pose.lua) - 17 COCO keypoints
- **YOLOv5 Detection**: [scripts/yolov5_detector.lua](scripts/yolov5_detector.lua) - Classic anchor-based detection

### Performance Comparison (YOLO11n Detection)

| Implementation | Time (ms) | Overhead | Code Flexibility | Status |
|:---------------|----------:|:--------:|:----------------:|:------:|
| C++ (`cpp_infer`) | 180 | - | ❌ Low (hardcoded) | Baseline |
| **Lua + Vectorized Tensor API** | **190** | **+5.6%** | ✅ **High** | ✅ **Recommended** |
| Lua + `filter_yolo` | 290 | +61% | ⚠️ Medium (YOLO-specific) | Legacy |
| Lua + Naive Tensor API | 515 | +186% | ✅ High | ❌ Avoid |

**Key Insights:**
- ✅ **Vectorized Tensor API achieves near-C++ performance** while remaining fully generic
- ✅ Proper API design eliminates the need for task-specific optimizations
- ✅ `where_indices()` + `extract_columns()` + `index_select()` enable high-performance filtering
- ⚠️ Legacy `filter_yolo()` retained for backward compatibility but no longer needed

**Migration Path:** All `*_tensor_*.lua` scripts have been updated to use the vectorized API. See [scripts/yolo11_tensor_detector.lua](scripts/yolo11_tensor_detector.lua) for reference implementation.

### Testing Your Script

```bash
# Run your script
./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg

# Test tensor operations
./build/test_tensor lua scripts/test_tensor_api.lua

# Benchmark performance
time ./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg
```
