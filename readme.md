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

### YOLOv5n Performance (640x640, tested with precise measurements)

**C++ Baseline (`cpp_infer`):**
```
Preprocess:  19 ms
Inference:  144 ms
Postprocess: 21 ms
Total:     ~184 ms
```

**Lua Implementation (`yolov5_benchmark.lua`):**
```
Preprocess:  ~16 ms (letterbox 2ms + to_tensor 14ms)
Inference:  ~144 ms (same ONNX Runtime)
Postprocess: ~23 ms (max_with_argmax 9ms + filtering 14ms + NMS 0.03ms)
Total:     ~183 ms
```

**Key Findings:**
- Lua preprocessing is actually **faster** than C++ (16ms vs 19ms)
- Postprocessing is nearly identical (23ms vs 21ms)
- ONNX inference time is unchanged (both use the same C++ runtime)
- The Tensor API adds negligible overhead (<0.02ms for tensor operations)
- Main cost is in Lua loop for proposal filtering (~14ms)

**Detailed profiling available:** Run `./model_infer scripts/yolov5_benchmark.lua models/yolov5n.onnx images/zidane.jpg` to see timing breakdown for each operation.

*Tested on Linux x64 AMD Ryzen 9 3900X 12-Core Processor.*

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
  - `yolov5_benchmark.lua`: YOLOv5 with detailed timing profiling.
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
local nn = lua_nn

-- Create inference session
local session = nn.Session.new(model_path)
local outputs = session:run(input_tensor)
local output = outputs["output0"]  -- Get output by name

-- ========== Level 1: Shape Operations (Zero-copy) ==========
local sliced = output:slice(dim, start, end, step)  -- Slice along dimension
local squeezed = output:squeeze(dim)                 -- Remove dimension
local unsqueezed = output:unsqueeze(dim)             -- Add dimension
local reshaped = output:reshape({-1, 85})            -- Reshape
local transposed = output:transpose()                -- Transpose 2D
local transposed = output:transpose_dims({1, 0, 2})  -- Permute dimensions
local col = output:get_column(idx)                   -- Get single column
local cols = output:slice_columns(start, end)        -- Slice columns

-- ========== Level 2: Math Operations ==========
-- Element-wise operations
local added = tensor:add(5.0)             -- Add scalar
local added = tensor:add_tensor(other)    -- Add tensor
local scaled = tensor:mul(2.0)            -- Multiply scalar
local divided = tensor:div(2.0)           -- Divide scalar

-- In-place operations (avoid allocation)
tensor:add_(5.0)   -- Modify in-place
tensor:sub_(1.0)
tensor:mul_(2.0)
tensor:div_(2.0)

-- Reduction operations
local sum_val = tensor:sum(axis, keepdims)    -- Sum along axis
local mean_val = tensor:mean(axis, keepdims)  -- Mean along axis
local max_val = tensor:max(axis, keepdims)    -- Max along axis
local min_val = tensor:min(axis, keepdims)    -- Min along axis
local argmax_idx = tensor:argmax(axis)        -- Argmax (returns Lua table)
local argmin_idx = tensor:argmin(axis)        -- Argmin (returns Lua table)

-- ⚡ Fused operation (single pass, recommended!)
local result = tensor:max_with_argmax(axis)   -- Returns {values=Tensor, indices=table}
local max_scores = result.values
local class_ids = result.indices

-- Activation functions
local activated = tensor:sigmoid()
local normalized = tensor:softmax(axis)
local exp_t = tensor:exp()
local log_t = tensor:log()

-- Comparison operations
local mask = tensor:gt(0.5)   -- Greater than
local mask = tensor:ge(0.5)   -- Greater or equal
local mask = tensor:lt(0.5)   -- Less than
local mask = tensor:le(0.5)   -- Less or equal
local mask = tensor:eq(0.5)   -- Equal

-- ========== Level 3: Advanced Operations ==========
-- ⚡ Vectorized filtering (High Performance!)
local indices = tensor:where_indices(threshold, "ge")  -- Get indices where >= threshold
local filtered = tensor:index_select(dim, indices)     -- Select by indices
local columns = tensor:extract_columns(indices)        -- Extract columns -> {{row1}, {row2}, ...}
local nz = tensor:nonzero()                            -- Non-zero indices

-- TopK
local result = tensor:topk_new(k, axis, largest)  -- Returns {values, indices}

-- Gather/Concat/Split
local gathered = tensor:gather(axis, indices_tensor)
local concat = nn.Tensor.concat({t1, t2, t3}, axis)
local splits = tensor:split(num_splits, axis)

-- Data conversion
local tbl = tensor:to_table()      -- Convert to Lua table (use after filtering!)
local str = tensor:to_string(10)   -- String representation
local val = tensor:get(idx)        -- Get single element (0-indexed)
local val = tensor:at(i, j)        -- Get 2D element (0-indexed)
tensor:set(idx, value)             -- Set single element
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

-- ⚡ Step 1: Fused max + argmax (single pass over 672K elements)
local result = scores:max_with_argmax(0)  -- {values=Tensor[8400], indices=table[8400]}
local max_scores = result.values
local class_ids = result.indices

-- ⚡ Step 2: Get valid indices in C++ (fast filtering)
local valid_indices = max_scores:where_indices(conf_thres, "ge")  -- Returns ~50 indices

-- ⚡ Step 3: Extract only valid data (batch operation in C++)
local filtered_boxes = boxes:extract_columns(valid_indices)  -- {{cx,cy,w,h}, ...}
local filtered_scores = max_scores:index_select(0, valid_indices):to_table()  -- Only 50 elements

-- Now loop over small filtered set (~50 iterations instead of 8,400!)
for i = 1, #valid_indices do
    local idx = valid_indices[i]
    local box_data = filtered_boxes[i]
    local score = filtered_scores[i]
    local cls_id = class_ids[idx + 1]  -- Lua 1-indexed
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

**After (Fast - 200ms):**
```lua
-- Fused max + argmax (single pass)
local result = scores:max_with_argmax(0)
local max_scores, class_ids = result.values, result.indices

-- C++ filtering
local valid_indices = max_scores:where_indices(threshold, "ge")  -- ~50 indices
local filtered_boxes = boxes:extract_columns(valid_indices)       -- 50 boxes
local filtered_scores = max_scores:index_select(0, valid_indices):to_table()

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

- **YOLO11 Detection**: [scripts/yolo11_tensor_detector.lua](scripts/yolo11_tensor_detector.lua) - Vectorized tensor operations
- **YOLO11 Segmentation**: [scripts/yolo11_tensor_seg.lua](scripts/yolo11_tensor_seg.lua) - Instance segmentation with masks
- **YOLO11 Pose**: [scripts/yolo11_tensor_pose.lua](scripts/yolo11_tensor_pose.lua) - 17 COCO keypoints
- **YOLOv5 Detection**: [scripts/yolov5_tensor_detector.lua](scripts/yolov5_tensor_detector.lua) - Classic anchor-based detection
- **YOLOv5 Benchmark**: [scripts/yolov5_benchmark.lua](scripts/yolov5_benchmark.lua) - Detailed timing profiling

### Testing Your Script

```bash
# Run your script
./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg

# Test tensor operations
./build/test_tensor scripts/test_tensor_api.lua

# Benchmark performance
time ./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg
```

## 🧪 Test Results

### Tensor API Unit Tests

All 98 unit tests pass, covering:

| Category | Tests | Description |
|----------|-------|-------------|
| Level 1: Shape | 15 | slice, squeeze, reshape, transpose |
| Level 2: Math | 25 | add, sub, mul, div, in-place ops |
| Level 3: Reduction | 12 | sum, mean, max, min, argmax, argmin |
| Level 4: Activation | 6 | sigmoid, softmax, exp, log |
| Level 5: Comparison | 5 | gt, ge, lt, le, eq |
| Level 6: Advanced | 20 | where_indices, index_select, extract_columns, max_with_argmax |
| Level 7: Gather/Split | 7 | gather, concat, split |
| Level 8: Non-contiguous | 8 | Operations on sliced tensors |

```bash
# Run all tests
./build/test_tensor scripts/test_tensor_api.lua

# Expected output:
# 测试总结: 98/98 通过
# ✓ 所有测试通过!
```

### YOLO Detection Tests

| Script | Model | Detections | Status |
|--------|-------|------------|--------|
| yolo11_tensor_detector.lua | yolo11n.onnx | 3 (person, person, tie) | ✅ |
| yolo11_tensor_seg.lua | yolo11n-seg.onnx | 3 | ✅ |
| yolo11_tensor_pose.lua | yolo11n-pose.onnx | 2 | ✅ |
| yolov5_tensor_detector.lua | yolov5n.onnx | 3 | ✅ |

### Postprocess Timing Breakdown

Detailed timing for YOLO11n postprocess (640x640, 8400 boxes):

```
Operation                        Time
─────────────────────────────────────
slice+squeeze+cont boxes         0.02 ms
slice+squeeze+cont scores        3.77 ms  (672K elements)
max_with_argmax (fused)          0.57 ms
where_indices                    0.07 ms
extract_columns                  0.02 ms
index_select + to_table          0.01 ms
build proposals (Lua loop)       0.02 ms
NMS                              0.01 ms
─────────────────────────────────────
Total Postprocess               ~4.5 ms
```
