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
The main engine uses Lua scripts to define the pipeline.

```bash
# Syntax
./model_infer <script_path> <model_path> <image_path> [show]

# Example
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg

# With Visualization
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg show
```

### Pure C++ Inference (Benchmark)
A hardcoded C++ implementation is provided for performance comparison.

```bash
# Syntax
./cpp_infer <model_path> <image_path> [show]

# Example
./cpp_infer ../models/yolov5n.onnx ../images/zidane.jpg
```

## � Benchmark Results

Comparison between Lua-scripted engine (`model_infer`) and pure C++ implementation (`cpp_infer`) on YOLOv5n.

| Metric | Lua Engine (`model_infer`) | C++ Engine (`cpp_infer`) | Difference |
| :--- | :--- | :--- | :--- |
| **Inference Time** (Real) | ~270 ms | ~240 ms | C++ is ~11% faster |
| **Memory Usage** (RSS) | ~150 MB | ~145 MB | C++ uses ~5 MB less |

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

#### ⚡ DO: Use Tensor Operations Directly
```lua
-- GOOD: Fast tensor operations (< 3 μs)
local boxes = output:slice(1, 0, 4)      -- Extract box coordinates
local scores = output:slice(1, 4, 85)    -- Extract class scores
local max_scores = scores:max(1)         -- Get max score per box
local class_ids = scores:argmax(1)       -- Get class IDs

-- Apply thresholding with tensor operations
local mask = max_scores:gt(conf_threshold)  -- Create boolean mask
```

#### 🐌 DON'T: Convert to Lua Tables Unless Necessary
```lua
-- BAD: to_table() is extremely slow for large tensors
-- Converting [25200,85] takes ~234ms (99% of overhead!)
local table_data = output:to_table()
for i = 1, #table_data do
    -- Process in Lua loop...
end
```

#### 🎯 Strategy: Minimize Data Transfers

**Option 1: Use Filter APIs (Recommended)**
```lua
-- Use specialized C++ methods that operate on tensors directly
local results = session:filter_yolo(output, conf_thresh, iou_thresh, labels)
```

**Option 2: Tensor Operations + Small Table Conversion**
```lua
-- Filter with tensor ops first, then convert only filtered results
local high_conf_mask = scores:gt(0.5)     -- Tensor operation
local filtered = output:masked_select(high_conf_mask)  -- Tensor operation
local small_table = filtered:to_table()   -- Only convert filtered data
```

**Option 3: Hybrid Approach**
```lua
-- Extract key information as small tensors/scalars
local n_boxes = output:size(0)
for i = 0, n_boxes - 1 do
    local box = output:slice(0, i, i+1)   -- Get single box (still tensor)
    local score = box:slice(1, 4, 5):item()  -- Extract confidence as scalar
    if score > threshold then
        local box_coords = box:slice(1, 0, 4):to_table()[1]  -- Convert only this box
        -- Process box_coords...
    end
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

### Performance Comparison

| Implementation | Time (ms) | Notes |
|:---------------|----------:|:------|
| C++ (`cpp_infer`) | 254 | Baseline |
| Lua + `filter_yolo` | 282 | 11% overhead, **recommended** |
| Lua + Tensor API | 515 | 83% overhead due to `to_table()` conversion |

**Recommendation**: Use specialized filter APIs (like `filter_yolo`) for production. Use Tensor API for prototyping and custom postprocessing where flexibility is more important than raw speed.

### Testing Your Script

```bash
# Run your script
./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg

# Test tensor operations
./build/test_tensor lua scripts/test_tensor_api.lua

# Benchmark performance
time ./build/model_infer scripts/your_script.lua models/model.onnx images/test.jpg
```
