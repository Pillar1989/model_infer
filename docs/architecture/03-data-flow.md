# 3. Data Flow

This document describes the complete data flow from application entry to inference output.

---

## 3.1 Application Entry Point

### Command Line Interface

```bash
# Inference mode:
./lua_runner <script.lua> <model.onnx> <input> [options]

# Test mode:
./lua_runner <test_script.lua>
```

**Options**:
- `show` - Display window during processing
- `save=path` - Save output
- `frames=N` - Process only N frames (video)
- `skip=N` - Process every Nth frame (video)

### Initialization Flow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Parse Arguments                                 │
└─────────────────────────────────────────────────────────┘
Extract: script_path, model_path, input_path, options

┌─────────────────────────────────────────────────────────┐
│ Step 2: Initialize Inference Context                    │
└─────────────────────────────────────────────────────────┘

2.1. Create Lua state
     L = luaL_newstate()
     luaL_openlibs(L)

2.2. Register C++ modules
     lua_cv::register_module(L)
     lua_nn::register_module(L)
     lua_utils::register_module(L)

2.3. Load ONNX model
     session = new lua_nn::Session(model_path)
     │
     └─→ inference::OnnxSession:
           - Create Ort::Env, Ort::Session
           - Query model metadata
           - Configure session options

2.4. Load Lua script
     luaL_dofile(L, script_path)
     │
     └─→ Script returns Model table:
           Model = {
             config = {...}
             preprocess_config = {...}  -- optional
             preprocess = function...    -- optional
             postprocess = function...   -- required
           }

2.5. Extract functions
     preprocess_config = Model["preprocess_config"]
     preprocess = Model["preprocess"]
     postprocess = Model["postprocess"]
```

---

## 3.2 Dual Preprocessing Paths

The system supports two preprocessing modes:

### Path Selection Logic

```
if preprocess_config exists AND type in PreprocessRegistry:
    ┌────────────────────────────────────┐
    │ C++ Preprocessing (Fast)           │
    └────────────────────────────────────┘
    Use PreprocessRegistry::run()
else:
    ┌────────────────────────────────────┐
    │ Lua Preprocessing (Fallback)       │
    └────────────────────────────────────┘
    Call Model.preprocess()
```

### C++ Preprocessing Path

```
Input: lua_cv::Image
  │
  ├─→ Extract config:
  │     type = "letterbox"
  │     input_size = {640, 640}
  │     stride = 32
  │     fill_value = 114
  │
  └─→ PreprocessRegistry::run(type, img, L, config)
        │
        └─→ preprocess_letterbox(img, L, config):
              │
              1. Calculate scale ratio
                   r = min(target_h/ori_h, target_w/ori_w)
              │
              2. img.resize(new_w, new_h)
                   [OpenCV cv::resize]
              │
              3. Calculate padding (aligned to stride)
                   dw = (target_w - new_w) % stride
                   dh = (target_h - new_h) % stride
              │
              4. img.pad(top, bottom, left, right, fill_value)
                   [OpenCV cv::copyMakeBorder]
              │
              5. tensor = img.to_tensor(1/255, {0,0,0}, {1,1,1})
                   HWC → CHW conversion + normalization
              │
              6. meta = {scale, pad_x, pad_y, ori_w, ori_h}
              │
              └─→ return PreprocessResult(tensor, meta)

Output: Tensor(1, 3, H, W) + meta (Lua table)
```

### Lua Preprocessing Path

```
Input: lua_cv::Image
  │
  └─→ Call Model.preprocess(img)
        │
        └─→ Lua function (e.g., scripts/lib/preprocess.lua):
              │
              1. Calculate scale ratio (Lua)
              │
              2. img:resize(new_w, new_h)
                   [Calls C++ lua_cv::Image::resize]
              │
              3. Calculate padding (Lua)
              │
              4. img:pad(top, bottom, left, right, fill_value)
                   [Calls C++ lua_cv::Image::pad]
              │
              5. tensor = img:to_tensor(1/255, {0,0,0}, {1,1,1})
                   [Calls C++ lua_cv::Image::to_tensor]
              │
              6. meta = {scale, pad_x, pad_y, ori_w, ori_h}
              │
              └─→ return tensor, meta

Output: Tensor(1, 3, H, W) + meta (Lua table)
```

**Note**: Both paths produce identical results. C++ path is faster due to reduced Lua/C++ boundary crossings.

---

## 3.3 Inference Execution

```
Input: Tensor(1, 3, H, W)
  │
  └─→ session:run(tensor)
        │
        └─→ lua_nn::Session::run(L, tensor):
              │
              1. Extract tensor data:
                   shape = tensor.shape()
                   data = tensor.data()
              │
              2. Call inference::OnnxSession::run():
                   │
                   a. Create Ort::Value from input
                   │
                   b. Type conversion (if needed):
                        float → float16 (if model expects FP16)
                   │
                   c. session_.Run(input_names, inputs, output_names)
                        [ONNX Runtime execution ~100ms]
                   │
                   d. Extract outputs from Ort::Value
                   │
                   e. Type conversion (if needed):
                        float16 → float
                   │
                   └─→ return vector<vector<float>>
              │
              3. Wrap outputs in Tensors:
                   For each output:
                     tensor = Tensor(data, shape)
              │
              4. Build Lua table:
                   {output0 = Tensor, output1 = Tensor, ...}
              │
              └─→ return to Lua

Output: Lua table of Tensors
```

---

## 3.4 Postprocessing (YOLO Example)

```
Input: outputs (Lua table), meta (Lua table)
  │
  └─→ Model.postprocess(outputs, meta)
        │
        └─→ Lua script (scripts/yolo11_tensor_detector.lua):
              │
              1. Extract output:
                   output = outputs["output0"]  -- [1, 84, 8400]
              │
              2. Separate boxes and scores:
                   boxes = output:slice(1, 0, 4):squeeze(0)
                          :contiguous()
                   -- [4, 8400], contiguous() ~4ms copy

                   scores = output:slice(1, 4, 84):squeeze(0)
                           :contiguous()
                   -- [80, 8400], contiguous() for efficiency
              │
              3. Find max score per detection:
                   result = scores:max_with_argmax(0)
                   -- Fused operation ~0.5ms
                   max_scores = result.values    -- [8400]
                   class_ids = result.indices    -- [8400]
              │
              4. Filter by confidence:
                   indices = max_scores:where_indices(0.25, "ge")
                   -- Returns: {42, 157, 891, ...}
              │
              5. Extract filtered data:
                   boxes_lua = boxes:extract_columns(indices)
                   -- Returns: {{cx,cy,w,h}, ...}

                   scores_tensor = max_scores:index_select(0, indices)
                   scores_lua = scores_tensor:to_table()
              │
              6. Build proposals (Lua loop):
                   for i = 1, #indices do
                     box = boxes_lua[i]
                     cls = class_ids[indices[i] + 1]
                     conf = scores_lua[i]

                     -- Center to corner format
                     x = box[1] - box[3]/2
                     y = box[2] - box[4]/2

                     proposals[i] = {
                       x=x, y=y, w=box[3], h=box[4],
                       score=conf, class_id=cls, label=...
                     }
                   end
              │
              7. Scale coordinates to original image:
                   for _, box in ipairs(proposals) do
                     box.x = (box.x - meta.pad_x) / meta.scale
                     box.y = (box.y - meta.pad_y) / meta.scale
                     box.w = box.w / meta.scale
                     box.h = box.h / meta.scale
                   end
              │
              8. Apply NMS:
                   final = utils.nms(proposals, iou_threshold)
              │
              └─→ return final

Output: Lua table of detections
  {
    {x=120, y=50, w=100, h=200, score=0.85, class_id=0, label="person"},
    {x=300, y=100, w=80, h=150, score=0.72, class_id=0, label="person"},
    ...
  }
```

---

## 3.5 Complete Pipeline Visualization

```
┌────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                          │
└────────────────────────────────────────────────────────────────┘
    Lua State            ONNX Model              Lua Script
        │                    │                       │
        ├─→ Register         │                       │
        │   modules          │                       │
        │                    ├─→ Load (~80ms)       │
        │                    │                       ├─→ dofile
        │                    │                       │   Extract funcs
        │                    │                       │
        ◄────────────────────┴───────────────────────┘
                InferenceContext ready

┌────────────────────────────────────────────────────────────────┐
│                    PER-FRAME PROCESSING                        │
└────────────────────────────────────────────────────────────────┘

Input Image                                        Output Detections
(RGB, HWC)                                         (Lua table)
    │                                                   ▲
    ▼                                                   │
┌───────────────────┐                        ┌─────────────────┐
│ PREPROCESSING     │                        │ POSTPROCESSING  │
│                   │                        │                 │
│ C++ or Lua path   │                        │ - Tensor ops    │
│ - resize, pad     │                        │ - Filter        │
│ - to_tensor       │                        │ - Coord scale   │
│ (~12ms)           │                        │ - NMS           │
└───────────────────┘                        │ (~5ms)          │
    │                                        └─────────────────┘
    │ Tensor(1,3,H,W) + meta                        ▲
    ▼                                                │
┌────────────────────────────────────────────────────┐
│ INFERENCE: session:run(tensor)                     │
│                                                    │
│ - Type conversion (if needed)                      │
│ - ONNX Runtime execution                           │
│ - Output extraction                                │
│ (~100ms)                                           │
└────────────────────────────────────────────────────┘
                    │
                    │ Tensor(1,84,8400)
                    │
                    ▼

Total: ~200ms per frame (YOLO11n, 640×640, RISC-V C906)
```

---

## 3.6 Video Processing Loop

```
main.cpp video mode:

cv::VideoCapture cap(video_path)
   │
   └─→ Frame loop:
         │
         ├─→ Read frame N
         │     │
         │     └─→ lua_cv::Image img(frame)
         │           │
         │           └─→ run_inference(ctx, img)
         │                 │
         │                 ├─→ Preprocessing
         │                 ├─→ Inference
         │                 └─→ Postprocessing
         │                       │
         │                       └─→ detections
         │
         ├─→ draw_detections(frame, detections)
         │     - Draw boxes, labels, keypoints
         │
         ├─→ VideoWriter::write(frame)
         │
         └─→ cv::imshow (if enabled)

Timeline (sequential):
  Frame 0: [Read][Preprocess][Infer][Post][Draw][Write]
  Frame 1:                                              [Read][Preprocess][Infer][Post][Draw][Write]
  Frame 2:                                                                                          [Read]...

  0ms    20ms    120ms   125ms  130ms
  │──────│───────│──────│─────│───────│
  Read   Prep    Infer  Post  Draw

Note: No overlap between frames (single-threaded)
```

### Video Statistics

The video mode tracks:
- **FPS**: Frames processed per second
- **Memory**: RSS usage per frame
- **Detections**: Count per frame

After processing:
```
=== Processing Complete ===
Processed: 100 frames
Time: 20.5 s
Average FPS: 4.88
Per frame: 205.0 ms

=== Memory Summary ===
Initial:  50.2 MB
Final:    52.1 MB
Peak:     54.3 MB
Increase: 1.9 MB
Per frame: 0.02 KB  ← Low indicates no leak
```

---

## 3.7 Image Processing Flow

```
main.cpp image mode:

lua_cv::imread(path)
   │
   └─→ lua_cv::Image img
         │
         ├─→ run_inference(ctx, img)
         │     └─→ detections
         │
         ├─→ print_results(detections)
         │     Print to console
         │
         └─→ (if show or save):
               cv::Mat draw_img = cv::imread(path)
               draw_detections(draw_img, detections)
               │
               ├─→ cv::imwrite(save_path)
               └─→ cv::imshow + cv::waitKey
```

---

## 3.8 Data Type Flow

Tracking data types through the pipeline:

```
cv::Mat                (uint8, BGR, HWC)
   ↓
lua_cv::Image          (wraps cv::Mat)
   ↓
tensor::Tensor         (float, RGB, NCHW)
   ↓
ONNX Runtime           (float or float16)
   ↓
tensor::Tensor         (float, model output format)
   ↓
Lua table              (float values, nested tables)
   ↓
Visualization          (draw on cv::Mat uint8)
```

**Key transformations**:
- BGR → RGB (color space)
- HWC → CHW (layout)
- uint8 → float (type + normalize)
- float → float16 (if model needs FP16)

---

## Summary

The data flow emphasizes:

✅ **Dual preprocessing**: C++ (fast) or Lua (flexible)
✅ **Type safety**: Explicit conversions at boundaries
✅ **Sequential execution**: Simple, predictable flow
✅ **Memory efficiency**: Zero-copy tensor views
✅ **Monitoring**: FPS, memory tracking in video mode
