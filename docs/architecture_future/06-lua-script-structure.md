# 6. Lua Script Structure

This document describes the Lua script structure for multi-model pipelines.

---

## 6.1 Script Organization

### Single Script per Pipeline

```
scripts/
├─────────────────────────────────────────────────────────────────┤
├─────────────────────────────────────────────────────────────────┤
├─────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

### Script Sections

```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Script Structure                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Configuration                                               │
│     - Model paths                                               │
│     - Execution mode                                            │
│     - Model-specific parameters                                 │
│                                                                 │
│  2. Model Definitions (per model)                               │
│     - preprocess function                                       │
│     - postprocess function                                      │
│     - (Async: frame_interval)                                   │
│                                                                 │
│  3. Fusion Function (Parallel-Sync only)                        │
│                                                                 │
│  4. Pipeline Creation                                           │
│                                                                 │
│  5. Main Loop (optional)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.2 Configuration Section

### Mode Selection

```
┌─────────────────────────────────────────────────────────────────┐
│  Mode Configuration                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  mode = "single"          -- 1 model                            │
│  mode = "serial"          -- 2-3 models, sequential             │
│  mode = "parallel_sync"   -- 2-3 models, concurrent + barrier   │
│  mode = "parallel_async"  -- 2-3 models, different rates        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Paths

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Path Configuration                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  models = {                                                     │
│      { name = "det",  path = "models/yolo11n.onnx" },           │
│      { name = "pose", path = "models/pose_256x192.onnx" },      │
│      { name = "reid", path = "models/osnet_x0_25.onnx" }        │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.3 Preprocess Function

### Function Signature

```
function preprocess(ctx) → tensor | nil

Parameters:
  ctx.image      -- Original image (cv::Mat wrapper)
  ctx.results    -- Previous model results (Serial mode)
  ctx.frame_id   -- Frame identifier

Returns:
  tensor         -- Input tensor for model
  nil            -- Skip this model
```

### Common Operations

```
┌─────────────────────────────────────────────────────────────────┐
│  Preprocess Common Patterns                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Resize and Normalize (Detection)                            │
│     - Resize to model input size                                │
│     - Convert to float tensor                                   │
│     - Normalize with mean/std                                   │
│                                                                 │
│  2. ROI Crop and Batch (Secondary model)                        │
│     - Get detections from ctx.results                           │
│     - Crop each ROI from original image                         │
│     - Resize each crop                                          │
│     - Stack into batch tensor                                   │
│                                                                 │
│  3. Skip Condition                                              │
│     - Check if previous results are empty                       │
│     - Return nil to skip model                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.4 Postprocess Function

### Function Signature

```
function postprocess(output, ctx) → result

Parameters:
  output         -- Model output tensor(s)
  ctx            -- Full context

Returns:
  result         -- Parsed result (any Lua type)
```

### Common Operations

```
┌─────────────────────────────────────────────────────────────────┐
│  Postprocess Common Patterns                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Detection Output Parsing                                    │
│     - Extract boxes, scores, classes                            │
│     - Apply confidence threshold                                │
│     - Run NMS                                                   │
│     - Scale coordinates to original image                       │
│                                                                 │
│  2. Keypoint Parsing                                            │
│     - Extract keypoint coordinates                              │
│     - Map to original image coordinates                         │
│     - Associate with detection boxes                            │
│                                                                 │
│  3. Feature Extraction                                          │
│     - Normalize feature vectors                                 │
│     - Convert to Lua table                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.5 Fusion Function (Parallel-Sync)

### Function Signature

```
function fusion(results) → combined_result

Parameters:
  results        -- { model1: r1, model2: r2, ... }

Returns:
  combined       -- Fused result
```

### Example Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  Fusion Example: Instance Segmentation                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│    results.det = { boxes, scores, classes }                     │
│    results.seg = { mask: HxW }                                  │
│                                                                 │
│  Process:                                                       │
│    For each box:                                                │
│      - Extract mask region                                      │
│      - Create instance mask                                     │
│                                                                 │
│  Output:                                                        │
│    { instances: [{ box, mask, class, score }, ...] }            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.6 Async Configuration (Parallel-Async)

### Worker Parameters

```
┌─────────────────────────────────────────────────────────────────┐
│  Parallel-Async Worker Configuration                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  model = {                                                      │
│      name = "det",                                              │
│      path = "model.onnx",                                       │
│      preprocess = preprocess_det,                               │
│      postprocess = postprocess_det,                             │
│                                                                 │
│      -- Async-specific:                                         │
│      frame_interval = 1,    -- Process every frame              │
│      queue_size = 2,        -- Buffer 2 frames                  │
│      drop_on_overflow = true                                    │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Frame Interval Examples

| Model Type | Interval | Effective Rate |
|------------|----------|----------------|
| Detection | 1 | 30 FPS |
| Scene | 10 | 3 FPS |
| Anomaly | 30 | 1 FPS |

---

## 6.7 Context Usage

### Accessing Previous Results (Serial)

```
┌─────────────────────────────────────────────────────────────────┐
│  Serial Mode: Accessing Previous Results                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model 1 (Detection):                                           │
│    ctx.results = {}  -- empty initially                         │
│                                                                 │
│  Model 2 (Pose):                                                │
│    ctx.results.det = [...]  -- detections available             │
│                                                                 │
│  Model 3 (ReID):                                                │
│    ctx.results.det = [...]  -- detections                       │
│    ctx.results.pose = [...] -- keypoints                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Original Image Access

```
┌─────────────────────────────────────────────────────────────────┐
│  Accessing Original Image for ROI Crop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  In preprocess function:                                        │
│    ctx.image              -- Full original image                │
│    ctx.image:width()      -- Image width                        │
│    ctx.image:height()     -- Image height                       │
│    ctx.image:crop(x,y,w,h) -- Extract ROI                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.8 Error Handling in Lua

### Skip Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│  Skip Model When No Input                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  In preprocess:                                                 │
│    local dets = ctx.results.det                                 │
│    if dets == nil or #dets == 0 then                            │
│        return nil  -- Signal: skip this model                   │
│    end                                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Validation

```
┌─────────────────────────────────────────────────────────────────┐
│  Input Validation                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  In preprocess:                                                 │
│    assert(ctx.image ~= nil, "Image is nil")                     │
│    assert(ctx.image:width() > 0, "Invalid image size")          │
│                                                                 │
│  In postprocess:                                                │
│    assert(output ~= nil, "Model output is nil")                 │
│    local shape = output:shape()                                 │
│    assert(#shape == 3, "Unexpected output dimensions")          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.9 Script Template

### Minimal Template

```
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline Script Template                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  -- 1. Configuration                                            │
│  local config = {                                               │
│      mode = "serial",  -- or "single", "parallel_sync", etc.    │
│      models = {                                                 │
│          { name = "model1", path = "path/to/model1.onnx" },     │
│          { name = "model2", path = "path/to/model2.onnx" },     │
│      }                                                          │
│  }                                                              │
│                                                                 │
│  -- 2. Model 1 functions                                        │
│  local function pre1(ctx) ... end                               │
│  local function post1(output, ctx) ... end                      │
│                                                                 │
│  -- 3. Model 2 functions                                        │
│  local function pre2(ctx) ... end                               │
│  local function post2(output, ctx) ... end                      │
│                                                                 │
│  -- 4. Attach functions to config                               │
│  config.models[1].preprocess = pre1                             │
│  config.models[1].postprocess = post1                           │
│  config.models[2].preprocess = pre2                             │
│  config.models[2].postprocess = post2                           │
│                                                                 │
│  -- 5. Create and return pipeline                               │
│  return Pipeline.new(config)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.10 Best Practices

### Do's

| Practice | Reason |
|----------|--------|
| Validate inputs | Catch errors early |
| Return nil to skip | Clean skip mechanism |
| Keep preprocess fast | Minimize CPU overhead |
| Reuse tensor buffers | Reduce allocations |

### Don'ts

| Anti-Pattern | Reason |
|--------------|--------|
| Heavy computation in Lua | Lua is slow for math |
| Large table copying | Memory overhead |
| Global variables | Thread safety |
| Blocking I/O | Blocks pipeline |
