# 2. Multi-Model Scenarios

## 2.1 Execution Mode Classification

Based on model count and execution pattern, there are **4 execution modes**:

| Mode | Model Count | Execution Pattern | Use Case |
|------|-------------|-------------------|----------|
| **Single** | 1 | One model per frame | Detection, Classification |
| **Serial** | 2-3 | Sequential, dependent | Det → Pose → ReID |
| **Parallel-Sync** | 2-3 | Concurrent, barrier | Det ∥ Seg → Fusion |
| **Parallel-Async** | 2-3 | Independent, different rates | Det@30fps ∥ Scene@3fps |

---

## 2.2 Mode 1: Single Model

### Description

The simplest mode. One model processes each frame independently.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Single Model Execution                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame N                                                        │
│    │                                                            │
│    ▼                                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Lua: preprocess(image)                                    │ │
│  │    - Resize, normalize, convert to tensor                  │ │
│  └───────────────────────────────────────────────────────────┘  │
│    │                                                            │
│    ▼                                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  C++: session.run(tensor)                                  │ │
│  │    - Model inference (CPU/NPU)                             │ │
│  └───────────────────────────────────────────────────────────┘  │
│    │                                                            │
│    ▼                                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Lua: postprocess(output)                                  │ │
│  │    - Parse output, NMS, format results                     │ │
│  └───────────────────────────────────────────────────────────┘  │
│    │                                                            │
│    ▼                                                            │
│  Result                                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Timing

```
Total Time = T_preprocess + T_inference + T_postprocess

Example (YOLO11n, 640×640):
  Preprocess:   12 ms
  Inference:   100 ms (CPU) / 15 ms (NPU)
  Postprocess:   5 ms
  ─────────────────────
  Total:       117 ms (CPU) / 32 ms (NPU)
```

### Typical Applications

| Application | Model | Output |
|-------------|-------|--------|
| Object Detection | YOLO | Bounding boxes + classes |
| Image Classification | ResNet/MobileNet | Class probabilities |
| Semantic Segmentation | DeepLab | Pixel-wise mask |
| Pose Estimation | HRNet | Keypoints |

---

## 2.3 Mode 2: Serial (Multi-Model Sequential)

### Description

Multiple models execute sequentially. Each model can access results from all previous models. Typically, later models process ROIs (regions of interest) from earlier detections.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Serial Execution (3 Models)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame N                                                        │
│    │                                                            │
│    ▼                                                            │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  Model 1: Detection                                        ║ │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: pre1(image)                                    │  ║ │
│  ║  │    Input: original image                             │  ║ │
│  ║  │    Output: tensor [1, 3, 640, 640]                   │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  C++: infer(tensor)                                  │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: post1(output)                                  │  ║ │
│  ║  │    Output: detections = [{bbox, class, score}, ...]  │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│    │                                                            │
│    │ ctx.results.det = detections                               │
│    │ (Store in context for next model)                          │
│    ▼                                                            │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  Model 2: Pose Estimation                                  ║ │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: pre2(image, ctx)                               │  ║ │
│  ║  │    Input: image + ctx.results.det (N detections)     │  ║ │
│  ║  │    Process: crop N ROIs, resize each, stack          │  ║ │
│  ║  │    Output: tensor [N, 3, 256, 192]                   │  ║ │
│  ║  │    Skip: if N = 0, return nil (skip this model)      │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  C++: infer(tensor)  -- batch inference for N ROIs   │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: post2(output)                                  │  ║ │
│  ║  │    Output: keypoints = [[17 points], ...]            │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│    │                                                            │
│    │ ctx.results.pose = keypoints                               │
│    ▼                                                            │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  Model 3: ReID Feature Extraction                          ║ │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: pre3(image, ctx)                               │  ║ │
│  ║  │    Input: image + ctx.results.det                    │  ║ │
│  ║  │    Process: crop N ROIs, resize to 256×128           │  ║ │
│  ║  │    Output: tensor [N, 3, 256, 128]                   │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  C++: infer(tensor)                                  │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ║  ┌─────────────────────────────────────────────────────┐  ║  │
│  ║  │  Lua: post3(output)                                  │  ║ │
│  ║  │    Output: features = [[512-dim vector], ...]        │  ║ │
│  ║  └─────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│    │                                                            │
│    ▼                                                            │
│  Final Result: {                                                │
│    det: [{bbox, class, score}, ...],                            │
│    pose: [[17 keypoints], ...],                                 │
│    reid: [[512-dim feature], ...]                               │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Dependency** | Model N can access results from Model 1..N-1 via context |
| **Dynamic Count** | Model 2/3 may process N items (N = detection count) |
| **Batch Inference** | Multiple ROIs stacked into batch tensor |
| **Skip Logic** | If earlier model returns empty, later models can skip |
| **Timing** | Total = T1 + T2 + T3 (sequential sum) |

### Timing Example

```
Pedestrian Analysis (Det → Pose → ReID):

  Model 1 (Detection):
    Pre:    10 ms
    Infer:  15 ms (NPU)
    Post:    3 ms
    ─────────────────
    Subtotal: 28 ms

  Model 2 (Pose, N=5 persons):
    Pre:     5 ms (5 ROI crops)
    Infer:  20 ms (batch=5)
    Post:    2 ms
    ─────────────────
    Subtotal: 27 ms

  Model 3 (ReID, N=5 persons):
    Pre:     4 ms (5 ROI crops)
    Infer:  15 ms (batch=5)
    Post:    1 ms
    ─────────────────
    Subtotal: 20 ms

  Total: 75 ms → 13 FPS
```

### Typical Applications

| Pipeline | Models | Description |
|----------|--------|-------------|
| Pedestrian Analysis | Det → Pose → ReID | Person detection, pose estimation, re-identification |
| Vehicle Analysis | VehicleDet → PlateDet → OCR | Vehicle → License plate → Text recognition |
| Face Recognition | FaceDet → Align → Embed | Face detection, alignment, feature extraction |
| Action Recognition | Det → Pose → Action | Person → Skeleton → Action classification |

---

## 2.4 Mode 3: Parallel-Sync (Parallel Synchronous)

### Description

Multiple models execute concurrently on the same input. A barrier waits for all models to complete, then results are fused.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Parallel-Sync Execution (2 Models)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frame N                                                        │
│    │                                                            │
│    ├──────────────────────────────────────────┐                 │
│    │                                          │                 │
│    ▼                                          ▼                 │
│  ╔════════════════════════════╗  ╔════════════════════════════╗ │
│  ║  Model 1: Detection        ║  ║  Model 2: Segmentation     ║ │
│  ║                            ║  ║                            ║ │
│  ║  Lua: pre1(image)          ║  ║  Lua: pre2(image)          ║ │
│  ║    → [1,3,640,640]         ║  ║    → [1,3,512,512]         ║ │
│  ║                            ║  ║                            ║ │
│  ║  C++: infer (NPU Core 0)   ║  ║  C++: infer (NPU Core 1)   ║ │
│  ║                            ║  ║                            ║ │
│  ║  Lua: post1(output)        ║  ║  Lua: post2(output)        ║ │
│  ║    → boxes                 ║  ║    → mask                  ║ │
│  ║                            ║  ║                            ║ │
│  ╚════════════╤═══════════════╝  ╚════════════╤═══════════════╝ │
│               │                               │                 │
│               └───────────────┬───────────────┘                 │
│                               │                                 │
│                         ══════╧══════                           │
│                          BARRIER                                │
│                         (wait all)                              │
│                         ══════╤══════                           │
│                               │                                 │
│                               ▼                                 │
│               ┌───────────────────────────────┐                 │
│               │  Lua: fusion(boxes, mask)     │                 │
│               │                               │                 │
│               │  For each box:                │                 │
│               │    - Extract mask region      │                 │
│               │    - Create instance mask     │                 │
│               │                               │                 │
│               │  Output: instance_masks       │                 │
│               └───────────────┬───────────────┘                 │
│                               │                                 │
│                               ▼                                 │
│                        Final Result                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Independence** | Models don't depend on each other's results |
| **Concurrency** | True parallel execution (multi-thread or multi-NPU) |
| **Barrier** | Wait for all models before fusion |
| **Fusion** | Lua function combines all model outputs |
| **Timing** | Total ≈ max(T1, T2) + T_fusion |

### Timing Example

```
Instance Segmentation (Det ∥ Seg → Fusion):

  ┌─ Model 1 (Detection):  28 ms ─┐
  │                               │ Parallel
  └─ Model 2 (Segmentation): 35 ms ┘

  Barrier wait: max(28, 35) = 35 ms

  Fusion: 5 ms

  Total: 35 + 5 = 40 ms → 25 FPS
  (vs Serial: 28 + 35 + 5 = 68 ms → 15 FPS)
```

### Typical Applications

| Pipeline | Models | Fusion |
|----------|--------|--------|
| Instance Segmentation | Det ∥ Seg | Combine boxes + global mask |
| Model Ensemble | YOLO ∥ SSD | Merge/vote detections |
| Multi-Modal | RGB ∥ Depth | Feature concatenation |

---

## 2.5 Mode 4: Parallel-Async (Parallel Asynchronous)

### Description

Multiple models run independently at different frame rates. No synchronization between models. Results are aggregated based on frame ID.

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Parallel-Async Execution                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Video Stream (30 FPS input)                                    │
│                                                                 │
│  Time →                                                         │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Model 1: Detection (every frame, 30 FPS)                       │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐               │
│  │ F0 │ │ F1 │ │ F2 │ │ F3 │ │ F4 │ │ F5 │ │ F6 │ ...           │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘               │
│                                                                 │
│  Model 2: Scene Classification (every 10 frames, 3 FPS)         │
│  ┌────────────────────────────────┐                             │
│  │             F0                 │                             │
│  └────────────────────────────────┘                             │
│                    ┌────────────────────────────────┐           │
│                    │             F10                │           │
│                    └────────────────────────────────┘           │
│                                                                 │
│  Model 3: Anomaly Detection (every 30 frames, 1 FPS)            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                           F0                               │ │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Result Aggregation:                                            │
│                                                                 │
│  Frame 0:  det[0] + scene[0] + anomaly[0]                       │
│  Frame 1:  det[1] + scene[0] + anomaly[0]  (reuse old results)  │
│  Frame 2:  det[2] + scene[0] + anomaly[0]                       │
│  ...                                                            │
│  Frame 10: det[10] + scene[10] + anomaly[0]                     │
│  ...                                                            │
│  Frame 30: det[30] + scene[30] + anomaly[30]                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Independence** | Each model has its own processing loop |
| **Different Rates** | Models run at different frame intervals |
| **No Barrier** | Models don't wait for each other |
| **Result Reuse** | Slower model results are reused for multiple frames |
| **Aggregation** | Results combined by frame ID |

### Configuration

```
Model 1: Detection
  - frame_interval: 1    (every frame)
  - queue_size: 2

Model 2: Scene Classification
  - frame_interval: 10   (every 10th frame)
  - queue_size: 1

Model 3: Anomaly Detection
  - frame_interval: 30   (every 30th frame)
  - queue_size: 1
```

### Timing Example

```
30 FPS video input:

Model 1 (Detection, every frame):
  Inference: 30 ms → can sustain 30 FPS

Model 2 (Scene, every 10 frames):
  Inference: 100 ms → 10 FPS capacity, only need 3 FPS

Model 3 (Anomaly, every 30 frames):
  Inference: 500 ms → 2 FPS capacity, only need 1 FPS

Resource Usage:
  - Model 1: 30 ms / 33 ms = 91% utilization
  - Model 2: 100 ms / 333 ms = 30% utilization
  - Model 3: 500 ms / 1000 ms = 50% utilization
```

### Typical Applications

| Pipeline | Models | Rates |
|----------|--------|-------|
| Video Analytics | Det + Scene + Anomaly | 30 / 3 / 1 FPS |
| Surveillance | Det + Track + FaceRec | 30 / 30 / 5 FPS |
| Quality Inspection | Coarse + Fine + Defect | 60 / 30 / 10 FPS |

---

## 2.6 Mode Comparison Summary

| Aspect | Single | Serial | Parallel-Sync | Parallel-Async |
|--------|--------|--------|---------------|----------------|
| **Model Count** | 1 | 2-3 | 2-3 | 2-3 |
| **Dependency** | N/A | Yes | No | No |
| **Sync** | N/A | Implicit | Barrier | None |
| **Frame Rate** | Same | Same | Same | Different |
| **Total Time** | T | T1+T2+T3 | max(T1,T2,T3) | Independent |
| **Complexity** | Low | Medium | Medium | High |

---

## 2.7 Mode Selection Decision Tree

```
                        ┌─────────────────┐
                        │ How many models?│
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
           ┌─────┐           ┌─────┐           ┌─────┐
           │  1  │           │ 2-3 │           │ 4+  │
           └──┬──┘           └──┬──┘           └──┬──┘
              │                 │                 │
              ▼                 ▼                 ▼
          ┌────────┐    ┌────────────────┐   ┌────────────┐
          │ Single │    │ Does Model N   │   │ Decompose  │
          │ Mode   │    │ depend on      │   │ into stages│
          └────────┘    │ Model N-1?     │   └────────────┘
                        └───────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                ┌───────┐               ┌───────┐
                │  Yes  │               │  No   │
                └───┬───┘               └───┬───┘
                    │                       │
                    ▼                       ▼
                ┌────────┐          ┌────────────────┐
                │ Serial │          │ Same frame     │
                │ Mode   │          │ rate needed?   │
                └────────┘          └───────┬────────┘
                                            │
                                ┌───────────┴───────────┐
                                ▼                       ▼
                            ┌───────┐               ┌───────┐
                            │  Yes  │               │  No   │
                            └───┬───┘               └───┬───┘
                                │                       │
                                ▼                       ▼
                        ┌───────────────┐       ┌───────────────┐
                        │ Parallel-Sync │       │ Parallel-Async│
                        └───────────────┘       └───────────────┘
```

---

## 2.8 Why Limit to 3 Models?

### Coverage Analysis

| Model Count | Scenario Coverage | Examples |
|-------------|-------------------|----------|
| 1 | 60% | Detection, Classification, Segmentation |
| 2 | 35% | Det+Pose, Det+ReID, Det+Seg |
| 3 | 4% | Det+Pose+ReID, Det+Seg+Class, Vehicle+Plate+OCR |
| 4+ | <1% | Rare, can be decomposed |

### Benefits of Limiting to 3

1. **Resource Predictability**: Fixed memory budget for 3 sessions
2. **Simplified Scheduling**: No complex graph execution needed
3. **Reduced Complexity**: Fewer failure modes and edge cases
4. **Sufficient Coverage**: 99% of real-world scenarios covered

### Handling 4+ Model Scenarios

For rare cases needing 4+ models:
1. **Decompose**: Split into two pipelines, run sequentially
2. **Time-Multiplex**: Swap models as needed
3. **Model Fusion**: Combine multiple models into one
