# Tensor Redesign Plan - 通用化设计方案

## 📋 问题分析

### 当前问题
1. **硬编码后处理**: `filter_yolo`, `filter_yolo_pose`, `filter_yolo_seg` 专为 Ultralytics YOLO 模型设计
2. **缺乏通用性**: 无法支持其他视觉模型（DINO, SAM, ViT, DETR, RT-DETR, 传统检测器等）
3. **扩展性差**: 添加新模型需要修改 C++ 代码并重新编译
4. **Lua灵活性不足**: Lua层缺少直接操作tensor的能力（切片、索引、reshape等）

### 设计目标
1. **通用性**: 支持任意视觉模型的后处理
2. **高性能**: C++ 实现核心操作（零拷贝、SIMD优化）
3. **灵活性**: Lua层可以灵活组合操作，快速适配新模型
4. **向后兼容**: 保留现有YOLO特化函数作为convenience方法

---

## 🎯 核心设计思路

### 1. 分层架构
```
┌─────────────────────────────────────────────────┐
│          Lua Layer (灵活的后处理逻辑)            │
│  - 模型特定的后处理脚本                          │
│  - 灵活组合基础操作                              │
│  - 快速迭代，无需重新编译                        │
└─────────────────────────────────────────────────┘
                      ↓↑ (LuaIntf)
┌─────────────────────────────────────────────────┐
│     C++ Tensor Layer (高性能基础操作)            │
│  - 通用tensor操作 (slice, reshape, transpose)   │
│  - 数学运算 (element-wise, reduction)           │
│  - 零拷贝视图 (view, subview)                   │
│  - SIMD优化 (argmax, softmax, etc.)            │
└─────────────────────────────────────────────────┘
```

### 2. 操作分类

#### Level 1: 基础形状操作 (Essential)
- **索引/切片**: `tensor[{0, slice(4, 84)}]` → 提取特定维度
- **Reshape**: `tensor:reshape({1, 84, 8400})` → 改变形状（零拷贝）
- **Transpose**: `tensor:transpose({0, 2, 1})` → 维度置换
- **View**: `tensor:view(start, end)` → 创建子视图（零拷贝）
- **Squeeze/Unsqueeze**: 增减维度

#### Level 2: 数学运算 (Performance-Critical)
- **Element-wise**: `add, sub, mul, div, max, min` 
- **Reduction**: `sum, mean, max, min, argmax, argmin` (指定axis)
- **Activation**: `sigmoid, softmax, exp, log`
- **比较**: `gt, lt, ge, le, eq` (返回mask)

#### Level 3: 高级操作 (Convenience)
- **NMS**: 通用的NMS算法（IoU计算）
- **Gather**: 根据索引收集元素
- **Concat/Split**: 拼接/分割tensor
- **TopK**: 返回前K个元素

#### Level 4: 专用函数 (Optional Legacy)
- 保留现有的 `filter_yolo`, `filter_yolo_pose` 等作为快捷方法
- 标记为 "convenience methods"，建议用户使用通用操作

---

## 🔧 API 设计

### C++ Tensor API

```cpp
class Tensor {
public:
    // ========== 构造/属性 ==========
    Tensor(std::vector<float>&& data, std::vector<int64_t> shape);
    Tensor(const float* data, size_t size, std::vector<int64_t> shape); // 零拷贝构造
    
    std::vector<int64_t> shape() const;
    int64_t ndim() const;
    int64_t size() const;
    int64_t size(int dim) const; // 特定维度大小
    
    // ========== Level 1: 形状操作 ==========
    // 切片 (支持负索引，支持省略)
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1);
    Tensor slice_multi(const std::vector<SliceSpec>& specs); // 多维切片
    
    // Reshape (零拷贝，仅改变shape_)
    Tensor reshape(const std::vector<int64_t>& new_shape);
    
    // Transpose (会产生数据重排，除非是简单转置可优化)
    Tensor transpose(const std::vector<int>& dims);
    Tensor transpose(); // 默认反转所有维度
    
    // View (子视图，零拷贝)
    Tensor view(int64_t offset, int64_t length);
    
    // Squeeze/Unsqueeze
    Tensor squeeze(int dim = -1);
    Tensor unsqueeze(int dim);
    
    // ========== Level 2: 数学运算 ==========
    // Element-wise (支持broadcasting)
    Tensor add(const Tensor& other);
    Tensor add(float scalar);
    Tensor sub(const Tensor& other);
    Tensor mul(const Tensor& other);
    Tensor div(const Tensor& other);
    
    // Reduction (axis=-1表示所有维度)
    Tensor sum(int axis = -1, bool keepdims = false);
    Tensor mean(int axis = -1, bool keepdims = false);
    Tensor max(int axis = -1, bool keepdims = false);
    Tensor min(int axis = -1, bool keepdims = false);
    
    // Argmax/Argmin (返回索引tensor，int64类型)
    LuaIntf::LuaRef argmax_lua(lua_State* L, int axis = -1); // 返回table或单值
    LuaIntf::LuaRef argmin_lua(lua_State* L, int axis = -1);
    
    // Activation
    Tensor sigmoid();
    Tensor softmax(int axis = -1);
    Tensor exp();
    Tensor log();
    
    // 比较 (返回bool类型的mask tensor)
    Tensor gt(float threshold);
    Tensor lt(float threshold);
    Tensor ge(float threshold);
    Tensor le(float threshold);
    
    // ========== Level 3: 高级操作 ==========
    // TopK (返回 {values, indices} 的Lua table)
    LuaIntf::LuaRef topk(lua_State* L, int k, int axis = -1, bool largest = true);
    
    // Gather (根据索引收集元素)
    Tensor gather(int axis, const Tensor& indices);
    
    // Concat/Split
    static Tensor concat(const std::vector<Tensor>& tensors, int axis);
    std::vector<Tensor> split(int num_splits, int axis);
    
    // ========== Level 4: 辅助方法 ==========
    // 直接数据访问 (for Lua)
    float get_item(const std::vector<int64_t>& indices);
    void set_item(const std::vector<int64_t>& indices, float value);
    
    // 转换为Lua table (小tensor用，调试用)
    LuaIntf::LuaRef to_table(lua_State* L);
    
    // 打印 (调试用)
    std::string to_string(int max_elements = 10);
    
    // ========== Legacy 方法 (标记为可选) ==========
    LuaIntf::LuaRef filter_yolo(lua_State* L, float conf_thres);
    // ... 其他YOLO特化方法
    
    // ========== 内部API ==========
    const float* data() const;
    float* data();
    
private:
    std::shared_ptr<std::vector<float>> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_; // 新增：支持非连续tensor
    int64_t offset_; // 新增：支持零拷贝切片
    bool contiguous_; // 新增：标记是否连续
    
    // 内部辅助
    Tensor contiguous() const; // 转换为连续tensor
    int64_t compute_offset(const std::vector<int64_t>& indices) const;
};
```

### Lua API 使用示例

#### 示例 1: YOLOv8 目标检测 (用通用操作实现)
```lua
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]  -- [1, 84, 8400]
    
    -- 1. 切片提取不同部分
    local boxes = output:slice(1, 0, 4)        -- [1, 4, 8400] (cx,cy,w,h)
    local scores = output:slice(1, 4, 84)      -- [1, 80, 8400] (class scores)
    
    -- 2. 转置为 [8400, 4] 和 [8400, 80]
    boxes = boxes:squeeze(0):transpose()       -- [8400, 4]
    scores = scores:squeeze(0):transpose()     -- [8400, 80]
    
    -- 3. 找到每个box的最大类别
    local max_scores, class_ids = scores:max(1)  -- [8400], [8400]
    
    -- 4. 过滤低置信度
    local mask = max_scores:ge(Model.config.conf_thres)  -- [8400] bool mask
    local filtered_boxes = boxes:gather(0, mask)
    local filtered_scores = max_scores:gather(0, mask)
    local filtered_classes = class_ids:gather(0, mask)
    
    -- 5. NMS
    local keep_indices = utils.nms(filtered_boxes, filtered_scores, Model.config.iou_thres)
    
    -- 6. 构造结果
    local results = {}
    for i, idx in ipairs(keep_indices) do
        local box = filtered_boxes[idx]
        table.insert(results, {
            x = box[0] - box[2]/2,
            y = box[1] - box[3]/2,
            w = box[2],
            h = box[3],
            score = filtered_scores[idx],
            class_id = filtered_classes[idx],
            label = Model.config.labels[filtered_classes[idx] + 1]
        })
    end
    
    return results
end
```

#### 示例 2: 分类模型 (ResNet/ViT)
```lua
function ClassificationModel.postprocess(outputs)
    local logits = outputs["output"]  -- [1, 1000]
    
    -- Softmax
    local probs = logits:softmax(1)
    
    -- TopK
    local top5 = probs:topk(5)
    
    local results = {}
    for i = 1, 5 do
        table.insert(results, {
            class_id = top5.indices[i],
            label = IMAGENET_LABELS[top5.indices[i] + 1],
            confidence = top5.values[i]
        })
    end
    
    return results
end
```

#### 示例 3: Segmentation 模型 (SAM/SegFormer)
```lua
function SegmentationModel.postprocess(outputs, meta)
    local logits = outputs["logits"]  -- [1, num_classes, H, W]
    
    -- Argmax获取类别
    local pred_classes = logits:argmax(1)  -- [1, H, W]
    
    -- 调整到原始图像大小
    pred_classes = pred_classes:squeeze(0)  -- [H, W]
    local resized = cv.resize_nearest(pred_classes, meta.orig_w, meta.orig_h)
    
    return resized:to_table()  -- 转换为Lua table返回
end
```

#### 示例 4: 保持向后兼容
```lua
-- 方式1: 使用legacy方法 (快速，但不通用)
local results = output:filter_yolo(0.25)

-- 方式2: 使用通用操作 (灵活，推荐)
local results = Model.postprocess_generic(output)
```

---

## 📐 实现细节

### 1. 零拷贝设计
```cpp
// 内部数据结构
struct TensorImpl {
    std::shared_ptr<std::vector<float>> data;  // 共享底层数据
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // 步长，支持非连续
    int64_t offset;                // 起始偏移
    bool contiguous;               // 是否连续
};

// 切片示例 (零拷贝)
Tensor Tensor::slice(int dim, int64_t start, int64_t end) {
    Tensor result;
    result.data_ = this->data_;  // 共享数据指针
    result.shape_ = compute_new_shape(dim, start, end);
    result.strides_ = this->strides_;
    result.offset_ = this->offset_ + start * strides_[dim];
    result.contiguous_ = (dim == shape_.size() - 1);  // 最后维度切片仍连续
    return result;
}
```

### 2. SIMD 优化
```cpp
// 使用OpenCV优化的操作
Tensor Tensor::sigmoid() {
    Tensor result(shape_);
    cv::Mat src(1, size(), CV_32F, (void*)data());
    cv::Mat dst(1, size(), CV_32F, result.data());
    
    cv::exp(-src, dst);
    dst = 1.0f / (1.0f + dst);
    
    return result;
}

// Argmax优化 (SIMD)
int64_t Tensor::argmax_impl(const float* data, int64_t size) {
    // 使用SSE/AVX加速
    #ifdef USE_AVX
        // AVX实现
    #else
        // 标量实现
        return std::max_element(data, data + size) - data;
    #endif
}
```

### 3. Lua绑定
```cpp
void register_module(lua_State* L) {
    LuaBinding(L)
        .beginModule("lua_nn")
            .beginClass<Tensor>("Tensor")
                // 构造
                .addConstructor(...)
                
                // Level 1: 形状操作
                .addFunction("slice", &Tensor::slice)
                .addFunction("reshape", &Tensor::reshape)
                .addFunction("transpose", 
                    static_cast<Tensor(Tensor::*)()>(&Tensor::transpose))
                
                // Level 2: 数学运算
                .addFunction("add", static_cast<Tensor(Tensor::*)(float)>(&Tensor::add))
                .addFunction("sum", &Tensor::sum)
                .addFunction("argmax", &Tensor::argmax_lua)
                .addFunction("sigmoid", &Tensor::sigmoid)
                .addFunction("softmax", &Tensor::softmax)
                .addFunction("gt", &Tensor::gt)
                
                // Level 3: 高级操作
                .addFunction("topk", &Tensor::topk)
                .addFunction("gather", &Tensor::gather)
                
                // 访问/调试
                .addFunction("get", &Tensor::get_item)
                .addFunction("set", &Tensor::set_item)
                .addFunction("to_table", &Tensor::to_table)
                
                // Metamethods
                .addMetaFunction("__index", &Tensor::lua_index)
                .addMetaFunction("__newindex", &Tensor::lua_newindex)
                .addMetaFunction("__tostring", &Tensor::to_string)
                .addMetaFunction("__add", &Tensor::add)
                .addMetaFunction("__sub", &Tensor::sub)
                .addMetaFunction("__mul", &Tensor::mul)
            .endClass()
        .endModule();
}
```

---

## 🚀 迁移策略

### Phase 1: 基础操作 (Week 1)
- [ ] 实现 `slice`, `reshape`, `transpose`
- [ ] 实现 `strides` 和零拷贝机制
- [ ] 添加基础测试

### Phase 2: 数学运算 (Week 2)
- [ ] 实现 element-wise 操作 (`add`, `mul`, 等)
- [ ] 实现 reduction 操作 (`sum`, `mean`, `argmax`)
- [ ] 实现 activation 函数 (`sigmoid`, `softmax`)
- [ ] SIMD优化关键路径

### Phase 3: 高级操作 (Week 3)
- [ ] 实现 `topk`, `gather`, `concat`
- [ ] 通用NMS算法 (移至 `lua_utils`)
- [ ] Broadcasting支持

### Phase 4: 集成与测试 (Week 4)
- [ ] 用通用操作重写YOLO后处理脚本
- [ ] 添加新模型示例 (ResNet分类, SegFormer分割)
- [ ] 性能对比 (通用操作 vs 特化函数)
- [ ] 文档更新

### 向后兼容
- 保留 `filter_yolo` 等函数，但标记为 **deprecated**
- 在文档中推荐使用通用操作
- 提供迁移指南

---

## 📊 性能考虑

### 优化策略
1. **零拷贝**: 使用 `shared_ptr` 和 `strides` 实现
2. **OpenCV加速**: 利用OpenCV的SIMD优化数学函数
3. **延迟求值**: 简单操作（如reshape）仅改变元数据
4. **缓存友好**: 连续内存访问模式
5. **并行化**: 大tensor使用OpenMP并行

### 性能目标
- 切片/reshape: < 1μs (零拷贝)
- Argmax (8400元素): < 10μs (SIMD)
- Softmax (1000元素): < 20μs (OpenCV)
- Transpose (8400x84): < 100μs (缓存优化)

---

## 🎓 使用场景扩展

### 支持的模型类型
1. **目标检测**: YOLO系列, DETR, RT-DETR, Faster R-CNN
2. **分类**: ResNet, ViT, ConvNeXt, EfficientNet
3. **分割**: SAM, SegFormer, DeepLab, Mask R-CNN
4. **姿态估计**: HRNet, MediaPipe, MMPose
5. **关键点检测**: SuperPoint, DISK
6. **深度估计**: MiDaS, DPT

### 示例：支持 RT-DETR
```lua
-- RT-DETR 输出: {boxes: [1, 300, 4], scores: [1, 300, 80]}
function RTDETR.postprocess(outputs, meta)
    local boxes = outputs["boxes"]:squeeze(0)    -- [300, 4]
    local scores = outputs["scores"]:squeeze(0)  -- [300, 80]
    
    -- 找到最大类别和分数
    local max_scores, class_ids = scores:max(1)  -- [300]
    
    -- 过滤
    local mask = max_scores:ge(0.3)
    local filtered_boxes = boxes:gather(0, mask)
    local filtered_scores = max_scores:gather(0, mask)
    local filtered_classes = class_ids:gather(0, mask)
    
    -- 转换坐标格式 (cxcywh -> xyxy)
    local x1 = filtered_boxes:slice(1, 0, 1) - filtered_boxes:slice(1, 2, 3) / 2
    local y1 = filtered_boxes:slice(1, 1, 2) - filtered_boxes:slice(1, 3, 4) / 2
    local x2 = filtered_boxes:slice(1, 0, 1) + filtered_boxes:slice(1, 2, 3) / 2
    local y2 = filtered_boxes:slice(1, 1, 2) + filtered_boxes:slice(1, 3, 4) / 2
    
    return build_results(x1, y1, x2, y2, filtered_scores, filtered_classes)
end
```

---

## ✅ 总结

### 优势
1. **通用性**: 一套API支持所有视觉模型
2. **灵活性**: Lua层快速迭代，无需重新编译
3. **高性能**: C++实现，零拷贝，SIMD优化
4. **可维护性**: 代码结构清晰，易于扩展
5. **向后兼容**: 不破坏现有代码

### 开发优先级
1. **P0**: 基础形状操作 (slice, reshape, transpose) - 解锁基本能力
2. **P1**: 数学运算 (argmax, softmax, gt) - 支持大部分模型
3. **P2**: 高级操作 (topk, gather) - 提升易用性
4. **P3**: 性能优化 (SIMD, 并行) - 提升性能
5. **P4**: 特化函数迁移 - 清理技术债

### 下一步
1. Review设计方案
2. 创建新的 `lua_nn.h` 头文件
3. 实现 Phase 1 基础操作
4. 编写单元测试
5. 更新文档和示例
