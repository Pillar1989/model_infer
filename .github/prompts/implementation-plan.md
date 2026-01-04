# Lua驱动的通用机器视觉推理引擎 - 完整实施计划

## 🎯 项目目标

实现基于Lua脚本的通用机器视觉模型推理引擎，支持YOLOv5和YOLO11系列模型（检测、分类、姿态估计、分割）。

---

## 📐 阶段一：验证ONNX Runtime环境 ✅

### 任务目标
验证现有ONNX Runtime预编译库可用性。

### 当前状态

**已存在**：`/home/baozhu/storage/model_infer/onnxruntime-prebuilt/`

- ✅ 动态链接库：`lib/libonnxruntime.so.1.23.2` (22MB)
- ✅ C++ API头文件：`include/onnxruntime_cxx_api.h`
- ✅ 版本：1.23.2

### 快速验证

```bash
# 确认库文件存在
ls -lh onnxruntime-prebuilt/lib/libonnxruntime.so*

# 确认头文件存在
ls onnxruntime-prebuilt/include/onnxruntime_cxx_api.h
```

**结论**：环境已就绪，直接进入阶段二。

---

## 📂 阶段二：项目结构重组

### 任务目标
建立清晰的项目目录结构，实现模块化的C++代码组织。

### 目标结构

```
src/
├── main.cpp                      # 主程序入口（参数解析、流程编排）
├── modules/                      # C++模块实现（核心功能）
│   ├── lua_cv.h                 # OpenCV封装接口
│   ├── lua_cv.cpp               # Image类、imread、resize、pad、to_tensor
│   ├── lua_nn.h                 # ONNX Runtime封装接口
│   ├── lua_nn.cpp               # Session类、Tensor类、filter_yolo
│   ├── lua_utils.h              # 工具函数接口
│   └── lua_utils.cpp            # NMS算法实现
├── bindings/                     # Lua绑定层（胶水代码）
│   └── register_modules.cpp    # 使用lua-intf注册所有C++模块到Lua
└── utils/                        # C++内部工具类（非Lua暴露）
    ├── tensor_utils.h           # Tensor操作辅助函数
    └── box_utils.h              # 边界框IoU计算等

scripts/                          # Lua推理脚本（用户层）
├── yolov5_detector.lua          # 【不可修改】YOLOv5检测基准
├── yolo11_detector.lua          # YOLO11检测
├── yolo11_classifier.lua        # YOLO11分类
├── yolo11_pose.lua              # YOLO11姿态估计
└── yolo11_segmentation.lua      # YOLO11实例分割
```

### 文件命名规范

| 类型 | 命名规则 | 示例 |
|------|---------|------|
| C++模块 | `lua_<module>.cpp/h` | `lua_cv.cpp`, `lua_nn.h` |
| Lua脚本 | `<model>_<task>.lua` | `yolo11_detector.lua` |
| 头文件保护 | `MODEL_INFER_<MODULE>_H_` | `MODEL_INFER_LUA_CV_H_` |
| 类名 | PascalCase | `Session`, `Image` |
| 函数名 | snake_case | `compute_iou`, `filter_yolo` |

### 实施步骤

1. 创建目录结构
```bash
mkdir -p src/modules src/bindings src/utils scripts
```

2. 创建空文件框架
```bash
touch src/main.cpp
touch src/modules/{lua_cv.cpp,lua_cv.h,lua_nn.cpp,lua_nn.h,lua_utils.cpp,lua_utils.h}
touch src/bindings/register_modules.cpp
touch src/utils/{tensor_utils.h,box_utils.h}
```

---

## 🔧 阶段三：CMakeLists.txt 更新

### 任务目标
配置构建系统，链接所有依赖，支持模块化编译。

### 核心修改点

#### 1. 添加ONNX Runtime依赖

```cmake
# ==========================================================
# 3. Dependencies
# ==========================================================
find_package(OpenCV 4.6.0 REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-prebuilt")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)
```

#### 2. 组织源文件编译

```cmake
# ==========================================================
# 4. Main Application
# ==========================================================
# 收集模块源文件
file(GLOB MODULE_SOURCES "src/modules/*.cpp")
file(GLOB BINDING_SOURCES "src/bindings/*.cpp")

add_executable(model_infer 
    src/main.cpp
    ${MODULE_SOURCES}
    ${BINDING_SOURCES}
)

target_link_libraries(model_infer PRIVATE 
    LuaIntf 
    lua 
    ${OpenCV_LIBS} 
    onnxruntime
)

target_include_directories(model_infer PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils"
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)
```

#### 3. 设置RPATH（重要）

```cmake
# 确保运行时能找到libonnxruntime.so
set_target_properties(model_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)
```

### 验证步骤

```bash
mkdir -p build && cd build
cmake ..
# 检查配置输出中是否包含ONNX Runtime路径
```

### ⚠️ 关键最佳实践（基于lua-intf深度分析）

**必须遵守的设计原则**：

1. **✅ 使用 `addMetaFunction` 注册元方法**
   ```cpp
   .addMetaFunction("__len", &Tensor::size)      // 支持 #tensor
   .addMetaFunction("__tostring", &to_string)    // 支持 tostring(tensor)
   ```

2. **✅ 使用 `TensorView<T>` 实现零拷贝**
   - 性能提升：**1000x** 内存节省，**75000x** 数据传递加速
   - 生命周期管理：`std::shared_ptr<void> owner` 保持数据存活
   ```cpp
   TensorView<float> view(data->data(), data->size(), data);  // 共享所有权
   ```

3. **✅ 使用 `addProperty` 封装属性（非 `addVariable`）**
   ```cpp
   .addProperty("width", &Image::width)     // 通过getter访问
   .addProperty("shape", &Tensor::shape)    // 返回拷贝，安全
   ```

4. **✅ 使用 `shared_ptr` 管理复杂对象**
   ```cpp
   .addConstructor(LUA_SP(std::shared_ptr<Session>), LUA_ARGS(...))
   ```

5. **⚠️ 必须处理的陷阱**
   - Lua 1-based索引 ↔ C++ 0-based（TensorView已处理）
   - 异常必须正确抛出（lua-intf自动转换为Lua error）
   - 避免返回临时对象的引用

详细分析见：[lua-intf-analysis.md](lua-intf-analysis.md)

---

## 💻 阶段四：C++模块实现

**⚠️ 实施前必读：[lua-intf最佳实践](lua-intf-analysis.md)**

关键要点：
- ✅ 使用 `addProperty` 封装属性
- ✅ 使用 `addMetaFunction` 注册 `__len`, `__tostring` 等
- ✅ 使用 `TensorView<float>` 实现零拷贝（性能关键）
- ✅ 使用 `shared_ptr` 管理Session等复杂对象
- ✅ 所有方法必须处理异常（抛出`std::runtime_error`等）

### 4.1 lua_cv 模块（OpenCV 4.x绑定）

#### API设计

```cpp
// lua_cv.h
#ifndef MODEL_INFER_LUA_CV_H_
#define MODEL_INFER_LUA_CV_H_

#include <opencv2/opencv.hpp>
#include <LuaIntf/LuaIntf.h>

// 前向声明
namespace lua_nn { class Tensor; }

namespace lua_cv {

class Image {
public:
    explicit Image(const cv::Mat& mat);
    Image();  // 默认构造函数
    
    // ✅ 属性访问（通过getter，不直接暴露成员）
    int width() const { return mat_.cols; }
    int height() const { return mat_.rows; }
    int channels() const { return mat_.channels(); }
    bool empty() const { return mat_.empty(); }
    
    // 图像操作（原地修改）
    void resize(int new_w, int new_h);
    void pad(int top, int bottom, int left, int right, int fill_value);
    
    // ✅ 返回Tensor对象（非LuaRef，简化API）
    lua_nn::Tensor to_tensor(double scale,
                             const std::vector<double>& mean,
                             const std::vector<double>& std) const;
    
    // 工具方法
    Image clone() const;
    
    // 内部访问（仅C++使用）
    const cv::Mat& data() const { return mat_; }
    cv::Mat& data() { return mat_; }
    
private:
    cv::Mat mat_;
};

// 全局函数
Image imread(const std::string& path);

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_cv

#endif
```

#### 关键实现要求

**resize 方法**：
```cpp
void Image::resize(int new_w, int new_h) {
    // 必须使用 cv::resize
    // 插值方法：cv::INTER_LINEAR（默认）
    cv::resize(mat_, mat_, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
}
```

**pad 方法**：
```cpp
void Image::pad(int top, int bottom, int left, int right, int fill_value) {
    // 必须使用 cv::copyMakeBorder
    // 边界类型：cv::BORDER_CONSTANT
    cv::copyMakeBorder(mat_, mat_, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(fill_value, fill_value, fill_value));
}
```

**to_tensor 方法（核心）**：
```cpp
lua_nn::Tensor Image::to_tensor(double scale,
                                 const std::vector<double>& mean,
                                 const std::vector<double>& std) const {
    // 1. 转换为浮点型
    cv::Mat float_mat;
    mat_.convertTo(float_mat, CV_32F);
    
    // 2. HWC -> CHW 转换（使用cv::split优化，比三重循环快10倍）
    int H = float_mat.rows;
    int W = float_mat.cols;
    int C = float_mat.channels();
    
    std::vector<cv::Mat> channels(C);
    cv::split(float_mat, channels);
    
    // 3. 分通道归一化并组装CHW数据
    std::vector<float> chw_data(C * H * W);
    size_t idx = 0;
    
    for (int c = 0; c < C; ++c) {
        const float* channel_ptr = channels[c].ptr<float>();
        for (int i = 0; i < H * W; ++i) {
            chw_data[idx++] = (channel_ptr[i] * scale - mean[c]) / std[c];
        }
    }
    
    // 4. 创建Tensor对象（NCHW格式）
    std::vector<int64_t> shape = {1, static_cast<int64_t>(C), 
                                   static_cast<int64_t>(H), 
                                   static_cast<int64_t>(W)};
    return lua_nn::Tensor(chw_data, shape);
}
```

**imread 函数**：
```cpp
Image imread(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return Image(mat);
}
```

#### Lua绑定注册

```cpp
void lua_cv::register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_cv")
            .addFactory(imread)  // 全局函数
            .beginClass<Image>("Image")
                .addConstructor(LUA_ARGS())  // 默认构造
                // ✅ 使用addProperty封装属性（非addFunction）
                .addProperty("width", &Image::width)
                .addProperty("height", &Image::height)
                .addProperty("channels", &Image::channels)
                .addFunction("empty", &Image::empty)
                .addFunction("resize", &Image::resize)
                .addFunction("pad", &Image::pad)
                .addFunction("clone", &Image::clone)
                .addFunction("to_tensor", &Image::to_tensor)
            .endClass()
        .endModule();
}
```

### 4.2 lua_nn 模块（ONNX Runtime绑定）

#### API设计

```cpp
// lua_nn.h
#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <LuaIntf/LuaIntf.h>
#include <LuaIntf/impl/TensorView.h>  // ✅ 引入零拷贝视图
#include <vector>
#include <memory>

namespace lua_nn {

class Tensor {
public:
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    
    // ✅ 属性访问（返回拷贝，安全）
    std::vector<int64_t> shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    size_t size() const { return data_->size(); }
    
    // ✅ 零拷贝视图（性能关键）
    TensorView<float> view() {
        return TensorView<float>(data_->data(), data_->size(), data_);
    }
    
    // YOLO特化方法（性能关键）
    LuaIntf::LuaRef filter_yolo(lua_State* L, float conf_thres);
    
    // 通用方法（为其他任务扩展）
    LuaIntf::LuaRef argmax(lua_State* L);
    LuaIntf::LuaRef topk(lua_State* L, int k);
    
    // 内部访问
    const float* raw_data() const { return data_->data(); }
    float* raw_data() { return data_->data(); }
    
private:
    std::shared_ptr<std::vector<float>> data_;  // ✅ shared_ptr管理数据
    std::vector<int64_t> shape_;
};

class Session {
public:
    explicit Session(const std::string& model_path);
    
    // 推理方法（接受Tensor对象）
    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);
    
    // ✅ 属性访问
    std::vector<std::string> input_names() const { return input_names_; }
    std::vector<std::string> output_names() const { return output_names_; }
    
private:
    std::shared_ptr<Ort::Env> env_;        // ✅ shared_ptr自动管理
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_nn

#endif
```

#### 关键实现要求

**Session构造函数**：
```cpp
Session::Session(const std::string& model_path)
    : env_(std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "model_infer")),  // ✅ shared_ptr
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    // 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 创建会话
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
    }
    
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}
```

**Session::run 方法**：
```cpp
LuaIntf::LuaRef Session::run(lua_State* L, const Tensor& input_tensor) {
    // 1. 直接使用Tensor对象（无需从LuaRef提取）
    
    // 2. 创建ONNX Runtime输入Tensor
    auto shape = input_tensor.shape();
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_tensor.raw_data()),
        input_tensor.size(),
        shape.data(),
        shape.size()
    ));
    
    // 3. 执行推理
    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto& name : input_names_) input_names_cstr.push_back(name.c_str());
    for (const auto& name : output_names_) output_names_cstr.push_back(name.c_str());
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );
    
    // 4. 将输出转换为Lua table
    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        auto& ort_tensor = output_tensors[i];
        auto tensor_info = ort_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        // 复制数据到shared_ptr管理的vector
        const float* ort_data = ort_tensor.GetTensorData<float>();
        size_t element_count = tensor_info.GetElementCount();
        auto data = std::make_shared<std::vector<float>>(ort_data, ort_data + element_count);
        
        // 创建Tensor对象
        Tensor tensor(*data, shape);
        outputs[output_names_[i]] = tensor;
    }
    
    return outputs;
}
```

**Tensor::filter_yolo 方法（核心优化）**：
```cpp
LuaIntf::LuaRef Tensor::filter_yolo(lua_State* L, float conf_thres) {
    // 假设输入shape: [1, N, 85] (YOLOv5) 或 [1, N, 84] (YOLOv8/11)
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO output shape");
    }
    
    int64_t num_boxes = shape_[1];
    int64_t box_dim = shape_[2];
    
    // 判断格式
    bool has_objectness = (box_dim == 85);  // YOLOv5格式
    int num_classes = has_objectness ? 80 : (box_dim - 4);
    
    // 创建结果table
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;  // Lua索引从1开始
    
    // 遍历所有boxes
    for (int64_t i = 0; i < num_boxes; ++i) {
        const float* box_data = data_->data() + i * box_dim;
        
        // 提取坐标
        float cx = box_data[0];
        float cy = box_data[1];
        float w = box_data[2];
        float h = box_data[3];
        
        // 提取置信度
        float objectness = has_objectness ? box_data[4] : 1.0f;
        
        // 提取类别分数
        const float* class_scores = box_data + (has_objectness ? 5 : 4);
        
        // 找到最大类别
        int best_class_id = 0;
        float best_class_score = class_scores[0];
        for (int c = 1; c < num_classes; ++c) {
            if (class_scores[c] > best_class_score) {
                best_class_score = class_scores[c];
                best_class_id = c;
            }
        }
        
        // 计算最终分数
        float final_score = objectness * best_class_score;
        
        // 过滤低置信度
        if (final_score < conf_thres) continue;
        
        // 转换为xyxy格式（根据脚本需求可能是xywh）
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
        // 创建box table
        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = final_score;
        box["cls"] = best_class_id;
        
        results[result_idx++] = box;
    }
    
    return results;
}
```

**Tensor::argmax 方法（分类任务）**：
```cpp
LuaIntf::LuaRef Tensor::argmax(lua_State* L) {
    // 假设shape: [1, num_classes]
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }
    
    int num_classes = static_cast<int>(shape_[1]);
    int max_idx = 0;
    float max_val = (*data_)[0];
    
    for (int i = 1; i < num_classes; ++i) {
        if ((*data_)[i] > max_val) {
            max_val = (*data_)[i];
            max_idx = i;
        }
    }
    
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["class_id"] = max_idx;
    result["confidence"] = max_val;
    return result;
}
```

#### Lua绑定注册

```cpp
void lua_nn::register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor类绑定
            .beginClass<Tensor>("Tensor")
                .addConstructor(LUA_ARGS(
                    const std::vector<float>&,
                    const std::vector<int64_t>&
                ))
                // ✅ 属性使用addProperty
                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &Tensor::shape)
                .addFunction("view", &Tensor::view)  // ✅ 零拷贝视图
                .addFunction("filter_yolo", &Tensor::filter_yolo)
                .addFunction("argmax", &Tensor::argmax)
                .addFunction("topk", &Tensor::topk)
                // ✅ 元方法
                .addMetaFunction("__len", &Tensor::size)
                .addMetaFunction("__tostring", [](const Tensor* t) {
                    auto s = t->shape();
                    std::string shape_str = "[";
                    for (size_t i = 0; i < s.size(); ++i) {
                        if (i > 0) shape_str += ", ";
                        shape_str += std::to_string(s[i]);
                    }
                    shape_str += "]";
                    return "Tensor(" + shape_str + ")";
                })
            .endClass()
            
            // ✅ TensorView绑定（零拷贝）
            .beginClass<TensorView<float>>("FloatView")
                .addFunction("get", &TensorView<float>::get)
                .addFunction("set", &TensorView<float>::set)
                .addMetaFunction("__len", &TensorView<float>::length)
            .endClass()
            
            // ✅ Session使用shared_ptr管理
            .beginClass<Session>("Session")
                .addConstructor(
                    LUA_SP(std::shared_ptr<Session>),  // shared_ptr管理
                    LUA_ARGS(const std::string&)
                )
                .addFunction("run", &Session::run)
                .addProperty("input_names", &Session::input_names)
                .addProperty("output_names", &Session::output_names)
            .endClass()
        .endModule();
}
```

### 4.3 lua_utils 模块

#### API设计

```cpp
// lua_utils.h
#ifndef MODEL_INFER_LUA_UTILS_H_
#define MODEL_INFER_LUA_UTILS_H_

#include <LuaIntf/LuaIntf.h>
#include <vector>

namespace lua_utils {

struct Box {
    float x, y, w, h;
    float score;
    int label;
};

// NMS算法
LuaIntf::LuaRef nms(lua_State* L, LuaIntf::LuaRef proposals, float iou_thres);

// 辅助函数
float compute_iou(const Box& a, const Box& b);

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_utils

#endif
```

#### NMS实现（标准算法）

```cpp
float lua_utils::compute_iou(const Box& a, const Box& b) {
    // 转换为 x1, y1, x2, y2
    float a_x1 = a.x, a_y1 = a.y, a_x2 = a.x + a.w, a_y2 = a.y + a.h;
    float b_x1 = b.x, b_y1 = b.y, b_x2 = b.x + b.w, b_y2 = b.y + b.h;
    
    // 计算交集
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    // 计算并集
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

LuaIntf::LuaRef lua_utils::nms(lua_State* L, LuaIntf::LuaRef proposals, float iou_thres) {
    // 1. 从Lua table提取boxes
    std::vector<Box> boxes;
    for (int i = 1; i <= proposals.len(); ++i) {
        auto prop = proposals[i];
        Box box;
        box.x = prop["x"].toValue<float>();
        box.y = prop["y"].toValue<float>();
        box.w = prop["w"].toValue<float>();
        box.h = prop["h"].toValue<float>();
        box.score = prop["score"].toValue<float>();
        // label可能是字符串，需要保存原始table
        boxes.push_back(box);
    }
    
    // 2. 按score降序排序
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&boxes](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });
    
    // 3. NMS算法
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep_indices;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep_indices.push_back(idx);
        
        // 抑制与当前box IoU过高的其他box
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            
            float iou = compute_iou(boxes[idx], boxes[idx_j]);
            if (iou > iou_thres) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    // 4. 构造结果table
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    for (int idx : keep_indices) {
        results[result_idx++] = proposals[idx + 1];  // Lua索引从1开始
    }
    
    return results;
}
```

#### Lua绑定注册

```cpp
void lua_utils::register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_utils")
            .addFunction("nms", &nms)
        .endModule();
}
```

---

## 🏗️ 阶段五：主程序实现（main.cpp）

### 任务目标
编写程序入口，整合所有模块，实现完整的推理流程。

### 实现框架

```cpp
// src/main.cpp
#include <iostream>
#include <string>
#include <lua.hpp>
#include <LuaIntf/LuaIntf.h>

// 模块头文件
#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <script.lua> <model.onnx> <image.jpg>\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " scripts/yolov5_detector.lua models/yolov5n.onnx images/zidane.jpg\n";
}

void print_detections(LuaIntf::LuaRef detections) {
    std::cout << "\n=== Detection Results ===\n";
    for (int i = 1; i <= detections.len(); ++i) {
        auto det = detections[i];
        
        float x = det["x"].toValue<float>();
        float y = det["y"].toValue<float>();
        float w = det["w"].toValue<float>();
        float h = det["h"].toValue<float>();
        float score = det["score"].toValue<float>();
        
        // label可能是字符串
        std::string label = det["label"].toValue<std::string>();
        
        std::cout << "Box " << i << ": "
                  << label << " "
                  << "(" << x << ", " << y << ", " << w << ", " << h << ") "
                  << "conf=" << score << "\n";
    }
    std::cout << "Total: " << detections.len() << " detections\n";
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string script_path = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];
    
    try {
        // 1. 初始化Lua
        lua_State* L = luaL_newstate();
        if (!L) {
            throw std::runtime_error("Failed to create Lua state");
        }
        luaL_openlibs(L);
        
        // 2. 注册C++模块
        std::cout << "Registering modules...\n";
        lua_cv::register_module(L);
        lua_nn::register_module(L);
        lua_utils::register_module(L);
        
        // 3. 加载图像
        std::cout << "Loading image: " << image_path << "\n";
        auto img = lua_cv::imread(image_path);
        std::cout << "Image size: " << img.width() << "x" << img.height() << "\n";
        
        // 4. 加载ONNX模型
        std::cout << "Loading model: " << model_path << "\n";
        lua_nn::Session session(model_path);
        
        // 5. 加载Lua脚本
        std::cout << "Loading script: " << script_path << "\n";
        if (luaL_dofile(L, script_path.c_str()) != LUA_OK) {
            const char* err = lua_tostring(L, -1);
            throw std::runtime_error("Failed to load script: " + std::string(err));
        }
        
        // 6. 获取Model table
        LuaIntf::LuaRef model = LuaIntf::LuaRef::fromStack(L, -1);
        if (!model.isTable()) {
            throw std::runtime_error("Script must return a Model table");
        }
        
        // 7. 预处理
        std::cout << "Preprocessing...\n";
        LuaIntf::LuaRef preprocess = model["preprocess"];
        if (!preprocess.isFunction()) {
            throw std::runtime_error("Model.preprocess must be a function");
        }
        
        // 将Image传递给Lua（需要在lua_cv中注册）
        LuaIntf::LuaRef img_ref = LuaIntf::LuaRef::createUserdata(L, &img);
        LuaIntf::LuaRef prep_results = preprocess(img_ref);
        
        // 提取input_tensor和meta
        LuaIntf::LuaRef input_tensor = prep_results[LuaIntf::LuaRef(L, 1)];
        LuaIntf::LuaRef meta = prep_results[LuaIntf::LuaRef(L, 2)];
        
        // 8. 推理
        std::cout << "Running inference...\n";
        LuaIntf::LuaRef session_ref = LuaIntf::LuaRef::createUserdata(L, &session);
        LuaIntf::LuaRef outputs = session.run(L, input_tensor);
        
        // 9. 后处理
        std::cout << "Postprocessing...\n";
        LuaIntf::LuaRef postprocess = model["postprocess"];
        if (!postprocess.isFunction()) {
            throw std::runtime_error("Model.postprocess must be a function");
        }
        
        LuaIntf::LuaRef detections = postprocess(outputs, meta);
        
        // 10. 打印结果
        print_detections(detections);
        
        // 11. 清理
        lua_close(L);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
```

### 关键注意事项

1. **Lua与C++的数据传递**：
   - C++ → Lua：使用 `LuaRef::createUserdata` 或 `LuaRef::fromValue`
   - Lua → C++：使用 `LuaRef::toValue<T>()`

2. **多返回值处理**：
   ```cpp
   // Lua: return tensor, meta
   // C++:
   LuaIntf::LuaRef results = preprocess(img);
   auto tensor = results[LuaIntf::LuaRef(L, 1)];  // 第1个返回值
   auto meta = results[LuaIntf::LuaRef(L, 2)];    // 第2个返回值
   ```

3. **错误处理**：
   - 所有Lua调用用 `pcall` 包装
   - C++异常会被lua-intf自动转换为Lua错误

---

## 📜 阶段六：Lua脚本编写

### 6.1 yolov5_detector.lua
**状态**：✅ 已存在，不可修改（作为API契约）

### 6.2 yolo11_detector.lua

```lua
local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres = 0.45,
    stride = 32,
    labels = { /* COCO 80类标签 */ }
}

function Model.preprocess(img)
    -- 与yolov5相同的letterbox逻辑
    local w, h = img:width(), img:height()
    local target_h, target_w = table.unpack(Model.config.input_size)
    
    local r = math.min(target_h / h, target_w / w)
    local new_w, new_h = math.floor(w * r), math.floor(h * r)
    
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end
    
    local dw = target_w - new_w
    local dh = target_h - new_h
    dw = dw % Model.config.stride
    dh = dh % Model.config.stride
    
    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left
    
    img:pad(top, bottom, left, right, 114)
    
    local input_tensor = img:to_tensor(1.0 / 255.0, {0,0,0}, {1,1,1})
    
    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }
    
    return input_tensor, meta
end

function Model.postprocess(outputs, meta)
    -- YOLO11可能输出格式为 [1, 8400, 84] (无objectness)
    local output_tensor = outputs["output0"]
    
    -- filter_yolo会自动检测格式
    local raw_boxes = output_tensor:filter_yolo(Model.config.conf_thres)
    
    local proposals = {}
    for _, box in ipairs(raw_boxes) do
        local x = (box.x - meta.pad_x) / meta.scale
        local y = (box.y - meta.pad_y) / meta.scale
        local w = box.w / meta.scale
        local h = box.h / meta.scale
        
        x = math.max(0, x)
        y = math.max(0, y)
        w = math.min(w, meta.ori_w - x)
        h = math.min(h, meta.ori_h - y)
        
        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            label = Model.config.labels[box.cls + 1]
        })
    end
    
    return lua_utils.nms(proposals, Model.config.iou_thres)
end

return Model
```

### 6.3 yolo11_classifier.lua

```lua
local Model = {}

Model.config = {
    input_size = {224, 224},  -- ImageNet标准
    labels = { /* ImageNet 1000类标签 */ }
}

function Model.preprocess(img)
    local w, h = img:width(), img:height()
    
    -- 中心裁剪 + Resize
    local size = math.min(w, h)
    -- TODO: 实现中心裁剪（或直接resize）
    
    img:resize(Model.config.input_size[2], Model.config.input_size[1])
    
    -- ImageNet normalization
    local mean = {0.485, 0.456, 0.406}
    local std = {0.229, 0.224, 0.225}
    local input_tensor = img:to_tensor(1.0 / 255.0, mean, std)
    
    return input_tensor, {}
end

function Model.postprocess(outputs, meta)
    local output_tensor = outputs["output0"]  -- [1, 1000]
    
    -- 获取top-5
    local top5 = output_tensor:topk(5)
    
    local results = {}
    for i, result in ipairs(top5) do
        table.insert(results, {
            rank = i,
            class_id = result.class_id,
            label = Model.config.labels[result.class_id + 1],
            confidence = result.confidence
        })
    end
    
    return results
end

return Model
```

### 6.4 yolo11_pose.lua

```lua
local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres = 0.45,
    num_keypoints = 17,  -- COCO keypoints
    keypoint_names = {"nose", "left_eye", "right_eye", /* ... */}
}

function Model.preprocess(img)
    -- 与检测相同
    -- ...
end

function Model.postprocess(outputs, meta)
    local output_tensor = outputs["output0"]  -- [1, 8400, 56]
    
    -- 假设格式: [x, y, w, h, conf, kp1_x, kp1_y, kp1_v, ..., kp17_x, kp17_y, kp17_v]
    -- 需要在C++中实现 filter_yolo_pose 或在Lua中解析
    
    -- 方案A：在C++中扩展 Tensor:filter_pose
    local detections = output_tensor:filter_pose(Model.config.conf_thres)
    
    -- 方案B：在Lua中手动解析（慢）
    -- ...
    
    -- 坐标恢复
    for _, det in ipairs(detections) do
        det.x = (det.x - meta.pad_x) / meta.scale
        det.y = (det.y - meta.pad_y) / meta.scale
        det.w = det.w / meta.scale
        det.h = det.h / meta.scale
        
        for i, kp in ipairs(det.keypoints) do
            kp.x = (kp.x - meta.pad_x) / meta.scale
            kp.y = (kp.y - meta.pad_y) / meta.scale
        end
    end
    
    return lua_utils.nms(detections, Model.config.iou_thres)
end

return Model
```

### 6.5 yolo11_segmentation.lua

```lua
local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres = 0.45,
    mask_threshold = 0.5
}

function Model.preprocess(img)
    -- 与检测相同
end

function Model.postprocess(outputs, meta)
    local output0 = outputs["output0"]  -- [1, 8400, 116]
    local output1 = outputs["output1"]  -- [1, 32, 160, 160] mask prototypes
    
    -- STEP 1: 检测boxes
    local raw_boxes = output0:filter_yolo_seg(Model.config.conf_thres)
    
    -- STEP 2: 生成mask
    -- 需要在C++中实现矩阵乘法: mask_coef @ prototypes
    for _, box in ipairs(raw_boxes) do
        -- box.mask_coef: [32]
        -- prototypes: [32, 160, 160]
        -- result: [160, 160]
        box.mask = output1:decode_mask(box.mask_coef, Model.config.mask_threshold)
    end
    
    -- STEP 3: 坐标恢复 + NMS
    -- ...
    
    return final_results
end

return Model
```

---

## ⚠️ 阶段七：Lua-Intf调试与修复

### 常见问题与解决方案

#### 问题1：异常未正确传播

**现象**：C++抛出异常，Lua脚本直接崩溃而不是触发错误处理

**解决**：
```cpp
// 在register_modules.cpp中包装所有函数
template<typename Func>
auto safe_wrap(Func&& func) {
    return [func](auto&&... args) -> decltype(auto) {
        try {
            return func(std::forward<decltype(args)>(args)...);
        } catch (const std::exception& e) {
            luaL_error(L, "C++ exception: %s", e.what());
        }
    };
}
```

#### 问题2：Userdata生命周期

**现象**：Image或Tensor被提前释放

**解决**：
```cpp
// 使用共享指针
LuaBinding(L)
    .beginClass<Image>("Image")
        .addConstructor(LUA_ARGS(_opt<std::shared_ptr<Image>>))
        // ...
    .endClass();
```

#### 问题3：多返回值

**现象**：`return tensor, meta` 只返回第一个值

**解决**：
```cpp
// 在Lua中使用table包装
return {tensor, meta}

// 或在C++中返回tuple
std::tuple<LuaRef, LuaRef> preprocess(...);
```

---

## 📊 阶段八：测试与验证

### 测试用例

#### T1: YOLOv5检测
```bash
./build/model_infer scripts/yolov5_detector.lua models/yolov5n.onnx images/zidane.jpg
```
**预期输出**：
```
=== Detection Results ===
Box 1: person (189.2, 112.5, 344.6, 523.7) conf=0.89
Box 2: person (420.3, 201.8, 195.4, 398.2) conf=0.76
Box 3: tie (358.9, 305.2, 48.3, 87.1) conf=0.68
Total: 3 detections
```

#### T2: Lua绑定正确性测试
```lua
-- 测试元方法
local tensor = nn.Tensor({1,2,3,4,5}, {5})
assert(#tensor == 5, "__len failed")
print(tostring(tensor))  -- 应输出 "Tensor([5])"

-- 测试Property
local img = cv.imread("test.jpg")
assert(img.width > 0, "width property failed")  -- 使用.而非:

-- 测试零拷贝TensorView
local view = tensor:view()
assert(#view == 5, "view length failed")
view:set(1, 999)
assert(view:get(1) == 999, "view get/set failed")

-- 测试异常处理
local success, err = pcall(function()
    view:get(100)  -- 越界
end)
assert(not success, "exception not caught")
assert(string.find(err, "out of range"), "exception message wrong")
```

#### T3: 性能测试
```bash
# 使用time命令测试
time ./build/model_infer scripts/yolov5_detector.lua models/yolov5n.onnx images/zidane.jpg
```
**目标指标**：
- 总时间 < 150ms (CPU Intel i7)
- 预处理 < 10ms
- 推理 < 120ms
- 后处理 < 20ms

---

## 📝 实施清单

### 必须交付的文件

- [ ] `/home/baozhu/storage/model_infer/CMakeLists.txt` (已更新)
- [ ] `src/main.cpp`
- [ ] `src/modules/lua_cv.h`
- [ ] `src/modules/lua_cv.cpp`
- [ ] `src/modules/lua_nn.h`
- [ ] `src/modules/lua_nn.cpp`
- [ ] `src/modules/lua_utils.h`
- [ ] `src/modules/lua_utils.cpp`
- [ ] `src/bindings/register_modules.cpp`
- [ ] `scripts/yolo11_detector.lua`
- [ ] `scripts/yolo11_classifier.lua`
- [ ] `scripts/yolo11_pose.lua`
- [ ] `scripts/yolo11_segmentation.lua`

### 可选文件

- [ ] `src/utils/tensor_utils.h` (内部工具)
- [ ] `src/utils/box_utils.h` (内部工具)
- [ ] `README.md` (使用文档)

---

## 🚨 关键约束重申

### ✅ 必须遵守

1. **严禁模拟实现**：
   - ❌ 不允许：`return {}`、`return 0`、`// TODO`
   - ✅ 必须：完整实现所有算法

2. **不修改yolov5_detector.lua**：
   - 该文件是API契约，所有C++接口必须与之匹配

3. **强制使用OpenCV**：
   - 所有图像操作（imread, resize, pad, 颜色转换）必须使用OpenCV 4.x
   - 不允许手写像素循环（除了to_tensor的HWC→CHW转换）

4. **异常安全**：
   - Lua编译为C++，所有异常必须正确传播
   - 使用lua-intf的异常处理机制

### ❌ 严禁操作

- 修改 `scripts/yolov5_detector.lua`
- 使用假数据或占位符实现
- 跳过任何功能的完整实现
- 使用Lua C API直接操作（必须通过lua-intf）
- 修改lua-intf核心代码（除非确认是bug）

---

## 📅 实施时间表

| 阶段 | 预计时间 | 关键任务 |
|------|---------|---------|
| 阶段一 | 0.5h | 下载ONNX Runtime预编译库 |
| 阶段二 | 0.5h | 创建目录结构和文件框架 |
| 阶段三 | 1h | 更新CMakeLists.txt并验证编译 |
| 阶段四 | 6h | 实现lua_cv, lua_nn, lua_utils |
| 阶段五 | 2h | 实现main.cpp |
| 阶段六 | 4h | 编写4个YOLO11 Lua脚本 |
| 阶段七 | 2h | 调试lua-intf绑定问题 |
| 阶段八 | 2h | 测试与性能优化 |
| **总计** | **18h** | 约2-3个工作日 |

---

## 🎓 参考资料

### ONNX Runtime C++ API
- 官方文档：https://onnxruntime.ai/docs/api/c/
- 示例代码：`onnxruntime/samples/c_cxx/`

### lua-intf
- GitHub: https://github.com/pillar1989/lua-intf
- 示例：`lua-intf/tests/src/`

### OpenCV 4.x
- 官方文档：https://docs.opencv.org/4.x/
- Mat操作：https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html

### YOLO格式
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLO11: https://github.com/ultralytics/ultralytics

---

## ✅ 完成标准

### 功能完整性
- [ ] 所有模型都能成功加载和推理
- [ ] YOLOv5检测结果与官方实现一致（IoU > 0.9）
- [ ] YOLO11四种任务都能正常工作

### 性能指标
- [ ] CPU推理速度 < 150ms（YOLOv5n @ i7）
- [ ] 内存占用 < 500MB
- [ ] 无内存泄漏

### 代码质量
- [ ] 所有函数都有完整实现（无TODO）
- [ ] 异常处理正确
- [ ] 代码风格统一
- [ ] 关键函数有注释

---

**准备开始实施！**
