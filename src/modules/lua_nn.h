#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include "onnxruntime_cxx_api.h"
#include "LuaIntf.h"
#include <vector>
#include <memory>

namespace lua_nn {

class Tensor {
public:
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape);
    
    // 属性访问（返回拷贝，安全）
    std::vector<int64_t> shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    size_t size() const { return data_->size(); }
    
    // 零拷贝视图（性能关键）
    LuaIntf::TensorView<float> view() {
        return LuaIntf::TensorView<float>(data_->data(), data_->size(), data_);
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
    std::shared_ptr<std::vector<float>> data_;  // shared_ptr管理数据
    std::vector<int64_t> shape_;
};

class Session {
public:
    explicit Session(const std::string& model_path);
    
    // 推理方法（接受Tensor对象）
    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);
    
    // 属性访问
    std::vector<std::string> input_names() const { return input_names_; }
    std::vector<std::string> output_names() const { return output_names_; }
    std::vector<ONNXTensorElementDataType> input_types() const { return input_types_; }
    std::vector<std::vector<int64_t>> input_shapes() const { return input_shapes_; }
    
private:
    std::shared_ptr<Ort::Env> env_;        // shared_ptr自动管理
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<ONNXTensorElementDataType> input_types_;
    std::vector<std::vector<int64_t>> input_shapes_;
};

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_nn

#endif
