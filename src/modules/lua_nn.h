#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include <vector>
#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "LuaIntf.h"

// 使用 tensor 模块的 Tensor 类
#include "tensor/tensor.h"

namespace lua_nn {

// 使用 tensor::Tensor 作为 lua_nn::Tensor
using Tensor = tensor::Tensor;

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
