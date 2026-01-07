/**
 * lua_nn.cpp - ONNX Runtime Session 绑定和 Lua 模块注册
 *
 * 这个文件仅包含:
 * - Session 类实现 (ONNX Runtime 推理)
 * - register_module() 函数 (Lua 绑定)
 *
 * Tensor 类已迁移到 tensor/tensor.h 和 tensor/tensor.cpp
 */

#include "lua_nn.h"
#include <algorithm>
#include <cstring>

namespace lua_nn {

// ========== Session 实现 ==========

Session::Session(const std::string& model_path)
    : env_(std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "model_infer")),
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

        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_types_.push_back(tensor_info.GetElementType());
        input_shapes_.push_back(tensor_info.GetShape());
    }

    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}

LuaIntf::LuaRef Session::run(lua_State* L, const Tensor& input_tensor) {
    // 创建ONNX Runtime输入Tensor
    auto shape = input_tensor.shape();
    std::vector<Ort::Value> input_tensors;

    // Check expected type (assuming single input or first input matches)
    ONNXTensorElementDataType target_type = input_types_.empty() ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : input_types_[0];

    // Check expected shape and pad if necessary
    std::vector<float> padded_data;
    const float* input_data_ptr = input_tensor.raw_data();
    size_t input_data_size = static_cast<size_t>(input_tensor.size());

    if (!input_shapes_.empty() && input_shapes_[0].size() == 4) {
        int64_t model_h = input_shapes_[0][2];
        int64_t model_w = input_shapes_[0][3];

        if (model_h > 0 && model_w > 0 && shape.size() == 4) {
            int64_t input_h = shape[2];
            int64_t input_w = shape[3];

            if (input_h < model_h || input_w < model_w) {
                // Need padding
                // Assuming NCHW layout
                int64_t N = shape[0];
                int64_t C = shape[1];

                // New shape
                shape[2] = model_h;
                shape[3] = model_w;

                size_t new_size = static_cast<size_t>(N * C * model_h * model_w);
                padded_data.resize(new_size, 114.0f/255.0f); // Pad with gray

                // Copy data
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        const float* src_ptr = input_data_ptr + (n * C + c) * input_h * input_w;
                        float* dst_ptr = padded_data.data() + (n * C + c) * model_h * model_w;

                        for (int64_t h = 0; h < input_h; ++h) {
                            std::copy(src_ptr + h * input_w, src_ptr + h * input_w + input_w, dst_ptr + h * model_w);
                        }
                    }
                }

                // Update pointer and size
                input_data_ptr = padded_data.data();
                input_data_size = padded_data.size();
            }
        }
    }

    // Keep data alive during Run
    std::vector<Ort::Float16_t> fp16_data;

    if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        size_t num_elements = input_data_size;
        const float* float_data = input_data_ptr;
        fp16_data.reserve(num_elements);

        for (size_t i = 0; i < num_elements; ++i) {
            fp16_data.emplace_back(float_data[i]);
        }

        input_tensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info_,
            fp16_data.data(),
            fp16_data.size(),
            shape.data(),
            shape.size()
        ));
    } else {
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data_ptr),
            input_data_size,
            shape.data(),
            shape.size()
        ));
    }

    // 执行推理
    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto& name : input_names_) input_names_cstr.push_back(name.c_str());
    for (const auto& name : output_names_) output_names_cstr.push_back(name.c_str());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );

    // 将输出转换为Lua table
    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);

    for (size_t i = 0; i < output_tensors.size(); ++i) {
        auto& ort_tensor = output_tensors[i];
        auto tensor_info = ort_tensor.GetTensorTypeAndShapeInfo();
        auto out_shape = tensor_info.GetShape();

        // 复制数据到shared_ptr管理的vector
        ONNXTensorElementDataType output_type = tensor_info.GetElementType();
        size_t element_count = tensor_info.GetElementCount();

        std::vector<float> result_vec;
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
             const Ort::Float16_t* ort_data = ort_tensor.GetTensorData<Ort::Float16_t>();
             result_vec.reserve(element_count);
             for(size_t k=0; k<element_count; ++k) {
                 result_vec.push_back(ort_data[k].ToFloat());
             }
        } else {
             const float* ort_data = ort_tensor.GetTensorData<float>();
             result_vec.assign(ort_data, ort_data + element_count);
        }

        Tensor tensor(std::move(result_vec), out_shape);
        outputs[output_names_[i]] = tensor;
    }

    return outputs;
}

// ========== Lua 模块注册 ==========

void register_module(lua_State* L) {
    using namespace LuaIntf;

    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor类绑定 (显式使用 tensor::Tensor)
            .beginClass<tensor::Tensor>("Tensor")
                .addStaticFunction("new", &tensor::Tensor::from_lua)

                // 属性
                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &tensor::Tensor::shape)
                .addFunction("strides", &tensor::Tensor::strides)
                .addFunction("size", static_cast<int64_t(Tensor::*)() const>(&Tensor::size))
                .addFunction("is_contiguous", &Tensor::is_contiguous)
                .addFunction("contiguous", &Tensor::contiguous)
                .addFunction("view", &Tensor::view)

                // Level 1: 基础形状操作
                .addFunction("slice", &tensor::Tensor::slice)
                .addFunction("select_dim", &tensor::Tensor::select_dim)
                .addFunction("get_column", &tensor::Tensor::get_column)
                .addFunction("slice_columns", &tensor::Tensor::slice_columns)
                .addFunction("reshape", &tensor::Tensor::reshape)
                .addFunction("transpose",
                    static_cast<tensor::Tensor(tensor::Tensor::*)() const>(&tensor::Tensor::transpose))
                .addFunction("transpose_dims",
                    static_cast<tensor::Tensor(tensor::Tensor::*)(const std::vector<int>&) const>(&tensor::Tensor::transpose))
                .addFunction("squeeze", &tensor::Tensor::squeeze)
                .addFunction("unsqueeze", &tensor::Tensor::unsqueeze)

                // Level 2: 数学运算
                .addFunction("add", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::add))
                .addFunction("add_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::add))
                .addFunction("sub", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::sub))
                .addFunction("sub_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::sub))
                .addFunction("mul", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::mul))
                .addFunction("mul_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::mul))
                .addFunction("div", static_cast<tensor::Tensor(tensor::Tensor::*)(float) const>(&tensor::Tensor::div))
                .addFunction("div_tensor", static_cast<tensor::Tensor(tensor::Tensor::*)(const tensor::Tensor&) const>(&tensor::Tensor::div))

                .addFunction("sum", &tensor::Tensor::sum)
                .addFunction("mean", &tensor::Tensor::mean)
                .addFunction("max", &tensor::Tensor::max)
                .addFunction("min", &tensor::Tensor::min)
                .addFunction("argmax", &tensor::Tensor::argmax_lua)
                .addFunction("argmin", &tensor::Tensor::argmin_lua)

                .addFunction("sigmoid", &tensor::Tensor::sigmoid)
                .addFunction("softmax", &tensor::Tensor::softmax)
                .addFunction("exp", &tensor::Tensor::exp_)
                .addFunction("log", &tensor::Tensor::log_)

                .addFunction("gt", &tensor::Tensor::gt)
                .addFunction("lt", &tensor::Tensor::lt)
                .addFunction("ge", &tensor::Tensor::ge)
                .addFunction("le", &tensor::Tensor::le)
                .addFunction("eq", &tensor::Tensor::eq)

                // Level 3: 高级操作
                .addFunction("topk_new", &tensor::Tensor::topk_lua)
                .addFunction("to_table", &tensor::Tensor::to_table)
                .addFunction("to_string", &tensor::Tensor::to_string)
                .addFunction("get", &tensor::Tensor::get_lua)
                .addFunction("set", &tensor::Tensor::set_lua)
                .addFunction("at", &tensor::Tensor::at2d)

                // 向量化过滤操作
                .addFunction("nonzero", &tensor::Tensor::nonzero)
                .addFunction("where_indices", &tensor::Tensor::where_indices)
                .addFunction("index_select", &tensor::Tensor::index_select)
                .addFunction("extract_columns", &tensor::Tensor::extract_columns_tensor)

                // Gather/Concat/Split
                .addFunction("gather", &tensor::Tensor::gather)
                .addStaticFunction("concat", &tensor::Tensor::concat)
                .addFunction("split", &tensor::Tensor::split)

                // Legacy方法（向后兼容）
                .addFunction("filter_yolo", &tensor::Tensor::filter_yolo)
                .addFunction("filter_yolo_pose", &tensor::Tensor::filter_yolo_pose)
                .addFunction("filter_yolo_seg", &tensor::Tensor::filter_yolo_seg)
                .addFunction("process_mask", &tensor::Tensor::process_mask)
                .addFunction("argmax_old", &tensor::Tensor::argmax)
                .addFunction("topk", &tensor::Tensor::topk)

                // Metamethods
                .addMetaFunction("__len", [](const tensor::Tensor* t) { return t->size(); })
                .addMetaFunction("__tostring", [](const tensor::Tensor* t) {
                    return t->to_string(10);
                })
                .addMetaFunction("__add", [](tensor::Tensor& t, float scalar) { return t.add(scalar); })
                .addMetaFunction("__sub", [](tensor::Tensor& t, float scalar) { return t.sub(scalar); })
                .addMetaFunction("__mul", [](tensor::Tensor& t, float scalar) { return t.mul(scalar); })
                .addMetaFunction("__div", [](tensor::Tensor& t, float scalar) { return t.div(scalar); })
            .endClass()

            // TensorView绑定
            .beginClass<TensorView<float>>("FloatView")
                .addFunction("get", &TensorView<float>::get)
                .addFunction("set", &TensorView<float>::set)
                .addMetaFunction("__len", [](const TensorView<float>* t) { return t->length(); })
            .endClass()

            // Session绑定
            .beginClass<Session>("Session")
                .addConstructor(
                    LUA_SP(std::shared_ptr<Session>),
                    LUA_ARGS(const std::string&)
                )
                .addFunction("run", &Session::run)
                .addProperty("input_names", &Session::input_names)
                .addProperty("output_names", &Session::output_names)
            .endClass()
        .endModule();
}

} // namespace lua_nn
