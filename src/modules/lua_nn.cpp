#include "lua_nn.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace lua_nn {

// Tensor Implementation
Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    : data_(std::make_shared<std::vector<float>>(data)), shape_(shape) {}

Tensor::Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape)
    : data_(std::make_shared<std::vector<float>>(std::move(data))), shape_(shape) {}

LuaIntf::LuaRef Tensor::filter_yolo(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO output shape");
    }
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 84, 8400] (YOLOv8/11) vs [1, 25200, 85] (YOLOv5)
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    bool has_objectness = (box_dim == 85);
    int num_classes = has_objectness ? 80 : (box_dim - 4);
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, objectness;
        
        if (transposed) {
            // [1, C, N] -> stride is N (num_boxes)
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            objectness = has_objectness ? data_ptr[4 * num_boxes + i] : 1.0f;
        } else {
            // [1, N, C] -> stride is C (box_dim)
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
            objectness = has_objectness ? box_data[4] : 1.0f;
        }
        
        int best_class_id = 0;
        float best_class_score = -1.0f;
        
        int class_start = has_objectness ? 5 : 4;
        
        if (transposed) {
             if (has_objectness && objectness < conf_thres) continue;

             // Initialize with first class
             best_class_score = data_ptr[(class_start + 0) * num_boxes + i];
             best_class_id = 0;

             for (int c = 1; c < num_classes; ++c) {
                 float score = data_ptr[(class_start + c) * num_boxes + i];
                 if (score > best_class_score) {
                     best_class_score = score;
                     best_class_id = c;
                 }
             }
        } else {
             const float* box_data = data_ptr + i * box_dim;
             const float* class_scores = box_data + class_start;
             
             if (has_objectness && objectness < conf_thres) continue;

             best_class_score = class_scores[0];
             for (int c = 1; c < num_classes; ++c) {
                 if (class_scores[c] > best_class_score) {
                     best_class_score = class_scores[c];
                     best_class_id = c;
                 }
             }
        }
        
        float final_score = objectness * best_class_score;
        if (final_score < conf_thres) continue;
        
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
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

LuaIntf::LuaRef Tensor::topk(lua_State* L, int k) {
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }
    
    int num_classes = static_cast<int>(shape_[1]);
    if (k > num_classes) k = num_classes;
    
    std::vector<std::pair<float, int>> scores(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        scores[i] = {(*data_)[i], i};
    }
    
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });
    
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    for (int i = 0; i < k; ++i) {
        LuaIntf::LuaRef item = LuaIntf::LuaRef::createTable(L);
        item["class_id"] = scores[i].second;
        item["confidence"] = scores[i].first;
        result[i + 1] = item;
    }
    return result;
}

LuaIntf::LuaRef Tensor::filter_yolo_pose(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Pose output shape");
    }
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 56, 8400]
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, score;
        
        if (transposed) {
            // [1, C, N] -> stride is N (num_boxes)
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            score = data_ptr[4 * num_boxes + i];
        } else {
            // [1, N, C] -> stride is C (box_dim)
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
            score = box_data[4];
        }
        
        if (score < conf_thres) continue;
        
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = score;
        box["class_id"] = 0; // Pose usually has only 1 class (person)
        
        // Extract Keypoints
        LuaIntf::LuaRef kpts = LuaIntf::LuaRef::createTable(L);
        for (int k = 0; k < 17; ++k) {
            float kx, ky, kv;
            if (transposed) {
                kx = data_ptr[(5 + k * 3 + 0) * num_boxes + i];
                ky = data_ptr[(5 + k * 3 + 1) * num_boxes + i];
                kv = data_ptr[(5 + k * 3 + 2) * num_boxes + i];
            } else {
                const float* box_data = data_ptr + i * box_dim;
                kx = box_data[5 + k * 3 + 0];
                ky = box_data[5 + k * 3 + 1];
                kv = box_data[5 + k * 3 + 2];
            }
            
            LuaIntf::LuaRef kp = LuaIntf::LuaRef::createTable(L);
            kp["x"] = kx;
            kp["y"] = ky;
            kp["v"] = kv;
            kpts[k + 1] = kp;
        }
        box["keypoints"] = kpts;
        
        results[result_idx++] = box;
    }
    
    return results;
}

LuaIntf::LuaRef Tensor::filter_yolo_seg(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Seg output shape");
    }
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 116, 8400] (32 masks + 4 box + 80 classes)
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    // 4 box + 80 classes + 32 masks = 116
    int num_classes = 80; 
    int num_masks = 32;
    
    if (box_dim != (4 + num_classes + num_masks)) {
        // Try to infer
        num_masks = 32;
        num_classes = box_dim - 4 - num_masks;
    }
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h;
        
        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
        } else {
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
        }
        
        int best_class_id = 0;
        float best_class_score = -1.0f;
        
        int class_start = 4;
        
        if (transposed) {
             best_class_score = data_ptr[(class_start + 0) * num_boxes + i];
             best_class_id = 0;

             for (int c = 1; c < num_classes; ++c) {
                 float score = data_ptr[(class_start + c) * num_boxes + i];
                 if (score > best_class_score) {
                     best_class_score = score;
                     best_class_id = c;
                 }
             }
        } else {
             const float* box_data = data_ptr + i * box_dim;
             const float* class_scores = box_data + class_start;
             
             best_class_score = class_scores[0];
             for (int c = 1; c < num_classes; ++c) {
                 if (class_scores[c] > best_class_score) {
                     best_class_score = class_scores[c];
                     best_class_id = c;
                 }
             }
        }
        
        if (best_class_score < conf_thres) continue;
        
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;
        
        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = best_class_score;
        box["class_id"] = best_class_id;
        
        // Extract Mask Coefficients
        LuaIntf::LuaRef mask_coeffs = LuaIntf::LuaRef::createTable(L);
        int mask_start = 4 + num_classes;
        
        for (int m = 0; m < num_masks; ++m) {
            float val;
            if (transposed) {
                val = data_ptr[(mask_start + m) * num_boxes + i];
            } else {
                val = data_ptr[i * box_dim + mask_start + m];
            }
            mask_coeffs[m + 1] = val;
        }
        box["mask_coeffs"] = mask_coeffs;
        
        results[result_idx++] = box;
    }
    
    return results;
}

LuaIntf::LuaRef Tensor::process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs, 
                                   const LuaIntf::LuaRef& box, 
                                   int img_w, int img_h,
                                   int input_w, int input_h,
                                   int pad_x, int pad_y) {
    const Tensor& proto = *this;
    // proto: [1, 32, 160, 160]
    // mask_coeffs: [32]
    
    auto proto_shape = proto.shape();
    if (proto_shape.size() != 4 || proto_shape[1] != 32) {
        throw std::runtime_error("Invalid proto mask shape");
    }
    
    int mh = proto_shape[2];
    int mw = proto_shape[3];
    int num_masks = 32;
    
    // 1. Matrix Multiplication: Mask = Coeffs * Proto
    cv::Mat proto_mat(num_masks, mh * mw, CV_32F, (void*)proto.raw_data());
    cv::Mat coeffs_mat(1, num_masks, CV_32F);
    
    for (int i = 0; i < num_masks; ++i) {
        coeffs_mat.at<float>(0, i) = mask_coeffs.get<float>(i + 1);
    }
    
    cv::Mat mask_flat = coeffs_mat * proto_mat; // 1 x 25600
    cv::Mat mask = mask_flat.reshape(1, mh); // 160 x 160
    
    // 2. Sigmoid
    cv::exp(-mask, mask);
    mask = 1.0f / (1.0f + mask);
    
    // 3. Resize to Input Size (e.g. 640x640)
    cv::Mat mask_input;
    cv::resize(mask, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    
    // 4. Crop Padding (Remove letterbox padding)
    // ROI: [pad_x, pad_y, input_w - 2*pad_x, input_h - 2*pad_y]
    // Ensure ROI is within bounds
    int roi_w = input_w - 2 * pad_x;
    int roi_h = input_h - 2 * pad_y;
    
    cv::Rect roi(pad_x, pad_y, roi_w, roi_h);
    roi = roi & cv::Rect(0, 0, input_w, input_h);
    
    if (roi.area() == 0) {
         return LuaIntf::LuaRef::fromValue(L, Tensor(std::vector<float>(img_w * img_h, 0), {1, (int64_t)img_h, (int64_t)img_w}));
    }
    
    cv::Mat mask_cropped = mask_input(roi);
    
    // 5. Resize to Original Image Size
    cv::Mat mask_original;
    cv::resize(mask_cropped, mask_original, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
    
    // 6. Crop by Box (set pixels outside box to 0)
    float bx = box.get<float>("x");
    float by = box.get<float>("y");
    float bw = box.get<float>("w");
    float bh = box.get<float>("h");
    
    cv::Rect box_rect(bx, by, bw, bh);
    box_rect = box_rect & cv::Rect(0, 0, img_w, img_h);
    
    cv::Mat final_mask = cv::Mat::zeros(img_h, img_w, CV_32F);
    if (box_rect.area() > 0) {
        mask_original(box_rect).copyTo(final_mask(box_rect));
    }
    
    // 7. Threshold (> 0.5)
    cv::threshold(final_mask, final_mask, 0.5, 1.0, cv::THRESH_BINARY);
    
    // Return as Tensor
    std::vector<float> mask_data(img_w * img_h);
    std::memcpy(mask_data.data(), final_mask.data, img_w * img_h * sizeof(float));
    
    return LuaIntf::LuaRef::fromValue(L, Tensor(std::move(mask_data), {1, (int64_t)img_h, (int64_t)img_w}));
}

// Session Implementation
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
    size_t input_data_size = input_tensor.size();
    
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
                
                size_t new_size = N * C * model_h * model_w;
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
        auto shape = tensor_info.GetShape();
        
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
        
        Tensor tensor(std::move(result_vec), shape);
        outputs[output_names_[i]] = tensor;
    }
    
    return outputs;
}

void register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor类绑定
            .beginClass<Tensor>("Tensor")
                .addConstructor(LUA_ARGS(
                    const std::vector<float>&,
                    const std::vector<int64_t>&
                ))
                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &Tensor::shape)
                .addFunction("view", &Tensor::view)
                .addFunction("filter_yolo", &Tensor::filter_yolo)
                .addFunction("filter_yolo_pose", &Tensor::filter_yolo_pose)
                .addFunction("filter_yolo_seg", &Tensor::filter_yolo_seg)
                .addFunction("process_mask", &Tensor::process_mask)
                .addFunction("argmax", &Tensor::argmax)
                .addFunction("topk", &Tensor::topk)
                .addMetaFunction("__len", [](const Tensor* t) { return t->size(); })
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
