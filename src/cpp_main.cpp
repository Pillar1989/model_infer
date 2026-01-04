#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
// Try to include float16 header if available, otherwise rely on CXX API
// Note: On some versions it is onnxruntime_float16.h
#if __has_include("onnxruntime_float16.h")
#include "onnxruntime_float16.h"
#endif

// Configuration matches yolov5_detector.lua
const int INPUT_W = 640;
const int INPUT_H = 640;
const float CONF_THRES = 0.25f;
const float IOU_THRES = 0.45f;
const int STRIDE = 32;

// COCO labels
const std::vector<std::string> LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

struct Detection {
    float x, y, w, h;
    float score;
    int class_id;
};

struct PreprocessResult {
    cv::Mat img_resized;
    std::vector<float> blob;
    float scale;
    int pad_w;
    int pad_h;
    int ori_w;
    int ori_h;
};

// IoU Calculation
float compute_iou(const Detection& a, const Detection& b) {
    float a_x1 = a.x, a_y1 = a.y, a_x2 = a.x + a.w, a_y2 = a.y + a.h;
    float b_x1 = b.x, b_y1 = b.y, b_x2 = b.x + b.w, b_y2 = b.y + b.h;
    
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

// NMS
std::vector<Detection> nms(std::vector<Detection>& boxes, float iou_thres) {
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<Detection> result;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (compute_iou(boxes[i], boxes[j]) > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

PreprocessResult preprocess(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) throw std::runtime_error("Failed to load image");
    
    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    int w = img.cols;
    int h = img.rows;
    
    float r = std::min((float)INPUT_H / h, (float)INPUT_W / w);
    int new_w = std::floor(w * r);
    int new_h = std::floor(h * r);
    
    cv::Mat resized;
    if (new_w != w || new_h != h) {
        cv::resize(img, resized, cv::Size(new_w, new_h));
    } else {
        resized = img.clone();
    }
    
    int dw = INPUT_W - new_w;
    int dh = INPUT_H - new_h;
    
    // Modulo stride
    dw = dw % STRIDE;
    dh = dh % STRIDE;
    
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // HWC -> CHW, Normalize
    std::vector<float> blob(3 * padded.rows * padded.cols);
    
    // Manual conversion to match lua_cv logic (scale=1/255, mean=0, std=1)
    // padded is CV_8UC3
    int H = padded.rows;
    int W = padded.cols;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                blob[c * H * W + i * W + j] = padded.at<cv::Vec3b>(i, j)[c] / 255.0f;
            }
        }
    }
    
    return {padded, blob, r, left, top, w, h};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.onnx> <image.jpg> [show]\n";
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    bool show_result = (argc > 3 && std::string(argv[3]) == "show");
    
    try {
        // 1. Preprocess
        auto start_pre = std::chrono::high_resolution_clock::now();
        PreprocessResult prep = preprocess(image_path);
        auto end_pre = std::chrono::high_resolution_clock::now();
        
        // 2. Inference
        auto start_infer = std::chrono::high_resolution_clock::now();
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpp_infer");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        Ort::Session session(env, model_path.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session.GetInputCount();
        std::vector<const char*> input_node_names;
        std::vector<std::string> input_node_names_alloc;
        for(size_t i = 0; i < num_input_nodes; i++){
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names_alloc.push_back(input_name.get());
            input_node_names.push_back(input_node_names_alloc.back().c_str());
        }
        
        // Output info
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> output_node_names;
        std::vector<std::string> output_node_names_alloc;
        for(size_t i = 0; i < num_output_nodes; i++){
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names_alloc.push_back(output_name.get());
            output_node_names.push_back(output_node_names_alloc.back().c_str());
        }
        
        // Check input shape and pad if necessary (Auto-padding logic from main.cpp)
        auto input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
        
        std::vector<float> input_tensor_values = prep.blob;
        // prep.img_resized is actually the padded image returned by preprocess
        std::vector<int64_t> input_shape = {1, 3, prep.img_resized.rows, prep.img_resized.cols};
        
        int64_t current_h = input_shape[2];
        int64_t current_w = input_shape[3];
        
        if (input_dims.size() >= 4) {
            int64_t model_h = input_dims[2];
            int64_t model_w = input_dims[3];
            
            if (model_h > 0 && model_w > 0) {
                if (current_h < model_h || current_w < model_w) {
                    // Create new larger buffer
                    std::vector<float> padded_blob(1 * 3 * model_h * model_w, 114.0f/255.0f);
                    
                    for (int c = 0; c < 3; ++c) {
                        for (int h = 0; h < current_h; ++h) {
                            // Copy row
                            const float* src = input_tensor_values.data() + c * current_h * current_w + h * current_w;
                            float* dst = padded_blob.data() + c * model_h * model_w + h * model_w;
                            std::copy(src, src + current_w, dst);
                        }
                    }
                    
                    input_tensor_values = padded_blob;
                    input_shape[2] = model_h;
                    input_shape[3] = model_w;
                }
            }
        }
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
        
        // Handle Float16 input
        std::vector<Ort::Float16_t> fp16_input_values;
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            fp16_input_values.reserve(input_tensor_values.size());
            for (float v : input_tensor_values) {
                fp16_input_values.emplace_back(v);
            }
            input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, fp16_input_values.data(), fp16_input_values.size(), input_shape.data(), input_shape.size());
        }

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        
        auto end_infer = std::chrono::high_resolution_clock::now();
        
        // 3. Postprocess
        auto start_post = std::chrono::high_resolution_clock::now();
        
        float* floatarr = nullptr;
        std::vector<float> float_output_buffer;
        
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_dims = output_info.GetShape();
        
        if (output_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
             const Ort::Float16_t* fp16_out = output_tensors[0].GetTensorData<Ort::Float16_t>();
             size_t count = output_info.GetElementCount();
             float_output_buffer.resize(count);
             for(size_t i=0; i<count; ++i) {
                 float_output_buffer[i] = fp16_out[i].ToFloat();
             }
             floatarr = float_output_buffer.data();
        } else {
             floatarr = output_tensors[0].GetTensorMutableData<float>();
        }
        
        // Handle [1, 25200, 85] vs [1, 85, 25200]
        int64_t dim1 = output_dims[1];
        int64_t dim2 = output_dims[2];
        bool transposed = (dim1 < dim2 && dim2 > 100);
        int64_t num_boxes = transposed ? dim2 : dim1;
        int64_t box_dim = transposed ? dim1 : dim2;
        
        std::vector<Detection> proposals;
        
        for (int i = 0; i < num_boxes; ++i) {
            float cx, cy, w, h, obj_conf;
            
            if (transposed) {
                cx = floatarr[0 * num_boxes + i];
                cy = floatarr[1 * num_boxes + i];
                w  = floatarr[2 * num_boxes + i];
                h  = floatarr[3 * num_boxes + i];
                obj_conf = floatarr[4 * num_boxes + i];
            } else {
                const float* row = floatarr + i * box_dim;
                cx = row[0];
                cy = row[1];
                w  = row[2];
                h  = row[3];
                obj_conf = row[4];
            }
            
            if (obj_conf < CONF_THRES) continue;
            
            float max_cls_conf = 0;
            int cls_id = 0;
            
            int cls_start = 5;
            int num_classes = box_dim - 5;
            
            if (transposed) {
                for (int c = 0; c < num_classes; ++c) {
                    float conf = floatarr[(cls_start + c) * num_boxes + i];
                    if (conf > max_cls_conf) {
                        max_cls_conf = conf;
                        cls_id = c;
                    }
                }
            } else {
                const float* row = floatarr + i * box_dim;
                for (int c = 0; c < num_classes; ++c) {
                    float conf = row[cls_start + c];
                    if (conf > max_cls_conf) {
                        max_cls_conf = conf;
                        cls_id = c;
                    }
                }
            }
            
            float score = obj_conf * max_cls_conf;
            if (score < CONF_THRES) continue;
            
            // Coordinate restoration
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;
            
            // Reverse letterbox
            x = (x - prep.pad_w) / prep.scale;
            y = (y - prep.pad_h) / prep.scale;
            w = w / prep.scale;
            h = h / prep.scale;
            
            proposals.push_back({x, y, w, h, score, cls_id});
        }
        
        std::vector<Detection> results = nms(proposals, IOU_THRES);
        
        auto end_post = std::chrono::high_resolution_clock::now();
        
        // Print timings
        std::cout << "Preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count() << " ms\n";
        std::cout << "Inference: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count() << " ms\n";
        std::cout << "Postprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_post - start_post).count() << " ms\n";
        
        // Print results
        std::cout << "\n=== Detection Results ===\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& det = results[i];
            std::string label = (det.class_id >= 0 && det.class_id < LABELS.size()) ? LABELS[det.class_id] : "unknown";
            std::cout << "Box " << (i+1) << ": " << label << " (" << det.x << ", " << det.y << ", " << det.w << ", " << det.h << ") conf=" << det.score << "\n";
        }
        std::cout << "Total: " << results.size() << " detections\n";
        
        if (show_result) {
            cv::Mat draw_img = cv::imread(image_path);
            for (const auto& det : results) {
                cv::rectangle(draw_img, cv::Rect(det.x, det.y, det.w, det.h), cv::Scalar(0, 255, 0), 2);
                std::string label = (det.class_id >= 0 && det.class_id < LABELS.size()) ? LABELS[det.class_id] : "unknown";
                std::string text = label + " " + std::to_string(det.score).substr(0, 4);
                cv::putText(draw_img, text, cv::Point(det.x, det.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            cv::imshow("CPP Detections", draw_img);
            std::cout << "Press any key to exit...\n";
            cv::waitKey(0);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
