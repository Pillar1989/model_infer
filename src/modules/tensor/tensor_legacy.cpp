#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <vector>

#include "LuaIntf.h"
#include <opencv2/opencv.hpp>

namespace tensor {

// ========== Legacy 方法 ==========

LuaIntf::LuaRef Tensor::filter_yolo(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO output shape");
    }

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    // Heuristic for [1, 84, 8400] (YOLOv8/11) vs [1, 25200, 85] (YOLOv5)
    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    bool has_objectness = (box_dim == 85);
    int num_classes = has_objectness ? 80 : (static_cast<int>(box_dim) - 4);

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, objectness;

        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            objectness = has_objectness ? data_ptr[4 * num_boxes + i] : 1.0f;
        } else {
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

LuaIntf::LuaRef Tensor::filter_yolo_pose(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Pose output shape");
    }

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, score;

        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            score = data_ptr[4 * num_boxes + i];
        } else {
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
        box["class_id"] = 0;

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

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    int num_classes = 80;
    int num_masks = 32;

    if (box_dim != (4 + num_classes + num_masks)) {
        num_masks = 32;
        num_classes = static_cast<int>(box_dim) - 4 - num_masks;
    }

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

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
        LuaIntf::LuaRef mask_coeffs_out = LuaIntf::LuaRef::createTable(L);
        int mask_start = 4 + num_classes;

        for (int m = 0; m < num_masks; ++m) {
            float val;
            if (transposed) {
                val = data_ptr[(mask_start + m) * num_boxes + i];
            } else {
                val = data_ptr[i * box_dim + mask_start + m];
            }
            mask_coeffs_out[m + 1] = val;
        }
        box["mask_coeffs"] = mask_coeffs_out;

        results[result_idx++] = box;
    }

    return results;
}

LuaIntf::LuaRef Tensor::process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs,
                                      const LuaIntf::LuaRef& box,
                                      int img_w, int img_h,
                                      int input_w, int input_h,
                                      int pad_x, int pad_y) {
    check_cpu();
    Tensor proto = contiguous();

    auto proto_shape = proto.shape();
    if (proto_shape.size() != 4 || proto_shape[1] != 32) {
        throw std::runtime_error("Invalid proto mask shape");
    }

    int mh = static_cast<int>(proto_shape[2]);
    int mw = static_cast<int>(proto_shape[3]);
    int num_masks = 32;

    const float* proto_data = static_cast<const float*>(proto.buffer_->data()) + proto.offset_;

    // 1. Matrix Multiplication: Mask = Coeffs * Proto
    cv::Mat proto_mat(num_masks, mh * mw, CV_32F, const_cast<float*>(proto_data));
    cv::Mat coeffs_mat(1, num_masks, CV_32F);

    for (int i = 0; i < num_masks; ++i) {
        coeffs_mat.at<float>(0, i) = mask_coeffs.get<float>(i + 1);
    }

    cv::Mat mask_flat = coeffs_mat * proto_mat;
    cv::Mat mask = mask_flat.reshape(1, mh);

    // 2. Sigmoid
    cv::exp(-mask, mask);
    mask = 1.0f / (1.0f + mask);

    // 3. Resize to Input Size
    cv::Mat mask_input;
    cv::resize(mask, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    // 4. Crop Padding
    int roi_w = input_w - 2 * pad_x;
    int roi_h = input_h - 2 * pad_y;

    cv::Rect roi(pad_x, pad_y, roi_w, roi_h);
    roi = roi & cv::Rect(0, 0, input_w, input_h);

    if (roi.area() == 0) {
         return LuaIntf::LuaRef::fromValue(L, Tensor(std::vector<float>(img_w * img_h, 0), {1, static_cast<int64_t>(img_h), static_cast<int64_t>(img_w)}));
    }

    cv::Mat mask_cropped = mask_input(roi);

    // 5. Resize to Original Image Size
    cv::Mat mask_original;
    cv::resize(mask_cropped, mask_original, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    // 6. Crop by Box
    float bx = box.get<float>("x");
    float by = box.get<float>("y");
    float bw = box.get<float>("w");
    float bh = box.get<float>("h");

    cv::Rect box_rect(static_cast<int>(bx), static_cast<int>(by),
                      static_cast<int>(bw), static_cast<int>(bh));
    box_rect = box_rect & cv::Rect(0, 0, img_w, img_h);

    cv::Mat final_mask = cv::Mat::zeros(img_h, img_w, CV_32F);
    if (box_rect.area() > 0) {
        mask_original(box_rect).copyTo(final_mask(box_rect));
    }

    // 7. Threshold
    cv::threshold(final_mask, final_mask, 0.5, 1.0, cv::THRESH_BINARY);

    // Return as Tensor
    std::vector<float> mask_data(img_w * img_h);
    std::memcpy(mask_data.data(), final_mask.data, img_w * img_h * sizeof(float));

    return LuaIntf::LuaRef::fromValue(L, Tensor(std::move(mask_data), {1, static_cast<int64_t>(img_h), static_cast<int64_t>(img_w)}));
}

LuaIntf::LuaRef Tensor::argmax(lua_State* L) {
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }

    check_cpu();
    Tensor a = contiguous();

    int num_classes = static_cast<int>(shape_[1]);
    int max_idx = 0;
    const float* ptr = static_cast<const float*>(a.buffer_->data()) + a.offset_;
    float max_val = ptr[0];

    for (int i = 1; i < num_classes; ++i) {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
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

    check_cpu();
    Tensor a = contiguous();

    int num_classes = static_cast<int>(shape_[1]);
    if (k > num_classes) k = num_classes;

    const float* ptr = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    std::vector<std::pair<float, int>> scores(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        scores[i] = {ptr[i], i};
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


} // namespace tensor
