#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "inference/inference.h"
#include "main_util.h"

// Configuration
const int INPUT_W = 640;
const int INPUT_H = 640;
const float CONF_THRES = 0.25f;
const float IOU_THRES = 0.45f;
const int STRIDE = 32;

// ============ Data Structures ============

struct Detection {
    float x, y, w, h;
    float score;
    int class_id;
};

struct PreprocessMeta {
    float scale;
    int pad_x, pad_y;
    int ori_w, ori_h;
};

// YOLO model configuration (auto-detected from output shape)
struct YoloConfig {
    int num_boxes;
    int box_dim;
    bool has_objectness;
    bool transposed;  // [1, 85, 25200] vs [1, 25200, 85]

    // Auto-detect YOLO version from output shape
    // YOLOv5: [1, 25200, 85] with objectness
    // YOLO11: [1, 84, 8400] without objectness (CHW format)
    static YoloConfig detect(const std::vector<int64_t>& shape) {
        YoloConfig cfg;
        int64_t dim1 = shape[1];
        int64_t dim2 = shape[2];

        cfg.transposed = (dim1 < dim2 && dim2 > 100);
        cfg.num_boxes = cfg.transposed ? dim2 : dim1;
        cfg.box_dim = cfg.transposed ? dim1 : dim2;

        // YOLOv5: 85 = 4(xywh) + 1(objectness) + 80(classes)
        // YOLO11: 84 = 4(xywh) + 80(classes)
        cfg.has_objectness = (cfg.box_dim == 85);

        return cfg;
    }

    void print() const {
        std::cout << "YOLO Config:\n";
        std::cout << "  Format: " << (transposed ? "CHW [1, " + std::to_string(box_dim) + ", " + std::to_string(num_boxes) + "]"
                                                  : "HWC [1, " + std::to_string(num_boxes) + ", " + std::to_string(box_dim) + "]") << "\n";
        std::cout << "  Version: " << (has_objectness ? "YOLOv5" : "YOLO11") << "\n";
        std::cout << "  Boxes: " << num_boxes << "\n";
    }
};

// ============ Preprocessing (Optimized) ============

PreprocessMeta letterbox_resize(const cv::Mat& img, cv::Mat& output,
                                 int target_w, int target_h, int stride, uint8_t fill_value) {
    int w = img.cols;
    int h = img.rows;

    float r = std::min((float)target_h / h, (float)target_w / w);
    int new_w = std::floor(w * r);
    int new_h = std::floor(h * r);

    cv::Mat resized;
    if (new_w != w || new_h != h) {
        cv::resize(img, resized, cv::Size(new_w, new_h));
    } else {
        resized = img;  // No copy needed
    }

    int dw = target_w - new_w;
    int dh = target_h - new_h;

    // Note: No stride alignment needed when target size is fixed
    // The target_w and target_h are already multiples of stride

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    cv::copyMakeBorder(resized, output, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(fill_value, fill_value, fill_value));

    return {r, left, top, w, h};
}

void hwc_to_chw_bgr2rgb(const cv::Mat& padded, std::vector<float>& blob, bool pre_allocated = false) {
    const int H = padded.rows;
    const int W = padded.cols;
    const int HW = H * W;

    // Only resize if not pre-allocated (video mode uses pre-allocated buffers)
    if (!pre_allocated) {
        blob.resize(3 * HW);
    }

    const uint8_t* src = padded.data;

    // Optimized: direct pointer access + BGR→RGB conversion
    // Use multiplication instead of division (3-5x faster on embedded systems)
    constexpr float scale = 1.0f / 255.0f;

    for (int i = 0; i < H; ++i) {
        const uint8_t* row = src + i * W * 3;
        for (int j = 0; j < W; ++j) {
            const int idx = i * W + j;
            // BGR → RGB: swap channels 0 and 2
            blob[2 * HW + idx] = row[j * 3 + 0] * scale;  // B → R
            blob[1 * HW + idx] = row[j * 3 + 1] * scale;  // G → G
            blob[0 * HW + idx] = row[j * 3 + 2] * scale;  // R → B
        }
    }
}

// ============ Postprocessing ============

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

void restore_coords(Detection& det, const PreprocessMeta& meta) {
    det.x = (det.x - meta.pad_x) / meta.scale;
    det.y = (det.y - meta.pad_y) / meta.scale;
    det.w = det.w / meta.scale;
    det.h = det.h / meta.scale;
}

// Generic YOLO postprocessing (supports both v5 and v11)
std::vector<Detection> postprocess_yolo(
    const float* output,
    const YoloConfig& cfg,
    const PreprocessMeta& meta,
    float conf_thres,
    float iou_thres
) {
    std::vector<Detection> proposals;
    proposals.reserve(cfg.num_boxes / 20);

    const int cls_start = cfg.has_objectness ? 5 : 4;
    const int num_classes = cfg.box_dim - cls_start;

    if (cfg.transposed) {
        // CHW format: [1, 84/85, 8400/25200]
        const float* cx_ptr = output + 0 * cfg.num_boxes;
        const float* cy_ptr = output + 1 * cfg.num_boxes;
        const float* w_ptr  = output + 2 * cfg.num_boxes;
        const float* h_ptr  = output + 3 * cfg.num_boxes;
        const float* obj_ptr = cfg.has_objectness ? (output + 4 * cfg.num_boxes) : nullptr;

        for (int i = 0; i < cfg.num_boxes; ++i) {
            float obj_conf = obj_ptr ? obj_ptr[i] : 1.0f;
            if (obj_conf < conf_thres) continue;

            // Find max class score
            float max_cls_conf = 0;
            int cls_id = 0;
            for (int c = 0; c < num_classes; ++c) {
                float conf = output[(cls_start + c) * cfg.num_boxes + i];
                if (conf > max_cls_conf) {
                    max_cls_conf = conf;
                    cls_id = c;
                }
            }

            float final_score = obj_conf * max_cls_conf;
            if (final_score < conf_thres) continue;

            Detection det{
                cx_ptr[i] - w_ptr[i] * 0.5f,
                cy_ptr[i] - h_ptr[i] * 0.5f,
                w_ptr[i], h_ptr[i],
                final_score, cls_id
            };
            proposals.push_back(det);
        }
    } else {
        // HWC format: [1, 8400/25200, 84/85]
        for (int i = 0; i < cfg.num_boxes; ++i) {
            const float* row = output + i * cfg.box_dim;

            float obj_conf = cfg.has_objectness ? row[4] : 1.0f;
            if (obj_conf < conf_thres) continue;

            // Find max class score
            float max_cls_conf = 0;
            int cls_id = 0;
            const float* cls_scores = row + cls_start;
            for (int c = 0; c < num_classes; ++c) {
                if (cls_scores[c] > max_cls_conf) {
                    max_cls_conf = cls_scores[c];
                    cls_id = c;
                }
            }

            float final_score = obj_conf * max_cls_conf;
            if (final_score < conf_thres) continue;

            Detection det{
                row[0] - row[2] * 0.5f,
                row[1] - row[3] * 0.5f,
                row[2], row[3],
                final_score, cls_id
            };
            proposals.push_back(det);
        }
    }

    // Perform NMS on letterbox coordinates
    auto final_boxes = nms(proposals, iou_thres);

    // Restore coordinates to original image space (only for final detections)
    // This is more efficient than restoring all proposals (typically 100-500 boxes)
    for (auto& det : final_boxes) {
        restore_coords(det, meta);
    }

    return final_boxes;
}

// ============ Visualization ============

void print_results(const std::vector<Detection>& results) {
    std::cout << "\n=== Detection Results ===\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& det = results[i];
        std::string label = (det.class_id >= 0 && det.class_id < COCO_LABELS.size())
                          ? COCO_LABELS[det.class_id] : "unknown";
        std::cout << "Box " << (i+1) << ": " << label << " "
                  << "(" << det.x << ", " << det.y << ", " << det.w << ", " << det.h << ") "
                  << "conf=" << det.score << "\n";
    }
    std::cout << "Total: " << results.size() << " detections\n";
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, cv::Rect(det.x, det.y, det.w, det.h),
                     cv::Scalar(0, 255, 0), 2);
        std::string label = (det.class_id >= 0 && det.class_id < COCO_LABELS.size())
                          ? COCO_LABELS[det.class_id] : "unknown";
        std::string text = label + " " + std::to_string(det.score).substr(0, 4);
        cv::putText(frame, text, cv::Point(det.x, det.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// ============ Utilities ============

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model.onnx> <input> [show] [save=output.mp4] [frames=N]\n";
    std::cout << "\nInput: image (.jpg, .png) or video (.mp4, .avi)\n";
    std::cout << "Options:\n";
    std::cout << "  show         - Display results\n";
    std::cout << "  save=FILE    - Save output video\n";
    std::cout << "  frames=N     - Process first N frames only\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " yolo11n.onnx image.jpg show\n";
    std::cout << "  " << prog << " yolov5n.onnx video.mp4 show save=out.mp4\n";
}

// ============ Main ============

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];

    bool show_result = false;
    std::string save_path = "";
    int max_frames = -1;

    // Parse options
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "show") {
            show_result = true;
        } else if (arg.find("save=") == 0) {
            save_path = arg.substr(5);
        } else if (arg.find("frames=") == 0) {
            max_frames = std::stoi(arg.substr(7));
        }
    }

    try {
        // Load model
        std::cout << "Loading model: " << model_path << "\n";
        inference::OnnxSession session(model_path, 4);

        // Lambda for inference (image mode)
        auto infer_func = [&](const cv::Mat& frame) -> std::vector<Detection> {
            // 1. Preprocess
            cv::Mat padded;
            auto meta = letterbox_resize(frame, padded, INPUT_W, INPUT_H, STRIDE, 114);
            std::vector<float> blob;
            hwc_to_chw_bgr2rgb(padded, blob);

            // 2. Inference
            auto [output, shape] = session.run(blob.data(), {1, 3, INPUT_H, INPUT_W});

            // 3. Postprocess (auto-detect and cache YOLO config)
            static YoloConfig cached_cfg;
            static bool first_run = true;
            if (first_run) {
                cached_cfg = YoloConfig::detect(shape);
                cached_cfg.print();
                std::cout << "\n";
                first_run = false;
            }

            return postprocess_yolo(output.data(), cached_cfg, meta, CONF_THRES, IOU_THRES);
        };

        if (is_video_file(input_path)) {
            // ========== Video inference ==========
            cv::VideoCapture cap(input_path);
            if (!cap.isOpened()) {
                throw std::runtime_error("Failed to open video: " + input_path);
            }

            int fps = cap.get(cv::CAP_PROP_FPS);
            int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

            std::cout << "\n=== Video Info ===\n";
            std::cout << "Resolution: " << width << "x" << height << "\n";
            std::cout << "FPS: " << fps << "\n";
            std::cout << "Total frames: " << total_frames << "\n";
            if (max_frames > 0) std::cout << "Limit: " << max_frames << " frames\n";
            std::cout << "\n";

            cv::VideoWriter writer;
            if (!save_path.empty()) {
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open(save_path, fourcc, fps, cv::Size(width, height));
                if (writer.isOpened()) {
                    std::cout << "Output: " << save_path << "\n\n";
                }
            }

            // Pre-allocate buffers for video processing (avoid per-frame allocation)
            cv::Mat padded_buffer;
            std::vector<float> blob_buffer(3 * INPUT_H * INPUT_W);

            // Optimized video inference lambda (uses pre-allocated buffers + cached config)
            auto video_infer_func = [&](const cv::Mat& frame) -> std::vector<Detection> {
                // 1. Preprocess (reuse buffers)
                auto meta = letterbox_resize(frame, padded_buffer, INPUT_W, INPUT_H, STRIDE, 114);
                hwc_to_chw_bgr2rgb(padded_buffer, blob_buffer, true);  // true = pre-allocated

                // 2. Inference
                auto [output, shape] = session.run(blob_buffer.data(), {1, 3, INPUT_H, INPUT_W});

                // 3. Postprocess (cache YOLO config to avoid repeated detection)
                static YoloConfig cached_cfg;
                static bool first_run = true;
                if (first_run) {
                    cached_cfg = YoloConfig::detect(shape);
                    cached_cfg.print();
                    std::cout << "\n";
                    first_run = false;
                }

                return postprocess_yolo(output.data(), cached_cfg, meta, CONF_THRES, IOU_THRES);
            };

            std::cout << "Processing video...\n\n";

            // Memory monitoring
            MemoryInfo mem_start, mem_current, mem_peak;
            mem_start.update();
            mem_current = mem_start;
            mem_peak = mem_start;

            int frame_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();

            cv::Mat frame;
            while (cap.read(frame)) {
                frame_count++;
                if (max_frames > 0 && frame_count > max_frames) break;

                auto results = video_infer_func(frame);
                draw_detections(frame, results);

                if (writer.isOpened()) writer.write(frame);
                if (show_result) {
                    cv::imshow("Inference", frame);
                    if (cv::waitKey(1) == 27) break;  // ESC to exit
                }

                // Update memory tracking
                mem_current.update();
                if (mem_current.vm_rss_kb > mem_peak.vm_rss_kb) {
                    mem_peak = mem_current;
                }

                // Print progress (every 30 frames or first frame)
                if (frame_count % 30 == 0 || frame_count == 1) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    double current_fps = frame_count * 1000.0 / elapsed_ms;
                    std::cout << "\rFrame: " << frame_count
                             << " | FPS: " << std::fixed << std::setprecision(1) << current_fps
                             << " | Det: " << results.size()
                             << " | " << mem_current.to_string() << "     " << std::flush;
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            std::cout << "\n\n=== Performance Summary ===\n";
            std::cout << "Processed: " << frame_count << " frames\n";
            std::cout << "Time: " << std::fixed << std::setprecision(2) << (total_ms / 1000.0) << " s\n";
            std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
                     << (frame_count * 1000.0 / total_ms) << "\n";

            // Memory summary
            std::cout << "\n=== Memory Summary ===\n";
            std::cout << "Initial:  " << mem_start.to_string() << "\n";
            std::cout << "Final:    " << mem_current.to_string() << "\n";
            std::cout << "Peak:     " << mem_peak.to_string() << "\n";

            // Memory leak detection
            long mem_increase_kb = static_cast<long>(mem_current.vm_rss_kb) - static_cast<long>(mem_start.vm_rss_kb);
            double mem_per_frame_kb = frame_count > 0 ? static_cast<double>(mem_increase_kb) / frame_count : 0.0;
            std::cout << "Increase: " << std::fixed << std::setprecision(1)
                     << (mem_increase_kb / 1024.0) << " MB total, "
                     << mem_per_frame_kb << " KB/frame\n";

            if (frame_count > 100 && mem_per_frame_kb > 10.0) {
                std::cout << "\n⚠️  Warning: Memory leak detected (>" << mem_per_frame_kb << " KB/frame)\n";
            }

            cap.release();
            if (writer.isOpened()) {
                writer.release();
                std::cout << "\nOutput saved: " << save_path << "\n";
            }
            if (show_result) cv::destroyAllWindows();

        } else {
            // ========== Image inference ==========
            std::cout << "Loading image: " << input_path << "\n";
            cv::Mat img = cv::imread(input_path);
            if (img.empty()) {
                throw std::runtime_error("Failed to load image");
            }
            std::cout << "Image size: " << img.cols << "x" << img.rows << "\n\n";

            auto start = std::chrono::high_resolution_clock::now();
            auto results = infer_func(img);
            auto end = std::chrono::high_resolution_clock::now();

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Inference time: " << elapsed << " ms\n";

            print_results(results);

            if (show_result || !save_path.empty()) {
                draw_detections(img, results);

                if (!save_path.empty()) {
                    cv::imwrite(save_path, img);
                    std::cout << "\nResult saved: " << save_path << "\n";
                }

                if (show_result) {
                    cv::imshow("Result", img);
                    std::cout << "Press any key to exit...\n";
                    cv::waitKey(0);
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
