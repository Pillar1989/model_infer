#include <iostream>
#include <string>
#include "LuaIntf.h"

// 模块头文件
#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"

// Forward declaration
void register_all_modules(lua_State* L);

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <script.lua> <model.onnx> <image.jpg>\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " scripts/yolov5_detector.lua models/yolov5n.onnx images/zidane.jpg\n";
}

void print_results(LuaIntf::LuaRef results) {
    int len = results.len();
    if (len == 0) {
        std::cout << "No results.\n";
        return;
    }

    LuaIntf::LuaRef first = results[1];
    if (first.has("x") && first.has("y")) {
        std::cout << "\n=== Detection Results ===\n";
        for (int i = 1; i <= len; ++i) {
            LuaIntf::LuaRef det = results[i];
            
            float x = det.get<float>("x");
            float y = det.get<float>("y");
            float w = det.get<float>("w");
            float h = det.get<float>("h");
            float score = det.get<float>("score");
            
            std::string label = "unknown";
            if (det.has("label")) {
                label = det.get<std::string>("label");
            }
            
            std::cout << "Box " << i << ": "
                      << label << " "
                      << "(" << x << ", " << y << ", " << w << ", " << h << ") "
                      << "conf=" << score;
            if (det.has("keypoints")) {
                std::cout << " +kpts";
            }
            std::cout << "\n";
        }
        std::cout << "Total: " << len << " detections\n";
    } else if (first.has("class_id")) {
        std::cout << "\n=== Classification Results ===\n";
        for (int i = 1; i <= len; ++i) {
            LuaIntf::LuaRef item = results[i];
            int class_id = item.get<int>("class_id");
            float conf = item.get<float>("confidence");
            std::string label = "";
            if (item.has("label")) {
                label = " (" + item.get<std::string>("label") + ")";
            }
            std::cout << "Top " << i << ": Class " << class_id << label << " Conf=" << conf << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string script_path = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];
    bool show_result = false;
    
    if (argc > 4 && std::string(argv[4]) == "show") {
        show_result = true;
    }
    
    lua_State* L = nullptr;

    try {
        // 1. 初始化Lua
        L = luaL_newstate();
        if (!L) {
            throw std::runtime_error("Failed to create Lua state");
        }
        luaL_openlibs(L);
        
        {
            // 2. 注册C++模块
            std::cout << "Registering modules...\n";
            register_all_modules(L);
            
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
            LuaIntf::LuaRef model = LuaIntf::LuaRef::popFromStack(L);
            if (!model.isTable()) {
                throw std::runtime_error("Script must return a Model table");
            }
            
            // 7. 预处理
            std::cout << "Preprocessing...\n";
            LuaIntf::LuaRef preprocess = model["preprocess"];
            if (!preprocess.isFunction()) {
                throw std::runtime_error("Model.preprocess must be a function");
            }
            
            // Push function and argument
            preprocess.pushToStack();
            LuaIntf::LuaRef::fromValue(L, img).pushToStack();
            
            // Call with 1 argument, 2 results
            if (lua_pcall(L, 1, 2, 0) != LUA_OK) {
                const char* err = lua_tostring(L, -1);
                throw std::runtime_error("Error running preprocess: " + std::string(err));
            }
            
            LuaIntf::LuaRef meta = LuaIntf::LuaRef::popFromStack(L);
            LuaIntf::LuaRef input_tensor_ref = LuaIntf::LuaRef::popFromStack(L);
            
            // 8. 推理
            std::cout << "Running inference...\n";
            // Convert LuaRef to Tensor object
            lua_nn::Tensor input_tensor = input_tensor_ref.toValue<lua_nn::Tensor>();
            LuaIntf::LuaRef outputs = session.run(L, input_tensor);
            
            // 9. 后处理
            std::cout << "Postprocessing...\n";
            LuaIntf::LuaRef postprocess = model["postprocess"];
            if (!postprocess.isFunction()) {
                throw std::runtime_error("Model.postprocess must be a function");
            }
            
            LuaIntf::LuaRef detections = postprocess.call<LuaIntf::LuaRef>(outputs, meta);
            
            // 10. 打印结果
            print_results(detections);
            
            // 11. 可视化结果
            if (show_result) {
                cv::Mat draw_img = cv::imread(image_path);
                int len = detections.len();
                
                if (len > 0) {
                    LuaIntf::LuaRef first = detections[1];
                    if (first.has("x") && first.has("y")) {
                        // Detection visualization
                        for (int i = 1; i <= len; ++i) {
                            LuaIntf::LuaRef det = detections[i];
                            float x = det.get<float>("x");
                            float y = det.get<float>("y");
                            float w = det.get<float>("w");
                            float h = det.get<float>("h");
                            float score = det.get<float>("score");
                            std::string label = det.has("label") ? det.get<std::string>("label") : "unknown";
                            
                            cv::rectangle(draw_img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
                            std::string text = label + " " + std::to_string(score).substr(0, 4);
                            int baseline = 0;
                            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                            cv::rectangle(draw_img, cv::Point(x, y - textSize.height - 5), cv::Point(x + textSize.width, y), cv::Scalar(0, 255, 0), -1);
                            cv::putText(draw_img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                            
                            // Pose Visualization
                            if (det.has("keypoints")) {
                                LuaIntf::LuaRef kpts = det.get<LuaIntf::LuaRef>("keypoints");
                                int num_kpts = kpts.len();
                                std::vector<cv::Point> points;
                                for(int k=1; k<=num_kpts; ++k) {
                                    LuaIntf::LuaRef kp = kpts[k];
                                    float kx = kp.get<float>("x");
                                    float ky = kp.get<float>("y");
                                    float kv = kp.get<float>("v");
                                    if (kv > 0.5) {
                                        cv::circle(draw_img, cv::Point(kx, ky), 3, cv::Scalar(0, 0, 255), -1);
                                        points.push_back(cv::Point(kx, ky));
                                    } else {
                                        points.push_back(cv::Point(-1, -1));
                                    }
                                }
                                
                                // Draw skeleton (COCO 17 keypoints)
                                const std::vector<std::pair<int, int>> skeleton = {
                                    {0,1}, {0,2}, {1,3}, {2,4}, // Face
                                    {5,6}, {5,7}, {7,9}, {6,8}, {8,10}, // Arms
                                    {11,12}, {11,13}, {13,15}, {12,14}, {14,16}, // Legs
                                    {5,11}, {6,12} // Torso
                                };
                                
                                for (const auto& limb : skeleton) {
                                    int idx1 = limb.first;
                                    int idx2 = limb.second;
                                    if (idx1 < points.size() && idx2 < points.size() && 
                                        points[idx1].x != -1 && points[idx2].x != -1) {
                                        cv::line(draw_img, points[idx1], points[idx2], cv::Scalar(255, 0, 0), 2);
                                    }
                                }
                            }
                        }
                    } else if (first.has("class_id")) {
                        // Classification visualization
                        for (int i = 1; i <= std::min(len, 5); ++i) {
                            LuaIntf::LuaRef item = detections[i];
                            int class_id = item.get<int>("class_id");
                            float conf = item.get<float>("confidence");
                            std::string text = "Class " + std::to_string(class_id) + ": " + std::to_string(conf).substr(0, 4);
                            cv::putText(draw_img, text, cv::Point(10, 30 * i), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                        }
                    }
                }
                
                cv::imshow("Result", draw_img);
                std::cout << "Press any key to exit..." << std::endl;
                cv::waitKey(0);
            }
        }
        
        // 12. 清理
        if (L) {
            lua_close(L);
            L = nullptr;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        if (L) lua_close(L);
        return 1;
    }
    
    return 0;
}
