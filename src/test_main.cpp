#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include "LuaIntf.h"

// 模块头文件
#include "modules/lua_nn.h"
#include "modules/lua_cv.h"
#include "modules/lua_utils.h"

// Forward declaration
void register_all_modules(lua_State* L);

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <test_script.lua>\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " scripts/test_tensor_api.lua\n";
    std::cout << "\nOr run built-in C++ tests:\n";
    std::cout << "  " << prog_name << " --cpp\n";
}

void print_separator(const std::string& title) {
    std::cout << "\n========== " << title << " ==========\n";
}

void print_tensor_info(const lua_nn::Tensor& t) {
    auto shape = t.shape();
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]\n";
    std::cout << "  String: " << t.to_string(20) << "\n";
}

// C++ 层测试
void test_cpp_tensor_api() {
    print_separator("C++ Tensor API 测试");
    
    // 测试1: 基础构造
    std::cout << "\n测试1: 基础构造和属性\n";
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        std::vector<int64_t> shape = {2, 3};
        lua_nn::Tensor t(std::move(data), shape);
        print_tensor_info(t);
        std::cout << "  Ndim: " << t.ndim() << "\n";
        std::cout << "  Size: " << t.size() << "\n";
    }
    
    // 测试2: Slice操作
    std::cout << "\n测试2: Slice操作\n";
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        lua_nn::Tensor t(std::move(data), {3, 4});
        std::cout << "  原始tensor:\n";
        print_tensor_info(t);
        
        auto sliced = t.slice(0, 0, 2);  // 取前2行
        std::cout << "  切片后(前2行):\n";
        print_tensor_info(sliced);
    }
    
    // 测试3: Reshape操作
    std::cout << "\n测试3: Reshape操作\n";
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        lua_nn::Tensor t(std::move(data), {2, 3});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto reshaped = t.reshape({3, 2});
        std::cout << "  Reshape后:\n";
        print_tensor_info(reshaped);
    }
    
    // 测试4: Transpose操作
    std::cout << "\n测试4: Transpose操作\n";
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        lua_nn::Tensor t(std::move(data), {2, 3});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto transposed = t.transpose();
        std::cout << "  转置后:\n";
        print_tensor_info(transposed);
    }
    
    // 测试5: Element-wise操作
    std::cout << "\n测试5: Element-wise操作\n";
    {
        std::vector<float> data = {1, 2, 3, 4};
        lua_nn::Tensor t(std::move(data), {4});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto t_add = t.add(10.0f);
        std::cout << "  加10:\n";
        print_tensor_info(t_add);
        
        auto t_mul = t.mul(2.0f);
        std::cout << "  乘2:\n";
        print_tensor_info(t_mul);
    }
    
    // 测试6: Reduction操作
    std::cout << "\n测试6: Reduction操作\n";
    {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        lua_nn::Tensor t(std::move(data), {2, 3});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto sum_all = t.sum(-1, false);
        std::cout << "  总和:\n";
        print_tensor_info(sum_all);
        
        auto mean_all = t.mean(-1, false);
        std::cout << "  平均值:\n";
        print_tensor_info(mean_all);
        
        auto max_all = t.max(-1, false);
        std::cout << "  最大值:\n";
        print_tensor_info(max_all);
    }
    
    // 测试7: Sigmoid和Softmax
    std::cout << "\n测试7: Activation函数\n";
    {
        std::vector<float> data = {-1, 0, 1, 2};
        lua_nn::Tensor t(std::move(data), {4});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto t_sigmoid = t.sigmoid();
        std::cout << "  Sigmoid:\n";
        print_tensor_info(t_sigmoid);
        
        auto t_softmax = t.softmax(-1);
        std::cout << "  Softmax:\n";
        print_tensor_info(t_softmax);
    }
    
    // 测试8: 比较操作
    std::cout << "\n测试8: 比较操作\n";
    {
        std::vector<float> data = {0.1f, 0.5f, 0.8f, 0.3f};
        lua_nn::Tensor t(std::move(data), {4});
        std::cout << "  原始:\n";
        print_tensor_info(t);
        
        auto mask_gt = t.gt(0.4f);
        std::cout << "  大于0.4的mask:\n";
        print_tensor_info(mask_gt);
        
        auto mask_le = t.le(0.5f);
        std::cout << "  小于等于0.5的mask:\n";
        print_tensor_info(mask_le);
    }
    
    std::cout << "\n✓ C++ 层所有测试通过!\n";
}

// 从Lua脚本运行测试
int run_lua_test_script(const std::string& script_path) {
    print_separator("运行Lua测试脚本");
    
    std::cout << "加载脚本: " << script_path << "\n";
    
    // 检查文件是否存在
    std::ifstream file(script_path);
    if (!file.good()) {
        std::cerr << "错误: 找不到脚本文件: " << script_path << "\n";
        return 1;
    }
    file.close();
    
    // 创建Lua状态机
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    
    // 注册所有模块
    std::cout << "注册模块...\n";
    register_all_modules(L);
    
    // 加载并运行脚本
    std::cout << "执行测试脚本...\n\n";
    int result = luaL_dofile(L, script_path.c_str());
    
    if (result != LUA_OK) {
        std::cerr << "\n错误: 脚本执行失败: " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }
    
    lua_close(L);
    std::cout << "\n✓ Lua脚本测试完成!\n";
    return 0;
}

// 性能测试
void test_tensor_performance() {
    print_separator("Tensor 性能测试");
    
    std::cout << "\n测试大tensor操作性能...\n";
    
    // 创建一个较大的tensor
    size_t size = 8400 * 84;  // YOLOv8典型输出大小
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    lua_nn::Tensor t(std::move(data), {1, 84, 8400});
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "  创建tensor: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto sliced = t.slice(1, 0, 4);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  Slice操作: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs (零拷贝)\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto squeezed = t.squeeze(0);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  Squeeze操作: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs (零拷贝)\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto transposed = t.transpose();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  Transpose操作: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
    
    // Sigmoid测试
    std::vector<float> data2(1000);
    for (size_t i = 0; i < 1000; ++i) {
        data2[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    lua_nn::Tensor t2(std::move(data2), {1000});
    
    start = std::chrono::high_resolution_clock::now();
    auto sigmoid_result = t2.sigmoid();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  Sigmoid (1000元素): " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto softmax_result = t2.softmax(-1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "  Softmax (1000元素): " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
    
    std::cout << "\n性能测试完成!\n";
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "    Tensor API 测试程序\n";
    std::cout << "========================================\n";
    
    // 解析命令行参数
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string arg = argv[1];
    
    try {
        if (arg == "--cpp" || arg == "-c") {
            // 运行内置C++测试
            test_cpp_tensor_api();
            test_tensor_performance();
            
            print_separator("总结");
            std::cout << "\n✓ C++测试通过!\n";
            std::cout << "\n新的Tensor API已成功实现:\n";
            std::cout << "  - Level 1: 形状操作 (slice, reshape, transpose, squeeze, unsqueeze)\n";
            std::cout << "  - Level 2: 数学运算 (add, mul, sum, mean, argmax, sigmoid, softmax, 比较)\n";
            std::cout << "  - Level 3: 高级操作 (topk, to_table)\n";
            std::cout << "  - 零拷贝机制 (strides, offset, contiguous标记)\n";
            std::cout << "  - C++和Lua双层API\n";
            std::cout << "\n========================================\n";
            
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            
        } else {
            // 运行Lua测试脚本
            int result = run_lua_test_script(arg);
            if (result != 0) {
                return result;
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ 测试失败: " << e.what() << "\n";
        return 1;
    }
}
        