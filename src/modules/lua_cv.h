#ifndef MODEL_INFER_LUA_CV_H_
#define MODEL_INFER_LUA_CV_H_

#include <opencv2/opencv.hpp>
#include <functional>
#include <map>
#include <string>
#include "LuaIntf.h"

// 使用 lua_nn.h 中的 Tensor (实际是 tensor::Tensor)
#include "lua_nn.h"

namespace lua_cv {

class Image {
public:
    explicit Image(const cv::Mat& mat);
    Image();  // 默认构造函数

    // 属性访问（通过getter，不直接暴露成员）
    int width() const { return mat_.cols; }
    int height() const { return mat_.rows; }
    int channels() const { return mat_.channels(); }
    bool empty() const { return mat_.empty(); }

    // 图像操作（原地修改）
    void resize(int new_w, int new_h);
    void pad(int top, int bottom, int left, int right, int fill_value);

    // 返回Tensor对象
    lua_nn::Tensor to_tensor(double scale,
                             const LuaIntf::LuaRef& mean,
                             const LuaIntf::LuaRef& std) const;

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

// ========== 预处理注册表 ==========

// 预处理结果：Tensor + Meta信息
struct PreprocessResult {
    lua_nn::Tensor tensor;
    LuaIntf::LuaRef meta;

    PreprocessResult(lua_nn::Tensor t, LuaIntf::LuaRef m)
        : tensor(std::move(t)), meta(m) {}
};

// 预处理函数签名
using PreprocessFunc = std::function<PreprocessResult(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
)>;

// 预处理函数注册表
class PreprocessRegistry {
public:
    static PreprocessRegistry& instance();

    // 注册预处理函数
    void register_func(const std::string& type, PreprocessFunc func);

    // 执行预处理
    PreprocessResult run(
        const std::string& type,
        Image& img,
        lua_State* L,
        const LuaIntf::LuaRef& config
    );

    // 检查是否支持某类型
    bool has(const std::string& type) const;

private:
    PreprocessRegistry();
    std::map<std::string, PreprocessFunc> registry_;
};

// 预定义的预处理函数
PreprocessResult preprocess_letterbox(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
);

PreprocessResult preprocess_resize_center_crop(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
);

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_cv

#endif
