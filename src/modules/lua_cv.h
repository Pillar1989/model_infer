#ifndef MODEL_INFER_LUA_CV_H_
#define MODEL_INFER_LUA_CV_H_

#include <opencv2/opencv.hpp>
#include "LuaIntf.h"

// 前向声明
namespace lua_nn { class Tensor; }

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
    
    // 返回Tensor对象（非LuaRef，简化API）
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

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_cv

#endif
