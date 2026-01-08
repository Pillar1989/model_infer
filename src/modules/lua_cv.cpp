#include "lua_cv.h"
#include "lua_nn.h"
#include <stdexcept>
#include <algorithm>

namespace lua_cv {

Image::Image(const cv::Mat& mat) : mat_(mat) {}
Image::Image() {}

void Image::resize(int new_w, int new_h) {
    // 必须使用 cv::resize
    // 插值方法：cv::INTER_LINEAR（默认）
    cv::resize(mat_, mat_, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
}

void Image::pad(int top, int bottom, int left, int right, int fill_value) {
    // 必须使用 cv::copyMakeBorder
    // 边界类型：cv::BORDER_CONSTANT
    cv::copyMakeBorder(mat_, mat_, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(fill_value, fill_value, fill_value));
}

lua_nn::Tensor Image::to_tensor(double scale,
                                 const LuaIntf::LuaRef& mean,
                                 const LuaIntf::LuaRef& std) const {
    std::vector<double> mean_vec, std_vec;
    
    if (mean.isTable()) {
        int len = mean.len();
        for (int i = 1; i <= len; ++i) {
            mean_vec.push_back(mean.get<double>(i));
        }
    }
    
    if (std.isTable()) {
        int len = std.len();
        for (int i = 1; i <= len; ++i) {
            std_vec.push_back(std.get<double>(i));
        }
    }

    // 1. 转换为浮点型
    cv::Mat float_mat;
    mat_.convertTo(float_mat, CV_32F);
    
    // 2. HWC -> CHW 转换（使用cv::split优化，比三重循环快10倍）
    int H = float_mat.rows;
    int W = float_mat.cols;
    int C = float_mat.channels();
    
    // Ensure mean/std have enough elements
    if (mean_vec.size() < C) mean_vec.resize(C, 0.0);
    if (std_vec.size() < C) std_vec.resize(C, 1.0);

    std::vector<cv::Mat> channels(C);
    cv::split(float_mat, channels);
    
    // 3. 分通道归一化并组装CHW数据
    std::vector<float> chw_data(C * H * W);
    size_t idx = 0;
    
    for (int c = 0; c < C; ++c) {
        const float* channel_ptr = channels[c].ptr<float>();
        double m = mean_vec[c];
        double s = std_vec[c];
        for (int i = 0; i < H * W; ++i) {
            chw_data[idx++] = (channel_ptr[i] * scale - m) / s;
        }
    }
    
    // 4. 创建Tensor对象（NCHW格式）
    std::vector<int64_t> shape = {1, static_cast<int64_t>(C), 
                                   static_cast<int64_t>(H), 
                                   static_cast<int64_t>(W)};
    return lua_nn::Tensor(std::move(chw_data), shape);
}

Image Image::clone() const {
    return Image(mat_.clone());
}

Image imread(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    // Convert BGR to RGB (OpenCV loads BGR, models expect RGB)
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    return Image(mat);
}

// ========== 预处理注册表实现 ==========

PreprocessRegistry& PreprocessRegistry::instance() {
    static PreprocessRegistry registry;
    return registry;
}

PreprocessRegistry::PreprocessRegistry() {
    // 在构造函数中注册所有预处理函数
    register_func("letterbox", preprocess_letterbox);
    register_func("resize_center_crop", preprocess_resize_center_crop);

    // 未来可以继续添加：
    // register_func("imagenet", preprocess_imagenet);
    // register_func("face_detection", preprocess_face);
}

void PreprocessRegistry::register_func(const std::string& type, PreprocessFunc func) {
    registry_[type] = func;
}

PreprocessResult PreprocessRegistry::run(
    const std::string& type,
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
) {
    auto it = registry_.find(type);
    if (it == registry_.end()) {
        throw std::runtime_error("Unknown preprocess type: " + type);
    }
    return it->second(img, L, config);
}

bool PreprocessRegistry::has(const std::string& type) const {
    return registry_.find(type) != registry_.end();
}

// ========== Letterbox 预处理实现 ==========

PreprocessResult preprocess_letterbox(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
) {
    // 解析配置
    LuaIntf::LuaRef input_size = config["input_size"];
    int target_h = input_size.get<int>(1);
    int target_w = input_size.get<int>(2);
    int stride = config.get<int>("stride");
    int fill_value = config.get<int>("fill_value");

    int ori_w = img.width();
    int ori_h = img.height();

    // 1. 计算缩放比例
    double r = std::min(
        static_cast<double>(target_h) / ori_h,
        static_cast<double>(target_w) / ori_w
    );
    int new_w = static_cast<int>(ori_w * r);
    int new_h = static_cast<int>(ori_h * r);

    // 2. 缩放
    if (new_w != ori_w || new_h != ori_h) {
        img.resize(new_w, new_h);
    }

    // 3. 计算padding（对齐到stride）
    int dw = target_w - new_w;
    int dh = target_h - new_h;

    dw = dw % stride;
    dh = dh % stride;

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    // 4. 添加padding
    img.pad(top, bottom, left, right, fill_value);

    // 5. 转换为tensor（scale=1/255, mean=0, std=1）
    LuaIntf::LuaRef empty_mean = LuaIntf::LuaRef::createTable(L);
    empty_mean[1] = 0.0; empty_mean[2] = 0.0; empty_mean[3] = 0.0;

    LuaIntf::LuaRef empty_std = LuaIntf::LuaRef::createTable(L);
    empty_std[1] = 1.0; empty_std[2] = 1.0; empty_std[3] = 1.0;

    lua_nn::Tensor tensor = img.to_tensor(1.0 / 255.0, empty_mean, empty_std);

    // 6. 构造meta表
    LuaIntf::LuaRef meta = LuaIntf::LuaRef::createTable(L);
    meta["scale"] = r;
    meta["pad_x"] = left;
    meta["pad_y"] = top;
    meta["ori_w"] = ori_w;
    meta["ori_h"] = ori_h;
    // 添加input_w/input_h作为ori_w/ori_h的别名（向后兼容segmentation脚本）
    meta["input_w"] = new_w + left + right;
    meta["input_h"] = new_h + top + bottom;

    return PreprocessResult(std::move(tensor), meta);
}

// ========== Resize Center Crop 预处理实现 ==========

PreprocessResult preprocess_resize_center_crop(
    Image& img,
    lua_State* L,
    const LuaIntf::LuaRef& config
) {
    int target_size = config.get<int>("size");

    int ori_w = img.width();
    int ori_h = img.height();

    // 1. 短边resize到target_size
    double scale;
    int new_w, new_h;

    if (ori_w < ori_h) {
        // 宽度是短边
        scale = static_cast<double>(target_size) / ori_w;
        new_w = target_size;
        new_h = static_cast<int>(ori_h * scale);
    } else {
        // 高度是短边
        scale = static_cast<double>(target_size) / ori_h;
        new_h = target_size;
        new_w = static_cast<int>(ori_w * scale);
    }

    // 边界检查
    if (new_w < target_size || new_h < target_size) {
        throw std::runtime_error(
            "Image too small for center crop after resize: " +
            std::to_string(new_w) + "x" + std::to_string(new_h) +
            " < " + std::to_string(target_size)
        );
    }

    // 2. Resize
    img.resize(new_w, new_h);

    // 3. Center crop
    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    cv::Rect roi(crop_x, crop_y, target_size, target_size);
    img.data() = img.data()(roi).clone();

    // 4. 转换为tensor
    LuaIntf::LuaRef empty_mean = LuaIntf::LuaRef::createTable(L);
    empty_mean[1] = 0.0; empty_mean[2] = 0.0; empty_mean[3] = 0.0;

    LuaIntf::LuaRef empty_std = LuaIntf::LuaRef::createTable(L);
    empty_std[1] = 1.0; empty_std[2] = 1.0; empty_std[3] = 1.0;

    lua_nn::Tensor tensor = img.to_tensor(1.0 / 255.0, empty_mean, empty_std);

    // 5. 构造meta表
    LuaIntf::LuaRef meta = LuaIntf::LuaRef::createTable(L);
    meta["scale"] = scale;
    meta["crop_x"] = crop_x;
    meta["crop_y"] = crop_y;
    meta["ori_w"] = ori_w;
    meta["ori_h"] = ori_h;

    return PreprocessResult(std::move(tensor), meta);
}

void register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_cv")
            .addFactory(imread)  // 全局函数
            .beginClass<Image>("Image")
                .addConstructor(LUA_ARGS())  // 默认构造
                // 使用addProperty封装属性（非addFunction）
                .addProperty("width", &Image::width)
                .addProperty("height", &Image::height)
                .addProperty("channels", &Image::channels)
                .addFunction("empty", &Image::empty)
                .addFunction("resize", &Image::resize)
                .addFunction("pad", &Image::pad)
                .addFunction("clone", &Image::clone)
                .addFunction("to_tensor", &Image::to_tensor)
            .endClass()
        .endModule();
}

} // namespace lua_cv
