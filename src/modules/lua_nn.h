#ifndef MODEL_INFER_LUA_NN_H_
#define MODEL_INFER_LUA_NN_H_

#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <cmath>

#include "onnxruntime_cxx_api.h"
#include "LuaIntf.h"

namespace lua_nn {

class Tensor {
public:
    // ========== 构造/析构 ==========
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape);
    
    // 从原始指针构造（零拷贝，需要shared_ptr管理生命周期）
    Tensor(const float* data, const std::vector<int64_t>& shape, std::shared_ptr<std::vector<float>> owner = nullptr);
    
    // 内部构造（支持strides和offset）
    Tensor(std::shared_ptr<std::vector<float>> data, 
           const std::vector<int64_t>& shape,
           const std::vector<int64_t>& strides,
           int64_t offset,
           bool contiguous);
    
    // ========== 属性访问 ==========
    std::vector<int64_t> shape() const { return shape_; }
    std::vector<int64_t> strides() const { return strides_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t size() const;  // 返回元素总数
    int64_t size(int dim) const;  // 返回指定维度大小
    bool is_contiguous() const { return contiguous_; }
    
    // 零拷贝视图（性能关键，向后兼容）
    LuaIntf::TensorView<float> view() {
        if (!contiguous_) {
            auto cont = contiguous_copy();
            return LuaIntf::TensorView<float>(cont.data_->data(), cont.data_->size(), cont.data_);
        }
        return LuaIntf::TensorView<float>(data_->data() + offset_, compute_size(), data_);
    }
    
    // ========== Level 1: 基础形状操作 ==========
    // 切片操作（零拷贝）
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    
    // Reshape（零拷贝，仅改变shape）
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    
    // Transpose（维度置换）
    Tensor transpose(const std::vector<int>& dims) const;
    Tensor transpose() const;  // 默认反转所有维度
    
    // Squeeze/Unsqueeze
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    
    // ========== Level 2: 数学运算 ==========
    // Element-wise操作
    Tensor add(const Tensor& other) const;
    Tensor add(float scalar) const;
    Tensor sub(const Tensor& other) const;
    Tensor sub(float scalar) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor div(const Tensor& other) const;
    Tensor div(float scalar) const;
    
    // Reduction操作（axis=-1表示所有维度）
    Tensor sum(int axis = -1, bool keepdims = false) const;
    Tensor mean(int axis = -1, bool keepdims = false) const;
    Tensor max(int axis = -1, bool keepdims = false) const;
    Tensor min(int axis = -1, bool keepdims = false) const;
    
    // Argmax/Argmin（返回Lua table或单值）
    LuaIntf::LuaRef argmax_lua(lua_State* L, int axis = -1) const;
    LuaIntf::LuaRef argmin_lua(lua_State* L, int axis = -1) const;
    
    // Activation函数
    Tensor sigmoid() const;
    Tensor softmax(int axis = -1) const;
    Tensor exp_() const;
    Tensor log_() const;
    
    // 比较操作（返回bool mask tensor）
    Tensor gt(float threshold) const;
    Tensor lt(float threshold) const;
    Tensor ge(float threshold) const;
    Tensor le(float threshold) const;
    Tensor eq(float threshold) const;
    
    // ========== Level 3: 高级操作 ==========
    // TopK（返回{values, indices}的Lua table）
    LuaIntf::LuaRef topk_lua(lua_State* L, int k, int axis = -1, bool largest = true) const;

    // Gather（根据索引收集元素）
    Tensor gather(int axis, const Tensor& indices) const;

    // Concat/Split
    static Tensor concat(const std::vector<Tensor>& tensors, int axis);
    std::vector<Tensor> split(int num_splits, int axis) const;

    // ========== 向量化过滤操作（方案3 - 通用API） ==========
    // 返回非零/满足条件的索引（关键优化：直接返回索引列表，避免大量bool tensor）
    std::vector<int64_t> nonzero() const;  // 返回非零元素的索引
    std::vector<int64_t> where_indices(float threshold, const std::string& op = "ge") const;

    // 根据索引选择元素（批量gather，避免逐个访问）
    Tensor index_select(int dim, const std::vector<int64_t>& indices) const;

    // 高效的多列提取（专为[C, N]格式优化）
    LuaIntf::LuaRef extract_columns(lua_State* L, const std::vector<int64_t>& col_indices) const;
    
    // ========== Level 4: Legacy方法（向后兼容） ==========
    LuaIntf::LuaRef filter_yolo(lua_State* L, float conf_thres);
    LuaIntf::LuaRef filter_yolo_pose(lua_State* L, float conf_thres);
    LuaIntf::LuaRef filter_yolo_seg(lua_State* L, float conf_thres);
    
    // Mask处理
    LuaIntf::LuaRef process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs, 
                                      const LuaIntf::LuaRef& box, 
                                      int img_w, int img_h,
                                      int input_w, int input_h,
                                      int pad_x, int pad_y);
    
    // 旧的通用方法（保留向后兼容）
    LuaIntf::LuaRef argmax(lua_State* L);
    LuaIntf::LuaRef topk(lua_State* L, int k);
    
    // ========== 辅助方法 ==========
    // 直接数据访问
    float get_item(const std::vector<int64_t>& indices) const;
    void set_item(const std::vector<int64_t>& indices, float value);
    
    // 转换为Lua table（调试用）
    LuaIntf::LuaRef to_table(lua_State* L) const;
    
    // 打印（调试用）
    std::string to_string(int max_elements = 10) const;
    
    // 内部访问
    const float* raw_data() const { return data_->data(); }
    float* raw_data() { return data_->data(); }
    const float* data() const { return data_->data() + offset_; }
    float* data() { return data_->data() + offset_; }
    
private:
    std::shared_ptr<std::vector<float>> data_;  // shared_ptr管理数据
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;  // 步长，支持非连续tensor
    int64_t offset_;  // 起始偏移
    bool contiguous_;  // 是否连续
    
    // 内部辅助方法
    int64_t compute_offset(const std::vector<int64_t>& indices) const;
    int64_t compute_size() const;
    std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) const;
    void normalize_axis(int& axis) const;
    Tensor contiguous_copy() const;  // 创建连续拷贝
};

class Session {
public:
    explicit Session(const std::string& model_path);
    
    // 推理方法（接受Tensor对象）
    LuaIntf::LuaRef run(lua_State* L, const Tensor& input_tensor);
    
    // 属性访问
    std::vector<std::string> input_names() const { return input_names_; }
    std::vector<std::string> output_names() const { return output_names_; }
    std::vector<ONNXTensorElementDataType> input_types() const { return input_types_; }
    std::vector<std::vector<int64_t>> input_shapes() const { return input_shapes_; }
    
private:
    std::shared_ptr<Ort::Env> env_;        // shared_ptr自动管理
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<ONNXTensorElementDataType> input_types_;
    std::vector<std::vector<int64_t>> input_shapes_;
};

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_nn

#endif
