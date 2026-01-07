#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <cmath>
#include <functional>

#include "device_type.h"
#include "tensor_storage.h"
#include "stream.h"

// 前向声明 Lua 相关类型
struct lua_State;
namespace LuaIntf {
    class LuaRef;
    template<typename T> class TensorView;
}

namespace tensor {

/**
 * Tensor - 张量类
 *
 * 特点：
 * - 使用 TensorStorage 抽象内存管理
 * - 支持多设备 (CPU, NPU, TPU)
 * - 支持非连续视图 (slice, transpose)
 * - 支持 stride-based 索引
 */
class Tensor {
public:
    // ========== 构造函数 ==========

    /**
     * 从数据创建（拷贝数据到 CPU Storage）
     */
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape);

    /**
     * 从原始指针创建（零拷贝引用或拷贝）
     * @param owner 如果提供，则共享所有权；否则拷贝数据
     */
    Tensor(const float* data, const std::vector<int64_t>& shape,
           std::shared_ptr<TensorStorage> owner = nullptr);

    /**
     * 内部构造（支持 strides 和 offset）
     */
    Tensor(std::shared_ptr<TensorStorage> storage,
           const std::vector<int64_t>& shape,
           const std::vector<int64_t>& strides,
           int64_t offset,
           bool contiguous);

    /**
     * Lua 工厂方法 - 显式处理 Lua table 转换
     */
    static Tensor create(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /**
     * Lua 工厂方法 - 从 LuaRef 表创建（手动解析）
     */
    static Tensor from_lua(lua_State* L, const LuaIntf::LuaRef& data_table, const LuaIntf::LuaRef& shape_table);

    /**
     * Lua 包装方法 - get 单索引
     */
    float get_lua(int idx) const;

    /**
     * Lua 包装方法 - set 单索引
     */
    void set_lua(int idx, float value);

    /**
     * Lua 包装方法 - 2D 元素访问
     */
    float at2d(int64_t i, int64_t j) const { return at(i, j); }

    /**
     * 提取指定列为新 Tensor (自动从 Lua table 转换)
     */
    Tensor extract_columns_tensor(const std::vector<int64_t>& cols) const;

    // ========== 属性访问 ==========

    std::vector<int64_t> shape() const { return shape_; }
    std::vector<int64_t> strides() const { return strides_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t size() const;
    int64_t size(int dim) const;
    bool is_contiguous() const { return contiguous_; }

    /**
     * 获取设备类型
     */
    DeviceType device() const { return storage_->device(); }

    /**
     * 获取底层存储
     */
    std::shared_ptr<TensorStorage> storage() const { return storage_; }

    // ========== 设备操作 ==========

    /**
     * 迁移到指定设备
     * @param device 目标设备
     * @return 新设备上的 Tensor（如果已在目标设备则返回自身）
     */
    Tensor to(DeviceType device) const;

    /**
     * 确保连续存储
     * @return 连续的 Tensor（如果已连续则返回自身）
     */
    Tensor contiguous() const;

    // ========== 异步设备操作 ==========

    /**
     * 异步迁移到指定设备
     * @param device 目标设备
     * @param stream 异步流（nullptr 表示使用默认流）
     * @return 新设备上的 Tensor（数据可能尚未就绪）
     *
     * 使用模式：
     *   auto npu_tensor = cpu_tensor.to_async(DeviceType::NPU, stream);
     *   // ... 其他操作 ...
     *   npu_tensor.sync(stream);  // 等待传输完成
     */
    Tensor to_async(DeviceType device, Stream* stream = nullptr) const;

    /**
     * 同步等待异步操作完成
     * @param stream 要同步的流（nullptr 表示默认流）
     */
    void sync(Stream* stream = nullptr) const;

    /**
     * 检查数据是否就绪（非阻塞）
     * @param stream 关联的流
     * @return true 如果数据已就绪
     */
    bool is_ready(Stream* stream = nullptr) const;

    // ========== 统一元素访问 ==========

    /**
     * at() - 自动处理 stride 的元素访问
     */
    float at(int64_t i) const;
    float at(int64_t i, int64_t j) const;
    float at(const std::vector<int64_t>& indices) const;

    float& at(int64_t i);
    float& at(int64_t i, int64_t j);

    // ========== Level 1: 基础形状操作 ==========

    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor select_dim(int dim, int64_t index) const;
    Tensor get_column(int64_t col_idx) const;
    Tensor slice_columns(int64_t start, int64_t end) const;

    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(const std::vector<int>& dims) const;
    Tensor transpose() const;

    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;

    // ========== Level 2: 数学运算 ==========

    Tensor add(const Tensor& other) const;
    Tensor add(float scalar) const;
    Tensor sub(const Tensor& other) const;
    Tensor sub(float scalar) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(float scalar) const;
    Tensor div(const Tensor& other) const;
    Tensor div(float scalar) const;

    Tensor sum(int axis = -1, bool keepdims = false) const;
    Tensor mean(int axis = -1, bool keepdims = false) const;
    Tensor max(int axis = -1, bool keepdims = false) const;
    Tensor min(int axis = -1, bool keepdims = false) const;

    LuaIntf::LuaRef argmax_lua(lua_State* L, int axis = -1) const;
    LuaIntf::LuaRef argmin_lua(lua_State* L, int axis = -1) const;

    Tensor sigmoid() const;
    Tensor softmax(int axis = -1) const;
    Tensor exp_() const;
    Tensor log_() const;

    Tensor gt(float threshold) const;
    Tensor lt(float threshold) const;
    Tensor ge(float threshold) const;
    Tensor le(float threshold) const;
    Tensor eq(float threshold) const;

    // ========== Level 3: 高级操作 ==========

    LuaIntf::LuaRef topk_lua(lua_State* L, int k, int axis = -1, bool largest = true) const;
    Tensor gather(int axis, const Tensor& indices) const;
    static Tensor concat(const std::vector<Tensor>& tensors, int axis);
    std::vector<Tensor> split(int num_splits, int axis) const;

    std::vector<int64_t> nonzero() const;
    std::vector<int64_t> where_indices(float threshold, const std::string& op = "ge") const;
    Tensor index_select(int dim, const std::vector<int64_t>& indices) const;

    // ========== Level 4: Legacy 方法 ==========

    LuaIntf::LuaRef filter_yolo(lua_State* L, float conf_thres);
    LuaIntf::LuaRef filter_yolo_pose(lua_State* L, float conf_thres);
    LuaIntf::LuaRef filter_yolo_seg(lua_State* L, float conf_thres);
    LuaIntf::LuaRef process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs,
                                  const LuaIntf::LuaRef& box,
                                  int img_w, int img_h,
                                  int input_w, int input_h,
                                  int pad_x, int pad_y);
    LuaIntf::LuaRef argmax(lua_State* L);
    LuaIntf::LuaRef topk(lua_State* L, int k);

    // ========== 辅助方法 ==========

    float get_item(const std::vector<int64_t>& indices) const;
    void set_item(const std::vector<int64_t>& indices, float value);

    LuaIntf::LuaRef to_table(lua_State* L) const;
    std::string to_string(int max_elements = 10) const;

    // 零拷贝视图（向后兼容）
    LuaIntf::TensorView<float> view();

    // 内部数据访问（仅 CPU 有效）
    const float* raw_data() const;
    float* raw_data();
    const float* data() const;
    float* data();

private:
    std::shared_ptr<TensorStorage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t offset_;
    bool contiguous_;

    // 内部辅助方法
    int64_t compute_offset(const std::vector<int64_t>& indices) const;
    int64_t compute_size() const;
    std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) const;
    void normalize_axis(int& axis) const;
    Tensor contiguous_copy() const;

    // 检查是否在 CPU 上
    void check_cpu() const;
};

} // namespace tensor
