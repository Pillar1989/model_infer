-- Tensor API单元测试脚本
-- 这个脚本可以在没有实际图片的情况下测试Tensor API

local cv = require("lua_cv")
local nn = require("lua_nn")
local utils = require("lua_utils")

-- 模型配置（虽然我们不会真正使用）
local Model = {
    config = {
        input_size = {640, 640},
        conf_thres = 0.25,
        iou_thres = 0.45,
    }
}

-- 主测试函数
function Model.test_tensor_api()
    print("========== Tensor通用API测试 ==========\n")

    -- 测试1: 基础构造和属性
    print("测试1: 基础构造和属性")
    local t1 = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    print("t1 shape:", table.concat(t1:shape(), ", "))
    print("t1 ndim:", t1.ndim)
    print("t1 size:", t1:size())
    print("t1:", tostring(t1))
    print()

    -- 测试2: Slice操作
    print("测试2: Slice操作")
    local t2 = nn.Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4})
    print("原始tensor:", tostring(t2))
    local t2_sliced = t2:slice(0, 0, 2)
    print("切片后(前2行):", tostring(t2_sliced))
    print("切片shape:", table.concat(t2_sliced:shape(), ", "))
    print()

    -- 测试3: Reshape操作
    print("测试3: Reshape操作")
    local t3 = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    print("原始shape:", table.concat(t3:shape(), ", "))
    local t3_reshaped = t3:reshape({3, 2})
    print("Reshape后:", tostring(t3_reshaped))
    print("新shape:", table.concat(t3_reshaped:shape(), ", "))
    print()

    -- 测试4: Transpose操作
    print("测试4: Transpose操作")
    local t4 = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    print("原始tensor:", tostring(t4))
    local t4_transposed = t4:transpose()
    print("转置后:", tostring(t4_transposed))
    print("转置shape:", table.concat(t4_transposed:shape(), ", "))
    print()

    -- 测试5: Squeeze/Unsqueeze
    print("测试5: Squeeze/Unsqueeze")
    local t5 = nn.Tensor({1, 2, 3, 4}, {1, 4, 1})
    print("原始shape:", table.concat(t5:shape(), ", "))
    local t5_squeezed = t5:squeeze(-1)
    print("Squeeze后shape:", table.concat(t5_squeezed:shape(), ", "))
    local t5_unsqueezed = t5_squeezed:unsqueeze(0)
    print("Unsqueeze后shape:", table.concat(t5_unsqueezed:shape(), ", "))
    print()

    -- 测试6: Element-wise操作
    print("测试6: Element-wise操作")
    local t6 = nn.Tensor({1, 2, 3, 4}, {4})
    print("原始tensor:", tostring(t6))
    local t6_add = t6:add(10)
    print("加10:", tostring(t6_add))
    local t6_mul = t6:mul(2)
    print("乘2:", tostring(t6_mul))
    print()

    -- 测试7: Reduction操作
    print("测试7: Reduction操作")
    local t7 = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    print("原始tensor:", tostring(t7))
    local t7_sum = t7:sum(-1)
    print("总和:", tostring(t7_sum))
    local t7_mean = t7:mean(-1)
    print("平均值:", tostring(t7_mean))
    local t7_max = t7:max(-1)
    print("最大值:", tostring(t7_max))
    print()

    -- 测试8: Argmax/Argmin
    print("测试8: Argmax/Argmin")
    local t8 = nn.Tensor({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    print("原始tensor:", tostring(t8))
    local max_idx = t8:argmax(-1)
    print("最大值索引:", max_idx)
    local min_idx = t8:argmin(-1)
    print("最小值索引:", min_idx)
    print()

    -- 测试9: Sigmoid和Softmax
    print("测试9: Activation函数")
    local t9 = nn.Tensor({-1, 0, 1, 2}, {4})
    print("原始tensor:", tostring(t9))
    local t9_sigmoid = t9:sigmoid()
    print("Sigmoid:", tostring(t9_sigmoid))
    local t9_softmax = t9:softmax(-1)
    print("Softmax:", tostring(t9_softmax))
    print()

    -- 测试10: 比较操作
    print("测试10: 比较操作")
    local t10 = nn.Tensor({0.1, 0.5, 0.8, 0.3}, {4})
    print("原始tensor:", tostring(t10))
    local t10_gt = t10:gt(0.4)
    print("大于0.4的mask:", tostring(t10_gt))
    local t10_le = t10:le(0.5)
    print("小于等于0.5的mask:", tostring(t10_le))
    print()

    -- 测试11: TopK
    print("测试11: TopK操作")
    local t11 = nn.Tensor({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    print("原始tensor:", tostring(t11))
    local topk_result = t11:topk_new(3, -1, true)
    print("Top 3 values:")
    for i = 1, 3 do
        print(string.format("  [%d] value=%.1f, index=%d", i, topk_result.values[i], topk_result.indices[i]))
    end
    print()

    -- 测试12: to_table转换
    print("测试12: to_table转换")
    local t12 = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    local table_result = t12:to_table()
    print("转换为Lua表:")
    for i, row in ipairs(table_result) do
        local row_str = ""
        for j, val in ipairs(row) do
            row_str = row_str .. string.format("%.1f ", val)
        end
        print("  Row " .. i .. ": [" .. row_str .. "]")
    end
    print()

    -- 测试13: 组合操作（模拟简单的后处理）
    print("测试13: 组合操作示例")
    local logits = nn.Tensor({2.1, 0.5, 3.2, 1.8, 0.1, 4.5, 1.2, 0.8, 2.7, 1.5}, {1, 10})
    print("原始logits:", tostring(logits))
    
    local logits_1d = logits:squeeze(0)
    print("Squeeze后:", tostring(logits_1d))
    
    local probs = logits_1d:softmax(-1)
    print("Softmax后的概率:", tostring(probs))
    
    local top3 = probs:topk_new(3, -1, true)
    print("Top 3分类结果:")
    for i = 1, 3 do
        print(string.format("  类别%d: 概率=%.4f", top3.indices[i], top3.values[i]))
    end
    print()

    print("========== 所有测试完成! ==========")
    
    -- 返回空结果（满足框架要求）
    return {}
end

-- 模拟预处理（返回一个假的tensor）
function Model.preprocess(image_path, session)
    print("跳过图片加载，直接运行Tensor API测试...\n")
    return nil
end

-- 模拟推理
function Model.inference(input_tensor, session)
    return nil
end

-- 使用后处理函数来运行测试
function Model.postprocess(outputs, meta)
    return Model.test_tensor_api()
end

-- 主入口：直接运行测试
if _G.ARG_MODEL_PATH and _G.ARG_IMAGE_PATH then
    -- 如果从main.cpp调用，这些全局变量会被设置
    print("参数:", _G.ARG_MODEL_PATH, _G.ARG_IMAGE_PATH)
    print()
    Model.test_tensor_api()
else
    -- 直接作为脚本运行
    Model.test_tensor_api()
end

return Model
