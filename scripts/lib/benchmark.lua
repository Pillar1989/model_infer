-- Benchmark Utilities
-- 提供计时统计和报告打印功能

local M = {}

-- 计算平均值
function M.avg(t)
    if #t == 0 then return 0 end
    local sum = 0
    for _, v in ipairs(t) do
        sum = sum + v
    end
    return sum / #t
end

-- 打印分隔线
local function print_separator(width)
    print(string.rep("=", width))
end

local function print_line(width)
    print(string.rep("-", width))
end

-- 打印计时项
local function print_timing_item(label, time_ms, width)
    width = width or 35
    print(string.format("  %-" .. width .. "s: %8.3f ms", label, time_ms))
end

-- 打印计时摘要
-- 参数:
--   model_name: 模型名称 (如 "YOLO11", "YOLOv5")
--   timings: 计时数据表
--   config: 配置表，包含:
--     - count_key: 用于计数的timing key (如 "slice_contiguous_times")
--     - preprocess_items: 预处理项列表 {{label, key}, ...}
--     - postprocess_items: 后处理项列表
--     - notes: 额外注释 (可选)
function M.print_timing_summary(model_name, timings, config)
    local avg = M.avg
    local width = 70

    local n = #timings[config.count_key]
    if n == 0 then return end

    print_separator(width)
    print("DETAILED TIMING BREAKDOWN (" .. model_name .. " - Average over " .. n .. " runs)")
    print_separator(width)

    -- 预处理部分
    print("\nPREPROCESS (Lua → C++ calls):")
    local total_preprocess = 0
    for _, item in ipairs(config.preprocess_items) do
        local label, key = item[1], item[2]
        local time = avg(timings[key])
        print_timing_item(label, time)
        total_preprocess = total_preprocess + time
    end
    print_timing_item("TOTAL PREPROCESS", total_preprocess)

    -- 后处理部分
    print("\nPOSTPROCESS (Lua + Tensor API):")
    local total_postprocess = 0
    for _, item in ipairs(config.postprocess_items) do
        local label, keys = item[1], item[2]
        local time = 0
        -- 支持单个key或多个key相加
        if type(keys) == "table" then
            for _, k in ipairs(keys) do
                time = time + avg(timings[k])
            end
        else
            time = avg(timings[keys])
        end
        print_timing_item(label, time)
        total_postprocess = total_postprocess + time
    end
    print_line(width)
    print_timing_item("TOTAL POSTPROCESS", total_postprocess)

    -- 总计
    print("\nOVERALL:")
    print_timing_item("Preprocess (Lua→C++)", total_preprocess)
    print_timing_item("Postprocess (Lua+Tensor)", total_postprocess)
    print_timing_item("TOTAL Lua Overhead", total_preprocess + total_postprocess)

    -- 注释
    print_separator(width)
    if config.notes then
        print("\nNOTE: Image Load and ONNX Inference times are shown by main.cpp")
        for _, note in ipairs(config.notes) do
            print("      " .. note)
        end
    end
    print_separator(width)
    print()
end

return M
