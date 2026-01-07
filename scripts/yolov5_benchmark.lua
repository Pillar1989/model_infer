-- YOLOv5 Benchmark Script with Detailed Timing
-- 基于 yolov5_tensor_detector.lua，添加精确计时统计

local utils = lua_utils
local nn = lua_nn

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    }
}

-- Timing accumulators
Model.timings = {
    -- Preprocess
    letterbox_times = {},
    cvtcolor_times = {},
    to_blob_times = {},

    -- Postprocess
    slice_times = {},
    contiguous_times = {},
    maxarg_times = {},
    where_times = {},
    extract_times = {},
    loop_times = {},
    nms_times = {}
}

-- Iteration counter for benchmarking
Model.iteration_count = 0
Model.print_every_n = 1  -- Print timing summary after each iteration for benchmarking

function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    local r = math.min(target_h / h, target_w / w)
    local new_h, new_w = math.floor(h * r + 0.5), math.floor(w * r + 0.5)
    local pad_h, pad_w = target_h - new_h, target_w - new_w

    local top = math.floor(pad_h / 2)
    local bottom = pad_h - top
    local left = math.floor(pad_w / 2)
    local right = pad_w - left

    -- Timing: letterbox (resize + pad)
    local t1 = os.clock()
    img:resize(new_w, new_h)
    img:pad(top, bottom, left, right, 114)
    local t2 = os.clock()
    table.insert(Model.timings.letterbox_times, (t2 - t1) * 1000)

    -- Timing: to_tensor (combines cvtColor + normalization + blob creation)
    local t3 = os.clock()
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})
    local t4 = os.clock()
    table.insert(Model.timings.cvtcolor_times, (t4 - t3) * 1000)  -- Reuse cvtcolor timing for to_tensor
    table.insert(Model.timings.to_blob_times, 0)  -- No separate to_blob in YOLOv5

    local meta = {
        scale = r,
        pad = {top, left},
        orig_shape = {w, h}
    }

    return input_tensor, meta
end

function Model.postprocess(outputs, meta)
    local output = outputs["output0"]:squeeze(0)  -- [25200, 85]

    -- Timing: extract columns
    local t1 = os.clock()
    local boxes_x = output:get_column(0)
    local boxes_y = output:get_column(1)
    local boxes_w = output:get_column(2)
    local boxes_h = output:get_column(3)
    local objectness = output:get_column(4)
    local class_scores = output:slice_columns(5, 85)  -- [25200, 80]
    local t2 = os.clock()
    table.insert(Model.timings.slice_times, (t2 - t1) * 1000)

    -- Timing: max_with_argmax (fused)
    local t3 = os.clock()
    local result = class_scores:max_with_argmax(1)
    local max_class_scores = result.values  -- [25200]
    local class_ids = result.indices  -- table[25200]
    local t4 = os.clock()
    table.insert(Model.timings.maxarg_times, (t4 - t3) * 1000)

    -- Note: contiguous timing is not applicable for YOLOv5 column-based approach
    table.insert(Model.timings.contiguous_times, 0)

    -- Timing: filtering + proposal building (combined in Lua loop for YOLOv5)
    local t5 = os.clock()
    local proposals = {}
    local num_boxes = 25200
    local scale = meta.scale
    local pad_top, pad_left = table.unpack(meta.pad)

    for i = 0, num_boxes - 1 do
        local obj = objectness:get(i)
        if obj >= Model.config.conf_thres then
            local max_cls_score = max_class_scores:get(i)
            local final_score = obj * max_cls_score

            if final_score >= Model.config.conf_thres then
                local cx = boxes_x:get(i)
                local cy = boxes_y:get(i)
                local w = boxes_w:get(i)
                local h = boxes_h:get(i)
                local cls_id = class_ids[i + 1]

                -- Convert center to top-left and scale back to original coordinates
                local x = (cx - w / 2 - pad_left) / scale
                local y = (cy - h / 2 - pad_top) / scale
                local scaled_w = w / scale
                local scaled_h = h / scale

                table.insert(proposals, {
                    x = x,
                    y = y,
                    w = scaled_w,
                    h = scaled_h,
                    score = final_score,
                    class_id = cls_id,
                    label = Model.config.labels[cls_id + 1] or "unknown"
                })
            end
        end
    end
    local t6 = os.clock()
    local loop_time = (t6 - t5) * 1000

    -- Split loop time into where/extract/loop for compatibility
    table.insert(Model.timings.where_times, loop_time * 0.3)
    table.insert(Model.timings.extract_times, loop_time * 0.2)
    table.insert(Model.timings.loop_times, loop_time * 0.5)

    if #proposals == 0 then
        table.insert(Model.timings.nms_times, 0)
        return {}
    end

    -- Timing: NMS
    local t7 = os.clock()
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    local t8 = os.clock()
    table.insert(Model.timings.nms_times, (t8 - t7) * 1000)

    -- Increment iteration counter and print summary if needed
    Model.iteration_count = Model.iteration_count + 1
    if Model.iteration_count % Model.print_every_n == 0 then
        Model.print_timing_summary()
    end

    return final_boxes
end

-- Print detailed timing summary
function Model.print_timing_summary()
    local function avg(t)
        if #t == 0 then return 0 end
        local sum = 0
        for _, v in ipairs(t) do
            sum = sum + v
        end
        return sum / #t
    end

    local n = #Model.timings.slice_times
    if n == 0 then return end

    print("\n" .. string.rep("=", 70))
    print("DETAILED TIMING BREAKDOWN (YOLOv5 - Average over " .. n .. " runs)")
    print(string.rep("=", 70))

    print("\nPREPROCESS (Lua → C++ calls):")
    print(string.format("  %-35s: %8.3f ms", "letterbox (resize + pad)", avg(Model.timings.letterbox_times)))
    print(string.format("  %-35s: %8.3f ms", "to_tensor (cvt+norm+blob)", avg(Model.timings.cvtcolor_times)))

    local total_preprocess = avg(Model.timings.letterbox_times) +
                             avg(Model.timings.cvtcolor_times)
    print(string.format("  %-35s: %8.3f ms", "TOTAL PREPROCESS", total_preprocess))

    print("\nPOSTPROCESS (Lua + Tensor API):")
    print(string.format("  %-35s: %8.3f ms", "get_column + slice_columns", avg(Model.timings.slice_times)))
    print(string.format("  %-35s: %8.3f ms", "max_with_argmax (fused)", avg(Model.timings.maxarg_times)))
    print(string.format("  %-35s: %8.3f ms", "filter + build proposals (Lua)", avg(Model.timings.where_times) + avg(Model.timings.extract_times) + avg(Model.timings.loop_times)))
    print(string.format("  %-35s: %8.3f ms", "NMS", avg(Model.timings.nms_times)))

    local total_postprocess = avg(Model.timings.slice_times) +
                              avg(Model.timings.maxarg_times) +
                              avg(Model.timings.where_times) +
                              avg(Model.timings.extract_times) +
                              avg(Model.timings.loop_times) +
                              avg(Model.timings.nms_times)
    print(string.format("  %-35s: %8.3f ms", "TOTAL POSTPROCESS", total_postprocess))

    print("\nOVERALL:")
    print(string.format("  %-35s: %8.3f ms", "Preprocess (Lua→C++)", total_preprocess))
    print(string.format("  %-35s: %8.3f ms", "Postprocess (Lua+Tensor)", total_postprocess))
    print(string.format("  %-35s: %8.3f ms", "TOTAL Lua Overhead", total_preprocess + total_postprocess))

    print(string.rep("=", 70))
    print("\nNOTE: Image Load and ONNX Inference times are shown by main.cpp")
    print("      Compare with cpp_infer baseline (pure C++ YOLOv5)")
    print("      to_tensor combines cvtColor + normalization + blob creation")
    print(string.rep("=", 70) .. "\n")
end

return Model
