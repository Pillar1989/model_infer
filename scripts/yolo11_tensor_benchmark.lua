-- YOLO11 Benchmark Script with Detailed Timing
-- 基于yolo11_tensor_detector.lua，添加详细的计时统计

local utils = lua_utils
local nn = lua_nn

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
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
    to_tensor_times = {},

    -- Postprocess
    slice_contiguous_times = {},
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
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    local dw = target_w - new_w
    local dh = target_h - new_h

    dw = dw % Model.config.stride
    dh = dh % Model.config.stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    -- Timing: letterbox (resize + pad are in-place operations)
    local t1 = os.clock()
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end
    img:pad(top, bottom, left, right, 114)
    local t2 = os.clock()
    table.insert(Model.timings.letterbox_times, (t2 - t1) * 1000)

    -- Timing: to_tensor (combines cvtColor + normalization + blob creation)
    local t3 = os.clock()
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})
    local t4 = os.clock()
    table.insert(Model.timings.to_tensor_times, (t4 - t3) * 1000)

    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLO11格式: [1, 84, 8400]
    local num_classes = 80

    -- Timing: slice + contiguous
    local t1 = os.clock()
    local boxes = output:slice(1, 0, 4, 1):squeeze(0):contiguous()  -- [4, 8400]
    local scores = output:slice(1, 4, 84, 1):squeeze(0):contiguous()  -- [80, 8400]
    local t2 = os.clock()
    table.insert(Model.timings.slice_contiguous_times, (t2 - t1) * 1000)

    -- Timing: max_with_argmax (fused)
    local t3 = os.clock()
    local result = scores:max_with_argmax(0)
    local max_scores = result.values
    local class_ids = result.indices
    local t4 = os.clock()
    table.insert(Model.timings.maxarg_times, (t4 - t3) * 1000)

    -- Timing: where_indices
    local t5 = os.clock()
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")
    local t6 = os.clock()
    table.insert(Model.timings.where_times, (t6 - t5) * 1000)

    if #valid_indices == 0 then
        table.insert(Model.timings.extract_times, 0)
        table.insert(Model.timings.loop_times, 0)
        table.insert(Model.timings.nms_times, 0)
        Model.iteration_count = Model.iteration_count + 1
        if Model.iteration_count % Model.print_every_n == 0 then
            Model.print_timing_summary()
        end
        return {}
    end

    -- Timing: extract_columns + index_select
    local t7 = os.clock()
    local filtered_boxes = boxes:extract_columns(valid_indices)
    local filtered_scores_tensor = max_scores:index_select(0, valid_indices)
    local filtered_scores = filtered_scores_tensor:to_table()
    local t8 = os.clock()
    table.insert(Model.timings.extract_times, (t8 - t7) * 1000)

    -- Timing: build proposals + coordinate transform
    local t9 = os.clock()
    local proposals = {}

    for i = 1, #valid_indices do
        local idx = valid_indices[i]
        local box_data = filtered_boxes[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]
        local conf = filtered_scores[i]

        local x = cx - w / 2.0
        local y = cy - h / 2.0

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = conf,
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown"
        })
    end

    -- Coordinate transform to original image
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
    end
    local t10 = os.clock()
    table.insert(Model.timings.loop_times, (t10 - t9) * 1000)

    -- Timing: NMS
    local t11 = os.clock()
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    local t12 = os.clock()
    table.insert(Model.timings.nms_times, (t12 - t11) * 1000)

    -- Increment iteration counter and print summary if needed
    Model.iteration_count = Model.iteration_count + 1
    if Model.iteration_count % Model.print_every_n == 0 then
        Model.print_timing_summary()
    end

    return final_boxes
end

-- Print timing summary
function Model.print_timing_summary()
    local function avg(t)
        if #t == 0 then return 0 end
        local sum = 0
        for _, v in ipairs(t) do
            sum = sum + v
        end
        return sum / #t
    end

    local n = #Model.timings.slice_contiguous_times
    if n == 0 then return end

    print("\n" .. string.rep("=", 70))
    print("DETAILED TIMING BREAKDOWN (YOLO11 - Average over " .. n .. " runs)")
    print(string.rep("=", 70))

    print("\nPREPROCESS (Lua → C++ calls):")
    print(string.format("  %-35s: %8.3f ms", "letterbox (resize + pad)", avg(Model.timings.letterbox_times)))
    print(string.format("  %-35s: %8.3f ms", "to_tensor (cvt+norm+blob)", avg(Model.timings.to_tensor_times)))

    local total_preprocess = avg(Model.timings.letterbox_times) +
                             avg(Model.timings.to_tensor_times)
    print(string.format("  %-35s: %8.3f ms", "TOTAL PREPROCESS", total_preprocess))

    print("\nPOSTPROCESS (Lua + Tensor API):")
    print(string.format("  %-35s: %8.3f ms", "slice + contiguous", avg(Model.timings.slice_contiguous_times)))
    print(string.format("  %-35s: %8.3f ms", "max_with_argmax (fused)", avg(Model.timings.maxarg_times)))
    print(string.format("  %-35s: %8.3f ms", "where_indices", avg(Model.timings.where_times)))
    print(string.format("  %-35s: %8.3f ms", "extract_columns + index_select", avg(Model.timings.extract_times)))
    print(string.format("  %-35s: %8.3f ms", "build proposals (Lua loop)", avg(Model.timings.loop_times)))
    print(string.format("  %-35s: %8.3f ms", "NMS", avg(Model.timings.nms_times)))
    print(string.rep("-", 70))

    local total_postprocess = avg(Model.timings.slice_contiguous_times) +
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
    print("      YOLO11 uses transpose-free slice operations for better performance")
    print(string.rep("=", 70) .. "\n")
end

return Model
