-- YOLO11 Object Detection Script (使用新Tensor API)
-- 展示如何用通用Tensor操作替代filter_yolo()等专用方法
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

function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    local r = math.min(target_h / h, target_w / w)
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end

    local dw = target_w - new_w
    local dh = target_h - new_h
    
    dw = dw % Model.config.stride
    dh = dh % Model.config.stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    img:pad(top, bottom, left, right, 114)

    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- 使用向量化Tensor操作实现YOLO后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLO11格式: [1, 84, 8400]
    -- 前4个是box坐标(xywh), 后80个是类别分数
    local num_classes = 80

    -- 1. 提取box坐标和类别分数
    -- 关键优化：立即调用contiguous()确保后续操作高效
    local boxes = output:slice(1, 0, 4, 1):squeeze(0):contiguous()  -- [4, 8400]
    local scores = output:slice(1, 4, 84, 1):squeeze(0):contiguous()  -- [80, 8400]

    -- 2. 对每个box找到最大分数和对应类别
    local max_scores = scores:max(0, false)  -- [8400]
    local class_ids = scores:argmax(0)  -- Lua table [8400]

    -- 3. 向量化过滤：找出满足条件的索引（关键优化！）
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    if #valid_indices == 0 then
        return {}
    end

    -- 4. 批量提取有效数据（避免大规模to_table转换）
    -- extract_columns返回[4, N] tensor，转置后to_table得到[[cx,cy,w,h], ...]
    local filtered_boxes_tensor = boxes:extract_columns(valid_indices)  -- [4, N] tensor
    local filtered_boxes = filtered_boxes_tensor:transpose():to_table()  -- 转置后变成[N, 4]，再转table

    -- 对于1D tensor [8400]，使用index_select提取指定元素，然后转为table
    local filtered_scores_tensor = max_scores:index_select(0, valid_indices)  -- [num_valid]
    local filtered_scores = filtered_scores_tensor:to_table()  -- 现在数据量很小，转换快速

    local proposals = {}

    -- 5. 只遍历过滤后的小数据集（通常<100个）
    for i = 1, #valid_indices do
        local idx = valid_indices[i]
        local box_data = filtered_boxes[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]  -- Lua索引从1开始（C++索引）
        local conf = filtered_scores[i]  -- 使用过滤后的scores

        -- 将中心点坐标转换为左上角坐标
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

    -- 6. 坐标还原到原图
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
    end

    -- 7. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    return final_boxes
end

return Model
