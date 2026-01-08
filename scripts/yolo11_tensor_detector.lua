-- YOLO11 Object Detection Script (使用新Tensor API)
-- 展示如何用通用Tensor操作替代filter_yolo()等专用方法
local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("scripts.lib.preprocess")
local coco_labels = require("scripts.lib.coco")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = coco_labels  -- 使用公共库中的COCO labels
}

function Model.preprocess(img)
    -- 使用公共库的letterbox函数
    return preprocess_lib.letterbox(img, Model.config.input_size, Model.config.stride)
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

    -- 2. 对每个box找到最大分数和对应类别 (融合操作，单次遍历)
    local result = scores:max_with_argmax(0)  -- {values=Tensor[8400], indices=table[8400]}
    local max_scores = result.values
    local class_ids = result.indices

    -- 3. 向量化过滤：找出满足条件的索引（关键优化！）
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    if #valid_indices == 0 then
        return {}
    end

    -- 4. 批量提取有效数据（避免大规模to_table转换）
    -- extract_columns 直接返回行格式 {{cx,cy,w,h}, ...}
    local filtered_boxes = boxes:extract_columns(valid_indices)

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

    -- 6. 坐标还原到原图（使用公共库函数）
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size(box.w, meta)
        box.h = preprocess_lib.scale_size(box.h, meta)
    end

    -- 7. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    return final_boxes
end

return Model
