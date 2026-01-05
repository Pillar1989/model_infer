-- YOLOv5 Object Detection Script (使用新Tensor API)
-- YOLOv5格式: [1, 25200, 85] = [batch, num_boxes, 4_coords + 1_obj + 80_classes]
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

-- 使用向量化Tensor操作实现YOLOv5后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLOv5格式: [1, 25200, 85] = [batch, num_boxes, 4_coords + 1_obj + 80_classes]
    -- 先squeeze掉batch维度 -> [25200, 85]
    local data = output:squeeze(0)

    -- 1. 提取不同部分（沿axis=1切片）
    local boxes = data:slice(1, 0, 4, 1)        -- [25200, 4]
    local objectness = data:slice(1, 4, 5, 1):squeeze(1)  -- [25200]
    local class_scores = data:slice(1, 5, 85, 1)  -- [25200, 80]

    -- 2. 预先用objectness过滤（第一轮过滤）
    local obj_valid_indices = objectness:where_indices(Model.config.conf_thres, "ge")

    if #obj_valid_indices == 0 then
        print("过滤后候选框: 0")
        return {}
    end

    -- 3. 只对通过objectness过滤的boxes进行类别分析
    local filtered_boxes_tensor = boxes:index_select(0, obj_valid_indices)  -- [M, 4]
    local filtered_obj_tensor = objectness:index_select(0, obj_valid_indices)  -- [M]
    local filtered_class_scores = class_scores:index_select(0, obj_valid_indices)  -- [M, 80]

    -- 4. 找到每个box的最大类别分数
    local max_class_scores = filtered_class_scores:max(1, false)  -- [M] 沿类别维度取最大值
    local class_ids = filtered_class_scores:argmax(1)  -- Lua table [M]

    -- 5. 计算最终分数（objectness * class_score），再次过滤
    local final_scores_tensor = filtered_obj_tensor:mul_tensor(max_class_scores)  -- [M]
    local final_valid_indices = final_scores_tensor:where_indices(Model.config.conf_thres, "ge")

    print(string.format("过滤后候选框: %d", #final_valid_indices))

    if #final_valid_indices == 0 then
        return {}
    end

    -- 6. 最终数据提取（数据量已经很小）
    local final_boxes = filtered_boxes_tensor:index_select(0, final_valid_indices):to_table()
    local final_scores = final_scores_tensor:index_select(0, final_valid_indices):to_table()

    local proposals = {}

    -- 7. 构建proposals
    for i = 1, #final_valid_indices do
        local box_data = final_boxes[i]
        local idx = final_valid_indices[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]  -- Lua索引从1开始

        -- 将中心点坐标转换为左上角坐标
        local x = cx - w / 2.0
        local y = cy - h / 2.0

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = final_scores[i],
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown"
        })
    end
    
    -- 坐标还原到原图
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
    end
    
    -- NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    return final_boxes
end

return Model
