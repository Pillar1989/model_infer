-- YOLOv5 Object Detection Script (纯Tensor API实现 - 使用修复后的API)
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

-- 使用纯Tensor API实现YOLOv5后处理
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLOv5格式: [1, 25200, 85]
    -- 85 = 4(xywh) + 1(objectness) + 80(classes)
    local data = output:squeeze(0)  -- [25200, 85]

    -- 提取各部分
    local boxes_x = data:get_column(0)      -- [25200]
    local boxes_y = data:get_column(1)      -- [25200]
    local boxes_w = data:get_column(2)      -- [25200]
    local boxes_h = data:get_column(3)      -- [25200]
    local objectness = data:get_column(4)   -- [25200]

    -- 提取类别分数 [25200, 80]
    local class_scores = data:slice_columns(5, 85)

    -- 1. 对每个box找最大类别分数和类别ID (融合操作，单次遍历)
    local result = class_scores:max_with_argmax(1)  -- {values=Tensor[25200], indices=table[25200]}
    local max_class_scores = result.values
    local class_ids = result.indices

    local proposals = {}
    local num_boxes = 25200

    -- 2. 遍历所有boxes，进行过滤（YOLOv5需要两次过滤）
    for i = 0, num_boxes - 1 do
        local obj = objectness:get(i)

        -- 第一次过滤：objectness 阈值
        if obj < Model.config.conf_thres then
            goto continue
        end

        local max_cls_score = max_class_scores:get(i)

        -- 计算最终分数 = objectness * max_class_score
        local final_score = obj * max_cls_score

        -- 第二次过滤：final_score 阈值
        if final_score >= Model.config.conf_thres then
            local cx = boxes_x:get(i)
            local cy = boxes_y:get(i)
            local w = boxes_w:get(i)
            local h = boxes_h:get(i)
            local cls_id = class_ids[i + 1]  -- Lua table 从1开始索引

            -- 中心点转左上角
            local x = cx - w / 2.0
            local y = cy - h / 2.0

            -- 坐标还原到原图
            x = (x - meta.pad_x) / meta.scale
            y = (y - meta.pad_y) / meta.scale
            w = w / meta.scale
            h = h / meta.scale

            table.insert(proposals, {
                x = x,
                y = y,
                w = w,
                h = h,
                score = final_score,
                class_id = cls_id,
                label = Model.config.labels[cls_id + 1] or "unknown"
            })
        end

        ::continue::
    end

    -- 3. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)

    return final_boxes
end

return Model
