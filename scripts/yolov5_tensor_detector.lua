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

-- 使用通用Tensor操作实现YOLOv5后处理
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]
    
    if not output then 
        error("Missing output0") 
    end

    -- YOLOv5格式: [1, 25200, 85]
    -- 85 = 4(xywh) + 1(objectness) + 80(classes)
    -- 先squeeze掉batch维度 -> [25200, 85]
    local data = output:squeeze(0)
    
    -- 使用实际长度
    local data_table = data:to_table()
    local actual_num_boxes = #data_table
    
    local proposals = {}
    
    -- 遍历每个box
    for i = 1, actual_num_boxes do
        local box_data = data_table[i]
        
        -- 提取坐标和objectness
        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local objectness = box_data[5]
        
        -- 预先过滤objectness
        if objectness >= Model.config.conf_thres then
            -- 找到最大类别分数
            local max_class_score = box_data[6]
            local max_class_id = 0
            
            for c = 1, 79 do  -- 剩余79个类别
                local score = box_data[5 + c + 1]
                if score > max_class_score then
                    max_class_score = score
                    max_class_id = c
                end
            end
            
            -- 计算最终分数 = objectness * class_score
            local final_score = objectness * max_class_score
            
            if final_score >= Model.config.conf_thres then
                -- 将中心点坐标转换为左上角坐标
                local x = cx - w / 2.0
                local y = cy - h / 2.0
                
                table.insert(proposals, {
                    x = x,
                    y = y,
                    w = w,
                    h = h,
                    score = final_score,
                    class_id = max_class_id,
                    label = Model.config.labels[max_class_id + 1] or "unknown"
                })
            end
        end
    end
    
    print(string.format("过滤后候选框: %d", #proposals))
    
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
