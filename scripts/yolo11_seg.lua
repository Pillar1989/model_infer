-- YOLO11 Instance Segmentation Script
-- local utils = require "lua_utils"
-- local nn = require "lua_nn"
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
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"
        -- ... (full COCO labels)
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

function Model.postprocess(outputs, meta)
    -- outputs: output0 (boxes+masks_coeffs), output1 (proto_masks)
    local output0 = outputs["output0"]
    local output1 = outputs["output1"]
    
    if not output0 or not output1 then error("Missing outputs") end

    -- 1. Filter Boxes
    local raw_boxes = output0:filter_yolo_seg(Model.config.conf_thres)

    local proposals = {}

    for _, box in ipairs(raw_boxes) do
        -- Restore Box
        local x = (box.x - meta.pad_x) / meta.scale
        local y = (box.y - meta.pad_y) / meta.scale
        local w = box.w / meta.scale
        local h = box.h / meta.scale
        
        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            class_id = box.class_id,
            label = Model.config.labels[box.class_id + 1] or "unknown",
            mask_coeffs = box.mask_coeffs
        })
    end

    -- 2. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    -- 3. Process Masks for final boxes
    for i = 1, #final_boxes do
        local box = final_boxes[i]
        -- Generate mask using C++ helper
        -- Note: We pass the original image size to get a mask of that size
        local mask_tensor = output1:process_mask(box.mask_coeffs, box, meta.ori_w, meta.ori_h)
        box.mask = mask_tensor
        box.mask_coeffs = nil -- Clean up
    end
    
    return final_boxes
end

return Model