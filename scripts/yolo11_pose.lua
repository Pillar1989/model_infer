-- YOLO11 Pose Estimation Script
-- local utils = require "lua_utils" -- lua_utils is registered globally

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = {"person"}
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
    local output_tensor = nil
    for k, v in pairs(outputs) do
        output_tensor = v
        break
    end
    
    if not output_tensor then error("No output tensor") end

    -- Use new C++ method for Pose
    local raw_boxes = output_tensor:filter_yolo_pose(Model.config.conf_thres)

    local proposals = {}

    for _, box in ipairs(raw_boxes) do
        -- Restore Box
        local x = (box.x - meta.pad_x) / meta.scale
        local y = (box.y - meta.pad_y) / meta.scale
        local w = box.w / meta.scale
        local h = box.h / meta.scale
        
        -- Restore Keypoints
        local kpts = box.keypoints
        for i = 1, #kpts do
            kpts[i].x = (kpts[i].x - meta.pad_x) / meta.scale
            kpts[i].y = (kpts[i].y - meta.pad_y) / meta.scale
        end
        
        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            class_id = box.class_id,
            label = "person",
            keypoints = kpts
        })
    end

    -- NMS
    local final_boxes = lua_utils.nms(proposals, Model.config.iou_thres)
    
    return final_boxes
end

return Model