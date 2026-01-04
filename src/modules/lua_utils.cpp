#include "lua_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace lua_utils {

float compute_iou(const Box& a, const Box& b) {
    // 转换为 x1, y1, x2, y2
    float a_x1 = a.x, a_y1 = a.y, a_x2 = a.x + a.w, a_y2 = a.y + a.h;
    float b_x1 = b.x, b_y1 = b.y, b_x2 = b.x + b.w, b_y2 = b.y + b.h;
    
    // 计算交集
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    // 计算并集
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

LuaIntf::LuaRef nms(LuaIntf::LuaRef proposals, float iou_thres) {
    lua_State* L = proposals.state();
    // 1. 从Lua table提取boxes
    std::vector<Box> boxes;
    int len = proposals.len();
    
    for (int i = 1; i <= len; ++i) {
        LuaIntf::LuaRef prop = proposals.rawget(i);
        
        if (!prop.isTable()) {
            continue;
        }
        
        Box box;
        box.x = prop.get<float>("x");
        box.y = prop.get<float>("y");
        box.w = prop.get<float>("w");
        box.h = prop.get<float>("h");
        box.score = prop.get<float>("score");
        
        // label可能是字符串，需要保存原始table
        boxes.push_back(box);
    }
    
    // 2. 按score降序排序
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&boxes](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });
    
    // 3. NMS算法
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep_indices;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep_indices.push_back(idx);
        
        // 抑制与当前box IoU过高的其他box
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            
            float iou = compute_iou(boxes[idx], boxes[idx_j]);
            if (iou > iou_thres) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    // 4. 构造结果table
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    for (int idx : keep_indices) {
        LuaIntf::LuaRef val = proposals.rawget(idx + 1);
        results.rawset(result_idx++, val);
    }
    
    return results;
}

void register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_utils")
            .addFunction("nms", &nms)
        .endModule();
}

} // namespace lua_utils
