#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// ============ Memory Monitoring Utilities ============

struct MemoryInfo {
    size_t vm_rss_kb = 0;   // Physical memory (KB)
    size_t vm_size_kb = 0;  // Virtual memory (KB)

    void update() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.find("VmRSS:") == 0) {
                sscanf(line.c_str(), "VmRSS: %zu", &vm_rss_kb);
            } else if (line.find("VmSize:") == 0) {
                sscanf(line.c_str(), "VmSize: %zu", &vm_size_kb);
            }
        }
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "RSS=" << std::fixed << std::setprecision(1)
            << (vm_rss_kb / 1024.0) << "MB, "
            << "VM=" << (vm_size_kb / 1024.0) << "MB";
        return oss.str();
    }
};

// ============ File Utilities ============

inline bool is_video_file(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
            ext == "flv" || ext == "wmv" || ext == "m4v");
}


// COCO labels
const std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
