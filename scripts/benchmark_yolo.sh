#!/bin/bash
# YOLO Performance Benchmark Script
# Usage: ./scripts/benchmark_yolo.sh [iterations]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"
IMAGE_DIR="$PROJECT_DIR/images"

# Default iterations
ITERATIONS=${1:-3}

# Check if build exists
if [ ! -f "$BUILD_DIR/model_infer" ]; then
    echo "Error: model_infer not found. Please build the project first."
    echo "  mkdir build && cd build && cmake .. && make"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  YOLO Performance Benchmark"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Date: $(date)"
echo ""

# Test configurations: script,model,description
declare -a TESTS=(
    "yolo11_tensor_detector.lua,yolo11n.onnx,YOLO11n Detection (Tensor API)"
    "yolo11_tensor_pose.lua,yolo11n-pose.onnx,YOLO11n Pose (Tensor API)"
    "yolo11_tensor_seg.lua,yolo11n-seg.onnx,YOLO11n Segmentation (Tensor API)"
    "yolov5_tensor_detector.lua,yolov5n.onnx,YOLOv5n Detection (Tensor API)"
    "yolo11_detector.lua,yolo11n.onnx,YOLO11n Detection (Legacy)"
    "yolo11_pose.lua,yolo11n-pose.onnx,YOLO11n Pose (Legacy)"
    "yolo11_seg.lua,yolo11n-seg.onnx,YOLO11n Segmentation (Legacy)"
    "yolov5_detector.lua,yolov5n.onnx,YOLOv5n Detection (Legacy)"
)

# Results file
RESULTS_FILE="$PROJECT_DIR/benchmark_results.txt"
echo "Benchmark Results - $(date)" > "$RESULTS_FILE"
echo "Iterations: $ITERATIONS" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

# Function to measure execution time and extract detection count
measure_time_and_accuracy() {
    local cmd="$1"
    local output_file=$(mktemp)

    local start_ns=$(date +%s%N)
    eval "$cmd" > "$output_file" 2>&1
    local exit_code=$?
    local end_ns=$(date +%s%N)

    if [ $exit_code -ne 0 ]; then
        rm -f "$output_file"
        echo "-1,0"
        return 1
    fi

    # Calculate time in milliseconds
    local elapsed_ns=$((end_ns - start_ns))
    local elapsed_ms=$((elapsed_ns / 1000000))

    # Extract detection count from output
    # Looking for patterns like "Total: N detections" or "NMS后最终框: N"
    local detection_count=0
    if grep -q "Total:.*detections" "$output_file"; then
        detection_count=$(grep "Total:.*detections" "$output_file" | grep -oP '\d+(?= detections)')
    elif grep -q "NMS后最终框:" "$output_file"; then
        detection_count=$(grep "NMS后最终框:" "$output_file" | grep -oP '(?<=: )\d+')
    fi

    rm -f "$output_file"
    echo "$elapsed_ms,$detection_count"
}

# Run tests
echo "Running benchmarks..."
echo ""

for test_config in "${TESTS[@]}"; do
    IFS=',' read -r script model description <<< "$test_config"

    script_path="$SCRIPT_DIR/$script"
    model_path="$MODEL_DIR/$model"
    image_path="$IMAGE_DIR/zidane.jpg"

    # Check if files exist
    if [ ! -f "$script_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $description - Script not found: $script"
        continue
    fi
    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $description - Model not found: $model"
        continue
    fi

    echo -e "${BLUE}[TEST]${NC} $description"

    # Warmup run (not counted)
    "$BUILD_DIR/model_infer" "$script_path" "$model_path" "$image_path" > /dev/null 2>&1

    total_time=0
    success=true
    times=()
    detections=()
    cmd="\"$BUILD_DIR/model_infer\" \"$script_path\" \"$model_path\" \"$image_path\""

    for i in $(seq 1 $ITERATIONS); do
        result=$(measure_time_and_accuracy "$cmd")
        IFS=',' read -r time_ms det_count <<< "$result"

        if [ "$time_ms" = "-1" ]; then
            echo -e "${RED}[FAIL]${NC} Iteration $i failed"
            success=false
            break
        fi

        times+=("$time_ms")
        detections+=("$det_count")
        total_time=$((total_time + time_ms))
    done

    if [ "$success" = true ] && [ ${#times[@]} -gt 0 ]; then
        avg_time=$((total_time / ${#times[@]}))
        min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
        max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)

        # Get first detection count (should be consistent across runs)
        det_count="${detections[0]}"

        # Check if detection count is consistent
        det_consistent=true
        for d in "${detections[@]}"; do
            if [ "$d" != "$det_count" ]; then
                det_consistent=false
                break
            fi
        done

        if [ "$det_consistent" = true ]; then
            echo -e "${GREEN}[PASS]${NC} Avg: ${avg_time}ms, Min: ${min_time}ms, Max: ${max_time}ms | Detections: ${det_count}"
            echo "$description: avg=${avg_time}ms min=${min_time}ms max=${max_time}ms detections=${det_count}" >> "$RESULTS_FILE"
        else
            echo -e "${YELLOW}[WARN]${NC} Avg: ${avg_time}ms | Detections: inconsistent ${detections[*]}"
            echo "$description: avg=${avg_time}ms detections=INCONSISTENT" >> "$RESULTS_FILE"
        fi
    else
        echo "$description: FAILED" >> "$RESULTS_FILE"
    fi
    echo ""
done

echo "========================================"
echo "Results saved to: $RESULTS_FILE"
echo ""

# Also run C++ baseline if available
if [ -f "$BUILD_DIR/cpp_infer" ]; then
    echo -e "${BLUE}[TEST]${NC} C++ Baseline (Pure C++, YOLOv5n only)"

    # Warmup
    "$BUILD_DIR/cpp_infer" "$MODEL_DIR/yolov5n.onnx" "$IMAGE_DIR/zidane.jpg" > /dev/null 2>&1

    total_time=0
    times=()
    detections=()
    cmd="\"$BUILD_DIR/cpp_infer\" \"$MODEL_DIR/yolov5n.onnx\" \"$IMAGE_DIR/zidane.jpg\""

    for i in $(seq 1 $ITERATIONS); do
        result=$(measure_time_and_accuracy "$cmd")
        IFS=',' read -r time_ms det_count <<< "$result"
        if [ "$time_ms" != "-1" ]; then
            times+=("$time_ms")
            detections+=("$det_count")
            total_time=$((total_time + time_ms))
        fi
    done

    if [ ${#times[@]} -gt 0 ]; then
        avg_time=$((total_time / ${#times[@]}))
        min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
        max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)
        det_count="${detections[0]}"

        echo -e "${GREEN}[PASS]${NC} Avg: ${avg_time}ms, Min: ${min_time}ms, Max: ${max_time}ms | Detections: ${det_count}"
        echo "C++ Baseline (YOLOv5n): avg=${avg_time}ms min=${min_time}ms max=${max_time}ms detections=${det_count}" >> "$RESULTS_FILE"
    fi
    echo ""
fi

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
cat "$RESULTS_FILE"

echo ""
echo "========================================"
echo "  Accuracy Validation"
echo "========================================"

# Extract detection counts for comparison
yolo11n_tensor=$(grep "YOLO11n Detection (Tensor API)" "$RESULTS_FILE" | grep -oP 'detections=\K\d+' || echo "N/A")
yolo11n_legacy=$(grep "YOLO11n Detection (Legacy)" "$RESULTS_FILE" | grep -oP 'detections=\K\d+' || echo "N/A")
yolov5n_tensor=$(grep "YOLOv5n Detection (Tensor API)" "$RESULTS_FILE" | grep -oP 'detections=\K\d+' || echo "N/A")
yolov5n_legacy=$(grep "YOLOv5n Detection (Legacy)" "$RESULTS_FILE" | grep -oP 'detections=\K\d+' || echo "N/A")
cpp_baseline=$(grep "C++ Baseline" "$RESULTS_FILE" | grep -oP 'detections=\K\d+' || echo "N/A")

echo "Detection count comparison (same image, same model):"
echo "  YOLO11n Tensor API:  $yolo11n_tensor"
echo "  YOLO11n Legacy:      $yolo11n_legacy"
echo ""
echo "  YOLOv5n Tensor API:  $yolov5n_tensor"
echo "  YOLOv5n Legacy:      $yolov5n_legacy"
echo "  YOLOv5n C++ Baseline: $cpp_baseline"
echo ""

# Check consistency
all_consistent=true

if [ "$yolo11n_tensor" != "N/A" ] && [ "$yolo11n_legacy" != "N/A" ]; then
    if [ "$yolo11n_tensor" = "$yolo11n_legacy" ]; then
        echo -e "${GREEN}✓${NC} YOLO11n implementations are consistent"
    else
        echo -e "${RED}✗${NC} YOLO11n implementations differ: Tensor=$yolo11n_tensor vs Legacy=$yolo11n_legacy"
        all_consistent=false
    fi
fi

if [ "$yolov5n_tensor" != "N/A" ] && [ "$yolov5n_legacy" != "N/A" ] && [ "$cpp_baseline" != "N/A" ]; then
    if [ "$yolov5n_tensor" = "$yolov5n_legacy" ] && [ "$yolov5n_legacy" = "$cpp_baseline" ]; then
        echo -e "${GREEN}✓${NC} YOLOv5n implementations are consistent"
    else
        echo -e "${RED}✗${NC} YOLOv5n implementations differ: Tensor=$yolov5n_tensor Legacy=$yolov5n_legacy C++=$cpp_baseline"
        all_consistent=false
    fi
fi

echo ""
if [ "$all_consistent" = true ]; then
    echo -e "${GREEN}Overall: All implementations produce consistent detection results${NC}"
else
    echo -e "${YELLOW}Overall: Some inconsistencies detected - review above${NC}"
fi
