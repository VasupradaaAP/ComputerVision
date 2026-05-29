

from pathlib import Path

# ============================================================================
# PATHS - MODIFY THESE ACCORDING TO YOUR FILE LOCATIONS
# ============================================================================

# Absolute path to your trained ONNX model
# Example: r"D:\Industrial defect\runs\detect\train\weights\best.onnx"
MODEL_PATH = r"D:\Industrial defect\phase3\trained_file.onnx"

# Absolute path to your test video
# Example: r"D:\Industrial defect\test_data\sample_video.mp4"
VIDEO_PATH = r"D:\Industrial defect\phase3\test_data\sample.mp4"

# Output video path (where results will be saved)
OUTPUT_PATH = r"D:\Industrial defect\phase3\output\inference_output.mp4"

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================

# Confidence threshold (detections below this will be filtered)
CONFIDENCE_THRESHOLD = 0.5

# Non-Maximum Suppression (NMS) threshold
NMS_THRESHOLD = 0.45

# Input image size for the model
IMG_SIZE = 640

# ============================================================================
# CLASS MAPPING (BuildingCracks = single class model)
# ============================================================================

CLASS_NAMES = {
    0: "BuildingCracks"
}



