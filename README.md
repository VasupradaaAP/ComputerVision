# ComputerVision - Real World Industrial Problem : Detecting Structural Cracks in Building Walls and Concrete Surfaces

## Overview
This module implements real-time inference on video using your trained ONNX model with live metrics overlay including FPS, latency, and detections.

## Prerequisites
- Python 3.8+
- Virtual environment activated: `.\.venv\Scripts\activate`
- ONNX model file: `best.onnx` (from Phase 2)
- Test video file: `.mp4` or other OpenCV-supported format

## Quick Start

### 1. Prepare Environment
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
cd phase3
pip install -r requirements.txt
```

### 2. Prepare Files

```
Industrial defect/
├── runs/detect/train/weights/best.onnx    ← Your ONNX model
└── test_data/
    └── sample_video.mp4                   ← Your test video
```

### 3. Update Configuration
Edit `config.py` and set correct paths:
```python
MODEL_PATH = r"D:\Industrial defect\runs\detect\train\weights\best.onnx"
VIDEO_PATH = r"D:\Industrial defect\test_data\sample_video.mp4"
OUTPUT_PATH = r"D:\Industrial defect\phase3\output\inference_output.mp4"
```


### 4. Run Inference
```bash
python live_inference.py
```

## Output
- **Video Output**: Saved to path specified in `config.py`
- **Console Output**: Real-time progress and summary statistics
- **Metrics Overlaid**:
  - Bounding boxes with "BuildingCracks" label and confidence
  - Live FPS (excluding rendering time)
  - Pre-processing latency (ms)
  - Inference latency (ms)
  - Post-processing latency (ms)
  - Detection count per frame

## Metrics Explanation

### Live FPS
- Frames per second calculated over a rolling window (last 30 frames)
- Excludes video rendering/writing time
- Represents pure inference throughput

### Latencies (milliseconds)
- **Preprocess**: Image preparation before model input
- **Inference**: Model forward pass execution
- **Postprocess**: NMS and output post-processing



## Performance Benchmark

### Model Comparison: FP32 (Baseline) vs. FP16 Quantized (Edge)

| Metric | FP32 Baseline | FP16 Quantized | Change | Notes |
|--------|---------------|----------------|--------|-------|
| **Model Format** | `.pt` (PyTorch) | `.onnx` (ONNX FP16) | - | Edge-optimized format |
| **Model Size (MB)** | 5.75 MB | 4.72 MB | -17.9% ↓ | 1.03 MB reduction |
| **Inference Latency (ms)** | ~101.8 ms | ~78.5 ms | -22.9% ↓ | Per-frame forward pass |
| **Throughput (FPS)** | ~9.8 FPS | ~12.7 FPS | +29.6% ↑ | On local CPU machine |

### Accuracy Metrics: mAP Comparison

| Metric | Baseline (FP32) | Quantized (FP16) | Accuracy Drop |
|--------|-----------------|------------------|---------------|
| **mAP50 (Precision @ IoU=0.5)** | ~0.54 | 0.5368 | -0.4% |
| **mAP50-95 (Full Range @ IoU=0.5:0.95)** | ~0.28 | 0.2204 | -21.3% |
| **Precision** | 0.583 | - | Validated on 54 images |
| **Recall** | 0.553 | - | Single class: BuildingCracks |

### Key Findings

1. **Size Optimization**: FP16 quantization reduced model size by **17.9%** (5.75 MB → 4.72 MB)
   - Suitable for edge devices with limited storage
   - Faster model loading time

2. **Speed Improvement**: Quantized model achieved **29.6% faster inference** (9.8 FPS → 12.7 FPS)
   - Preprocess: 1.8 ms (unchanged)
   - Inference: 101.8 ms → 78.5 ms (-22.9% faster)
   - Postprocess: 0.2 ms (unchanged)

3. **Accuracy Trade-off**: 
   - **mAP50 drop: 0.4%** (acceptable - 0.54 → 0.5368)
   - **mAP50-95 drop: 21.3%** (expected with FP16 quantization)
   - Suitable for building crack detection use case where rough detection is acceptable

### Recommendation

**FP16 Quantization is suitable** for this BuildingCracks detection task because:
- ✅ 29.6% speed improvement for edge deployment
- ✅ Minimal size reduction suitable for IoT/mobile devices
- ✅ Only 0.4% accuracy loss on mAP50 (most practical metric)
- ✅ Inference time reduced below 80ms, achievable on CPU


## File Structure
```
phase3/
├── live_inference.py          # Main inference script
├── config.py                  # Configuration (paths & parameters)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── output/                    # Output videos saved here
```

## Troubleshooting

### Model Not Found
```
Error: Model not found: D:\Industrial defect\runs\detect\train\weights\best.onnx
```
**Solution**: Verify ONNX file path in `config.py` matches your actual file location.

### Video Not Found
```
Error: Video not found: D:\Industrial defect\test_data\sample_video.mp4
```
**Solution**: Place your test video in `test_data/` folder and update path in `config.py`.

### OutOfMemory Error
**Solution**: 
- Reduce `IMG_SIZE` in `config.py` (e.g., 480 instead of 640)
- Process smaller video files first to test

### No Detections
**Solution**: 
- Lower `CONFIDENCE_THRESHOLD` in `config.py` (e.g., 0.3 instead of 0.5)
- Verify test video contains building cracks

### Slow Performance
**Check**:
1. CPU usage during inference
2. Video resolution (lower = faster)
3. Try different `IMG_SIZE` values (480, 416, etc.)

## Safe & Secure Design
✓ No external API calls or network dependencies
✓ All processing local to your machine
✓ No data transmission
✓ Standard open-source libraries only
✓ Can run offline without internet

## Performance Notes
- FPS depends on: video resolution, model size, system hardware
- Inference time dominates on CPU; faster on GPU if available
- Pre-processing and post-processing are typically <5ms combined

## Model Information
- **Model Type**: ONNX format (FP16 quantized)
- **Single Class**: BuildingCracks
- **Input Size**: Configurable (default 640x640)
- **Output**: Bounding boxes with confidence scores

## Support Files Needed From You
1. **best.onnx** - Your trained model (required)
2. **sample_video.mp4** - Your test video (required)

## Example Output Statistics
```
==================================================
INFERENCE SUMMARY
==================================================
Total frames processed: 150
Average Live FPS: 12.5
Average Preprocess: 2.34ms
Average Inference: 78.45ms
Average Postprocess: 1.21ms
Output saved to: D:\Industrial defect\phase3\output\inference_output.mp4
==================================================
```

---

