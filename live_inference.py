"""
Phase 3: Real-time Inference Script with Metrics Overlay
Runs ONNX model inference on video with live FPS and latency metrics
Model Class: BuildingCracks (single class)
"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from config import (
    MODEL_PATH,
    VIDEO_PATH,
    OUTPUT_PATH,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    IMG_SIZE
)


class InferenceEngine:
    """Handles model loading and inference with latency tracking"""
    
    def __init__(self, model_path):
        """Initialize inference engine with ONNX model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.class_name = "BuildingCracks"
        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        
    def predict(self, frame):
        """
        Run inference on frame with latency tracking
        Returns: (detections, preprocess_ms, inference_ms, postprocess_ms)
        """
        # Preprocess (included in frame → model)
        start_preprocess = time.perf_counter()
        
        # Run inference
        start_inference = time.perf_counter()
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        end_inference = time.perf_counter()
        
        # Postprocess (NMS is built into model output, but we measure total)
        start_postprocess = time.perf_counter()
        detections = results[0].boxes.data.cpu().numpy()
        end_postprocess = time.perf_counter()
        
        # Calculate timings (in milliseconds)
        preprocess_ms = (start_inference - start_preprocess) * 1000
        inference_ms = (end_inference - start_inference) * 1000
        postprocess_ms = (end_postprocess - start_postprocess) * 1000
        
        return detections, preprocess_ms, inference_ms, postprocess_ms


class VideoProcessor:
    """Handles video reading, frame processing, and output"""
    
    def __init__(self, video_path, output_path, engine):
        """Initialize video processor"""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        self.engine = engine
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        # FPS tracking
        self.frame_times = []
        self.max_frame_history = 30
        
    def draw_detections(self, frame, detections, preprocess_ms, inference_ms, postprocess_ms):
        """Draw bounding boxes and metrics on frame"""
        frame_display = frame.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(conf)
            
            # Draw bounding box (green)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence (as percentage out of 100)
            # Handle different confidence scales
            if conf > 100:
                conf_pct = conf / 100.0  # Large scale (0-10000) → 0-100
            else:
                conf_pct = conf * 100.0  # Small scale (0-1) → 0-100
            label = f"{self.engine.class_name} {conf_pct:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(
                frame_display,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame_display,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )
        
        # Calculate and display live FPS (excluding rendering)
        current_time = time.perf_counter()
        self.frame_times.append(current_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            live_fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
        else:
            live_fps = 0
        
        # Draw metrics panel (top-left corner)
        metrics_text = [
            f"Live FPS: {live_fps:.1f}",
            f"Preprocess: {preprocess_ms:.2f}ms",
            f"Inference: {inference_ms:.2f}ms",
            f"Postprocess: {postprocess_ms:.2f}ms",
            f"Detections: {len(detections)}"
        ]
        
        y_offset = 35
        for i, text in enumerate(metrics_text):
            cv2.putText(
                frame_display,
                text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        
        return frame_display, live_fps
    
    def process_video(self):
        """Process entire video and save output"""
        print(f"\nStarting video inference...")
        print(f"Video: {self.cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames @ {self.fps} FPS")
        print(f"Resolution: {self.width}x{self.height}")
        
        frame_count = 0
        all_fps = []
        total_preprocess = 0
        total_inference = 0
        total_postprocess = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Run inference
                detections, preprocess_ms, inference_ms, postprocess_ms = self.engine.predict(frame)
                
                # Draw results
                frame_display, live_fps = self.draw_detections(
                    frame, detections, preprocess_ms, inference_ms, postprocess_ms
                )
                
                # Write frame
                self.out.write(frame_display)
                
                # Accumulate stats
                all_fps.append(live_fps)
                total_preprocess += preprocess_ms
                total_inference += inference_ms
                total_postprocess += postprocess_ms
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{self.total_frames} frames...")
        
        finally:
            self.cap.release()
            self.out.release()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Average Live FPS: {np.mean(all_fps) if all_fps else 0:.1f}")
        print(f"Average Preprocess: {total_preprocess/frame_count if frame_count > 0 else 0:.2f}ms")
        print(f"Average Inference: {total_inference/frame_count if frame_count > 0 else 0:.2f}ms")
        print(f"Average Postprocess: {total_postprocess/frame_count if frame_count > 0 else 0:.2f}ms")
        print(f"Output saved to: {self.output_path}")
        print("="*50 + "\n")


def main():
    """Main execution function"""
    try:
        # Initialize inference engine
        print(f"Loading ONNX model: {MODEL_PATH}")
        engine = InferenceEngine(MODEL_PATH)
        print("✓ Model loaded successfully")
        
        # Initialize video processor
        processor = VideoProcessor(VIDEO_PATH, OUTPUT_PATH, engine)
        print(f"✓ Video loaded: {processor.width}x{processor.height} @ {processor.fps} FPS")
        
        # Process video
        processor.process_video()
        print("✓ Inference completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Model file exists at: {MODEL_PATH}")
        print(f"  2. Video file exists at: {VIDEO_PATH}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
