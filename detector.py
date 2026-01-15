"""
Object Detector Module for Ice Hockey
Wraps YOLO for detecting players, goaltenders, and puck.
"""
import torch
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple


class ObjectDetector:
    """YOLO detector wrapper for ice hockey."""
    
    # Classes that should be tracked (not Puck, which moves too fast)
    TRACK_CLASSES = ["Player", "Goaltender"]
    
    def __init__(self, config):
        self.config = config
        print(f"Loading detector: {config.model_path} on {config.device}")
        
        self.model = YOLO(config.model_path)
        if config.device != "cpu":
            self.model.to(config.device)
        
        self.device = config.device
        self.CLASS_NAMES = config.class_names
        
        print(f"Loaded detector with classes: {list(self.CLASS_NAMES.values())}")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects in frame."""
        with torch.no_grad():
            results = self.model(
                frame,
                verbose=False,
                conf=self.config.low_confidence_threshold
            )
        
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_labels = boxes.cls.cpu().numpy().astype(int)
        
        return coords, confidences, class_labels
    
    def filter_detections(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Dict[int, Dict]]:
        """Filter and organize detections for ice hockey."""
        height, width = frame_shape
        result = {
            "Player": {},
            "Goaltender": {},
            "Puck": {},
            "Referee": {}
        }
        
        puck_idx = 0
        for i in range(len(boxes)):
            class_idx = int(class_labels[i])
            class_name = self.CLASS_NAMES.get(class_idx)
            
            if class_name not in result:
                continue
            
            conf = float(confidences[i])
            if conf < self.config.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            
            # For puck, use center point instead of bottom center
            if class_name == "Puck":
                bottom_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            else:
                bottom_center = [int((x1 + x2) / 2), y2]
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "bottom_center": bottom_center
            }
            
            if class_name == "Puck":
                result[class_name][puck_idx] = detection
                puck_idx += 1
            else:
                result[class_name][i] = detection
        
        return result
    
    def get_detection_array(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray
    ) -> np.ndarray:
        """Format for tracker."""
        if len(boxes) == 0:
            return np.array([]).reshape(0, 6)
        return np.hstack((
            boxes,
            confidences.reshape(-1, 1),
            class_labels.reshape(-1, 1)
        ))
