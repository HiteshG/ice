"""
Unified Tracker with Mode Switching
Supports 3 tracking modes:
1. botsort - BoTSORT only (fast, uses ReID)
2. sam2_cutie - SAM2 + CUTIE only (mask-based, better occlusion handling)
3. hybrid - Both combined (best accuracy, slowest)
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class TrackingMode(Enum):
    BOTSORT = "botsort"
    SAM2_CUTIE = "sam2_cutie"
    HYBRID = "hybrid"


@dataclass
class UnifiedTrackerConfig:
    """Configuration for unified tracker."""
    # Mode selection
    mode: str = "botsort"  # "botsort", "sam2_cutie", "hybrid"
    
    # Device
    device: str = "cuda"
    
    # BoTSORT settings
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    
    # SAM2 settings
    sam2_model: str = "facebook/sam2-hiera-large"
    
    # CUTIE settings
    cutie_model: str = "cutie-base"
    cutie_mem_every: int = 5
    cutie_top_k: int = 30
    
    # Tracking thresholds
    mask_threshold: float = 0.5
    min_mask_area: int = 100
    iou_threshold: float = 0.3
    max_lost_frames: int = 30
    
    # Hybrid mode settings
    mask_weight: float = 0.3  # Weight for mask-based matching in hybrid mode
    
    # Performance
    use_amp: bool = True


class UnifiedTracker:
    """
    Unified tracker supporting multiple tracking modes.
    
    Modes:
    - botsort: Fast, uses appearance ReID, good for general cases
    - sam2_cutie: Mask-based tracking, better occlusion handling, slower
    - hybrid: Combines both for best accuracy
    """
    
    def __init__(self, config: UnifiedTrackerConfig):
        self.config = config
        self.mode = TrackingMode(config.mode)
        self.device = config.device
        
        print(f"\n{'='*60}")
        print(f"UNIFIED TRACKER - Mode: {self.mode.value.upper()}")
        print(f"{'='*60}\n")
        
        # Initialize trackers based on mode
        self.botsort_tracker = None
        self.cutie_tracker = None
        self.sam2_wrapper = None
        
        self._initialize_trackers()
        
        # State
        self.frame_count = 0
        self.current_masks: Dict[int, np.ndarray] = {}
        self.track_info: Dict[int, dict] = {}
    
    def _initialize_trackers(self):
        """Initialize trackers based on mode."""
        
        if self.mode in [TrackingMode.BOTSORT, TrackingMode.HYBRID]:
            self._init_botsort()
        
        if self.mode in [TrackingMode.SAM2_CUTIE, TrackingMode.HYBRID]:
            self._init_sam2_cutie()
    
    def _init_botsort(self):
        """Initialize BoTSORT tracker."""
        try:
            from boxmot import BotSort
            from pathlib import Path
            
            device = 0 if self.config.device == "cuda" else self.config.device
            if self.config.device == "mps":
                device = "cpu"
            
            self.botsort_tracker = BotSort(
                reid_weights=Path(self.config.reid_weights),
                device=device,
                half=False
            )
            print("✓ BoTSORT initialized")
            
        except Exception as e:
            print(f"⚠️ BoTSORT initialization failed: {e}")
            if self.mode == TrackingMode.BOTSORT:
                print("  → Using simple IoU tracker as fallback")
                self.botsort_tracker = SimpleIoUTracker()
    
    def _init_sam2_cutie(self):
        """Initialize SAM2 + CUTIE tracker."""
        try:
            from cutie_tracker import CUTIEStandaloneTracker, CUTIETrackerConfig
            
            cutie_config = CUTIETrackerConfig(
                sam2_model=self.config.sam2_model,
                cutie_model=self.config.cutie_model,
                cutie_mem_every=self.config.cutie_mem_every,
                cutie_top_k=self.config.cutie_top_k,
                mask_threshold=self.config.mask_threshold,
                min_mask_area=self.config.min_mask_area,
                max_lost_frames=self.config.max_lost_frames,
                device=self.config.device,
                use_amp=self.config.use_amp
            )
            
            self.cutie_tracker = CUTIEStandaloneTracker(cutie_config)
            print("✓ SAM2 + CUTIE initialized")
            
        except Exception as e:
            print(f"⚠️ SAM2 + CUTIE initialization failed: {e}")
            if self.mode == TrackingMode.SAM2_CUTIE:
                raise RuntimeError("SAM2 + CUTIE mode selected but initialization failed")
    
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: (N, 6) array [x1, y1, x2, y2, conf, class]
            frame: Current frame in BGR format
            
        Returns:
            (N, 8) array [x1, y1, x2, y2, track_id, conf, class, det_idx]
        """
        self.frame_count += 1
        
        if self.mode == TrackingMode.BOTSORT:
            return self._update_botsort(detections, frame)
        
        elif self.mode == TrackingMode.SAM2_CUTIE:
            return self._update_sam2_cutie(detections, frame)
        
        else:  # HYBRID
            return self._update_hybrid(detections, frame)
    
    def _update_botsort(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """Update using BoTSORT only."""
        if len(detections) == 0:
            return np.array([])
        
        try:
            tracks = self.botsort_tracker.update(detections, frame)
            return tracks if len(tracks) > 0 else np.array([])
        except Exception as e:
            print(f"BoTSORT update error: {e}")
            return np.array([])
    
    def _update_sam2_cutie(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """Update using SAM2 + CUTIE only."""
        if self.cutie_tracker is None:
            return self._update_botsort(detections, frame)
        
        tracks = self.cutie_tracker.update(detections, frame)
        
        # Store masks
        self.current_masks = self.cutie_tracker.get_masks()
        
        return tracks
    
    def _update_hybrid(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """Update using both BoTSORT and SAM2 + CUTIE."""
        if len(detections) == 0:
            return np.array([])
        
        # Get tracks from both
        botsort_tracks = self._update_botsort(detections.copy(), frame)
        
        if self.cutie_tracker is not None:
            cutie_tracks = self.cutie_tracker.update(detections.copy(), frame)
            self.current_masks = self.cutie_tracker.get_masks()
        else:
            cutie_tracks = np.array([])
        
        # Merge results - prefer BoTSORT IDs but use CUTIE for occlusion recovery
        if len(botsort_tracks) == 0:
            return cutie_tracks
        
        if len(cutie_tracks) == 0:
            return botsort_tracks
        
        # Hybrid merging logic
        return self._merge_tracks(botsort_tracks, cutie_tracks, frame)
    
    def _merge_tracks(
        self,
        botsort_tracks: np.ndarray,
        cutie_tracks: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Merge tracks from BoTSORT and CUTIE.
        
        Strategy:
        - Use BoTSORT IDs as primary
        - Use CUTIE masks for occlusion detection
        - If BoTSORT loses track but CUTIE maintains, prefer CUTIE
        """
        merged = []
        
        # Create mapping from detection index to tracks
        botsort_by_det = {int(t[7]): t for t in botsort_tracks}
        cutie_by_det = {int(t[7]): t for t in cutie_tracks}
        
        all_det_indices = set(botsort_by_det.keys()) | set(cutie_by_det.keys())
        
        for det_idx in all_det_indices:
            bot_track = botsort_by_det.get(det_idx)
            cutie_track = cutie_by_det.get(det_idx)
            
            if bot_track is not None and cutie_track is not None:
                # Both have this detection - use BoTSORT ID
                # But check if CUTIE suggests different (occlusion recovery)
                bot_id = int(bot_track[4])
                cutie_id = int(cutie_track[4])
                
                # Check if this might be a recovered track
                if self._is_recovered_track(cutie_id):
                    # CUTIE recovered a lost track - use CUTIE ID
                    result = cutie_track.copy()
                else:
                    # Normal case - use BoTSORT
                    result = bot_track.copy()
                
                merged.append(result)
                
            elif bot_track is not None:
                merged.append(bot_track)
                
            elif cutie_track is not None:
                merged.append(cutie_track)
        
        return np.array(merged) if merged else np.array([])
    
    def _is_recovered_track(self, track_id: int) -> bool:
        """Check if track was recently recovered by CUTIE."""
        if self.cutie_tracker is None:
            return False
        
        track_info = self.cutie_tracker.get_track_info(track_id)
        if track_info is None:
            return False
        
        # Consider it recovered if there was a gap in tracking
        # This is a simplified check - could be more sophisticated
        return False  # Placeholder - implement based on track history
    
    def organize_tracks(
        self,
        tracks: np.ndarray,
        class_names: Dict[int, str],
        confidence_threshold: float,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Dict[int, Dict]]:
        """
        Organize tracked objects by class.
        
        Returns:
            Dictionary organized by class name and track ID
        """
        result = {
            "Player": {},
            "Goalkeeper": {},
            "Ball": {}
        }
        
        if len(tracks) == 0:
            return result
        
        height, width = frame_shape
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, class_idx, _ = track
            
            track_id = int(track_id)
            class_idx = int(class_idx)
            conf = float(conf)
            
            class_name = class_names.get(class_idx, "Unknown")
            
            if class_name not in result:
                continue
            
            if conf < confidence_threshold:
                continue
            
            # Clip coordinates
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            
            # Bottom center
            bottom_center = [int((x1 + x2) / 2), y2]
            
            # Get mask if available
            mask = self.current_masks.get(track_id)
            
            result[class_name][track_id] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "bottom_center": bottom_center,
                "mask": mask,
                "is_occluded": self._check_occlusion(track_id)
            }
        
        return result
    
    def _check_occlusion(self, track_id: int) -> bool:
        """Check if track is currently occluded."""
        if self.mode == TrackingMode.BOTSORT:
            return False
        
        # Check mask overlap with other tracks
        if track_id not in self.current_masks:
            return False
        
        mask = self.current_masks[track_id]
        
        for other_id, other_mask in self.current_masks.items():
            if other_id == track_id:
                continue
            
            intersection = np.logical_and(mask, other_mask).sum()
            area = mask.sum()
            
            if area > 0 and intersection / area > self.config.iou_threshold:
                return True
        
        return False
    
    def get_masks(self) -> Dict[int, np.ndarray]:
        """Get current masks for all tracks."""
        return self.current_masks.copy()
    
    def get_mode(self) -> str:
        """Get current tracking mode."""
        return self.mode.value
    
    def set_mode(self, mode: str):
        """
        Change tracking mode at runtime.
        Note: This will reset tracking state.
        """
        new_mode = TrackingMode(mode)
        if new_mode != self.mode:
            print(f"\nSwitching mode: {self.mode.value} → {new_mode.value}")
            self.mode = new_mode
            self.reset()
            self._initialize_trackers()
    
    def reset(self):
        """Reset tracker state."""
        self.frame_count = 0
        self.current_masks = {}
        self.track_info = {}
        
        if self.botsort_tracker is not None:
            try:
                from boxmot import BotSort
                from pathlib import Path
                
                device = 0 if self.config.device == "cuda" else self.config.device
                self.botsort_tracker = BotSort(
                    reid_weights=Path(self.config.reid_weights),
                    device=device,
                    half=False
                )
            except:
                pass
        
        if self.cutie_tracker is not None:
            self.cutie_tracker.reset()
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        stats = {
            "mode": self.mode.value,
            "frame_count": self.frame_count,
            "active_tracks": len(self.current_masks),
        }
        
        if self.cutie_tracker is not None:
            stats["cutie_objects"] = len(self.cutie_tracker.cutie_objects)
        
        return stats


class SimpleIoUTracker:
    """Simple IoU-based tracker as fallback."""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.next_id = 1
        self.tracks: Dict[int, np.ndarray] = {}
        self.iou_threshold = iou_threshold
    
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return np.array([])
        
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        classes = detections[:, 5]
        
        results = []
        matched = set()
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            best_iou = 0
            best_id = None
            
            for track_id, track_box in self.tracks.items():
                if track_id in matched:
                    continue
                iou = self._iou(box, track_box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is None:
                track_id = self.next_id
                self.next_id += 1
            else:
                track_id = best_id
                matched.add(track_id)
            
            self.tracks[track_id] = box
            results.append([box[0], box[1], box[2], box[3], track_id, conf, cls, i])
        
        return np.array(results)
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


def create_tracker(
    mode: str = "botsort",
    device: str = "cuda",
    **kwargs
) -> UnifiedTracker:
    """
    Factory function to create unified tracker.
    
    Args:
        mode: "botsort", "sam2_cutie", or "hybrid"
        device: "cuda", "cpu", or "mps"
        **kwargs: Additional config options
        
    Returns:
        UnifiedTracker instance
    """
    config = UnifiedTrackerConfig(mode=mode, device=device, **kwargs)
    return UnifiedTracker(config)
