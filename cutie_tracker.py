"""
CUTIE-Based Standalone Tracker
Uses SAM2 for mask generation and CUTIE for mask-based tracking (ID assignment).
Can replace BoTSORT entirely.
"""
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CUTIETrackerConfig:
    """Configuration for CUTIE-based tracker."""
    # SAM2 settings
    sam2_model: str = "facebook/sam2-hiera-large"
    
    # CUTIE settings
    cutie_model: str = "cutie-base"
    cutie_mem_every: int = 5
    cutie_top_k: int = 30
    
    # Tracking settings
    mask_threshold: float = 0.5
    min_mask_area: int = 100
    new_object_threshold: float = 0.3  # IoU threshold for new object detection
    
    # ID management
    max_lost_frames: int = 30
    
    # Device
    device: str = "cuda"
    use_amp: bool = True


class CUTIEStandaloneTracker:
    """
    Standalone tracker using CUTIE for ID assignment and tracking.
    No BoTSORT dependency - CUTIE handles everything.
    
    Pipeline:
    1. YOLO detects objects → bounding boxes
    2. SAM2 generates masks from boxes
    3. CUTIE propagates masks AND maintains object IDs
    """
    
    def __init__(self, config: CUTIETrackerConfig):
        self.config = config
        self.device = config.device
        
        # Models
        self.sam2 = None
        self.cutie = None
        self._load_models()
        
        # Tracking state
        self.next_id = 1
        self.active_tracks: Dict[int, TrackInfo] = {}
        self.lost_tracks: Dict[int, TrackInfo] = {}
        
        # Frame state
        self.frame_count = 0
        self.current_masks: Dict[int, np.ndarray] = {}
        self.initialized = False
        
        # CUTIE memory state
        self.cutie_objects: List[int] = []
    
    def _load_models(self):
        """Load SAM2 and CUTIE models."""
        print("Loading SAM2 + CUTIE (standalone mode)...")
        
        # Load SAM2
        self._load_sam2()
        
        # Load CUTIE
        self._load_cutie()
        
        print("✓ CUTIE Standalone Tracker ready")
    
    def _load_sam2(self):
        """Load SAM2 model."""
        try:
            # Try transformers version first
            from transformers import Sam2Model, Sam2Processor
            
            self.sam2_processor = Sam2Processor.from_pretrained(self.config.sam2_model)
            self.sam2_model = Sam2Model.from_pretrained(self.config.sam2_model)
            self.sam2_model.to(self.device)
            self.sam2_model.eval()
            
            if self.config.use_amp and self.device == "cuda":
                self.sam2_model = self.sam2_model.half()
            
            self.sam2_type = "transformers"
            print("  ✓ SAM2 loaded (transformers)")
            
        except ImportError:
            try:
                # Try native sam2 package
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                self.sam2_model = build_sam2(
                    config_file="sam2_hiera_l.yaml",
                    ckpt_path="sam2_hiera_large.pt",
                    device=self.device
                )
                self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
                self.sam2_type = "native"
                print("  ✓ SAM2 loaded (native)")
                
            except Exception as e:
                print(f"  ⚠️ SAM2 not available: {e}")
                print("  → Using box-based masks as fallback")
                self.sam2_model = None
                self.sam2_type = "fallback"
    
    def _load_cutie(self):
        """Load CUTIE model."""
        try:
            from cutie.inference.inference_core import InferenceCore
            from cutie.utils.get_default_model import get_default_model
            
            cutie_model = get_default_model()
            self.cutie = InferenceCore(
                cutie_model,
                cfg={
                    "top_k": self.config.cutie_top_k,
                    "mem_every": self.config.cutie_mem_every
                }
            )
            print("  ✓ CUTIE loaded")
            
        except ImportError as e:
            print(f"  ⚠️ CUTIE not available: {e}")
            print("  → Using simple mask IoU tracking as fallback")
            self.cutie = None
    
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
        
        if len(detections) == 0:
            self._handle_no_detections(frame)
            return np.array([])
        
        # Extract detection info
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        class_labels = detections[:, 5].astype(int)
        
        # Generate masks for detections
        detection_masks = self._generate_masks(frame, boxes)
        
        # Match detections to existing tracks or create new ones
        if not self.initialized:
            # First frame - initialize all as new tracks
            track_ids = self._initialize_tracks(frame, detection_masks, boxes, confidences, class_labels)
            self.initialized = True
        else:
            # Subsequent frames - use CUTIE propagation + matching
            track_ids = self._match_and_update(frame, detection_masks, boxes, confidences, class_labels)
        
        # Build output array
        results = []
        for i, track_id in enumerate(track_ids):
            results.append([
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                track_id,
                confidences[i],
                class_labels[i],
                i  # detection index
            ])
        
        return np.array(results) if results else np.array([])
    
    def _generate_masks(
        self,
        frame: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """Generate masks for all bounding boxes."""
        if self.sam2_model is None or self.sam2_type == "fallback":
            return self._generate_box_masks(frame, boxes)
        
        masks = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.sam2_type == "transformers":
            masks = self._generate_masks_transformers(frame_rgb, boxes)
        else:
            masks = self._generate_masks_native(frame_rgb, boxes)
        
        return masks
    
    def _generate_masks_transformers(
        self,
        frame_rgb: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """Generate masks using transformers SAM2."""
        masks = []
        
        for box in boxes:
            try:
                inputs = self.sam2_processor(
                    images=frame_rgb,
                    input_boxes=[[box.tolist()]],
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    if self.config.use_amp and self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            outputs = self.sam2_model(**inputs)
                    else:
                        outputs = self.sam2_model(**inputs)
                
                processed_masks = self.sam2_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )[0]
                
                scores = outputs.iou_scores[0]
                best_idx = scores.argmax().item()
                mask = processed_masks[0, best_idx].cpu().numpy()
                masks.append(mask > self.config.mask_threshold)
                
            except Exception as e:
                # Fallback to box mask
                masks.append(self._create_box_mask(frame_rgb.shape[:2], box))
        
        return masks
    
    def _generate_masks_native(
        self,
        frame_rgb: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """Generate masks using native SAM2."""
        masks = []
        self.sam2_predictor.set_image(frame_rgb)
        
        for box in boxes:
            try:
                mask_outputs, scores, _ = self.sam2_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                best_idx = scores.argmax()
                masks.append(mask_outputs[best_idx])
            except Exception:
                masks.append(self._create_box_mask(frame_rgb.shape[:2], box))
        
        return masks
    
    def _generate_box_masks(
        self,
        frame: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """Generate simple box-based masks as fallback."""
        h, w = frame.shape[:2]
        masks = []
        
        for box in boxes:
            masks.append(self._create_box_mask((h, w), box))
        
        return masks
    
    def _create_box_mask(
        self,
        shape: Tuple[int, int],
        box: np.ndarray
    ) -> np.ndarray:
        """Create rectangular mask from bounding box."""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask
    
    def _initialize_tracks(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray
    ) -> List[int]:
        """Initialize tracks for first frame."""
        track_ids = []
        
        # Create combined mask for CUTIE
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.int32)
        
        for i, mask in enumerate(masks):
            track_id = self.next_id
            self.next_id += 1
            track_ids.append(track_id)
            
            # Store track info
            self.active_tracks[track_id] = TrackInfo(
                track_id=track_id,
                last_mask=mask,
                last_box=boxes[i],
                last_seen=self.frame_count,
                class_label=int(class_labels[i]),
                confidence=float(confidences[i])
            )
            
            # Add to combined mask
            combined_mask[mask] = track_id
            self.current_masks[track_id] = mask
        
        # Initialize CUTIE with first frame
        if self.cutie is not None:
            self._init_cutie(frame, combined_mask, track_ids)
        
        return track_ids
    
    def _init_cutie(
        self,
        frame: np.ndarray,
        combined_mask: np.ndarray,
        track_ids: List[int]
    ):
        """Initialize CUTIE with first frame masks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)
        
        self.cutie_objects = track_ids.copy()
        self.cutie.set_all_labels(track_ids)
        self.cutie.step(frame_tensor, mask_tensor, objects=track_ids)
    
    def _match_and_update(
        self,
        frame: np.ndarray,
        detection_masks: List[np.ndarray],
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_labels: np.ndarray
    ) -> List[int]:
        """Match detections to tracks using CUTIE propagation."""
        
        # Get propagated masks from CUTIE
        propagated_masks = self._propagate_cutie(frame)
        
        # Match detections to propagated masks
        track_ids = []
        matched_tracks = set()
        unmatched_detections = []
        
        for det_idx, det_mask in enumerate(detection_masks):
            best_track_id = None
            best_iou = 0
            
            # Try to match with propagated masks
            for track_id, prop_mask in propagated_masks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._mask_iou(det_mask, prop_mask)
                if iou > best_iou and iou > self.config.new_object_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Matched to existing track
                track_ids.append(best_track_id)
                matched_tracks.add(best_track_id)
                
                # Update track info
                self.active_tracks[best_track_id].last_mask = det_mask
                self.active_tracks[best_track_id].last_box = boxes[det_idx]
                self.active_tracks[best_track_id].last_seen = self.frame_count
                self.active_tracks[best_track_id].confidence = float(confidences[det_idx])
                self.current_masks[best_track_id] = det_mask
            else:
                # Unmatched - might be new object
                unmatched_detections.append(det_idx)
                track_ids.append(-1)  # Placeholder
        
        # Handle unmatched detections - create new tracks
        new_track_ids = []
        new_masks = []
        
        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            # Update placeholder
            track_ids[det_idx] = track_id
            new_track_ids.append(track_id)
            new_masks.append(detection_masks[det_idx])
            
            # Create track info
            self.active_tracks[track_id] = TrackInfo(
                track_id=track_id,
                last_mask=detection_masks[det_idx],
                last_box=boxes[det_idx],
                last_seen=self.frame_count,
                class_label=int(class_labels[det_idx]),
                confidence=float(confidences[det_idx])
            )
            self.current_masks[track_id] = detection_masks[det_idx]
        
        # Add new tracks to CUTIE
        if new_track_ids and self.cutie is not None:
            self._add_to_cutie(frame, new_masks, new_track_ids)
        
        # Handle lost tracks
        self._update_lost_tracks(matched_tracks)
        
        return track_ids
    
    def _propagate_cutie(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """Propagate masks using CUTIE."""
        if self.cutie is None:
            return self._simple_propagate()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        propagated = {}
        
        try:
            with torch.no_grad():
                if self.config.use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        output = self.cutie.step(frame_tensor)
                else:
                    output = self.cutie.step(frame_tensor)
            
            if output is not None:
                # Output shape: [1, num_objects, H, W]
                prob_masks = output.squeeze(0).cpu().numpy()
                
                for i, track_id in enumerate(self.cutie_objects):
                    if i < prob_masks.shape[0]:
                        mask = prob_masks[i] > self.config.mask_threshold
                        if np.sum(mask) >= self.config.min_mask_area:
                            propagated[track_id] = mask
        
        except Exception as e:
            print(f"CUTIE propagation error: {e}")
            return self._simple_propagate()
        
        return propagated
    
    def _simple_propagate(self) -> Dict[int, np.ndarray]:
        """Simple propagation using last known masks."""
        return {tid: info.last_mask for tid, info in self.active_tracks.items()}
    
    def _add_to_cutie(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        track_ids: List[int]
    ):
        """Add new objects to CUTIE memory."""
        if not masks or self.cutie is None:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Create combined mask for new objects
        combined_mask = np.zeros((h, w), dtype=np.int32)
        for mask, track_id in zip(masks, track_ids):
            combined_mask[mask] = track_id
        
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)
        
        # Update CUTIE objects list
        self.cutie_objects.extend(track_ids)
        self.cutie.set_all_labels(self.cutie_objects)
        self.cutie.step(frame_tensor, mask_tensor, objects=track_ids)
    
    def _update_lost_tracks(self, matched_tracks: set):
        """Move unmatched tracks to lost."""
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_tracks:
                track = self.active_tracks[track_id]
                frames_lost = self.frame_count - track.last_seen
                
                if frames_lost > self.config.max_lost_frames:
                    # Remove from active
                    del self.active_tracks[track_id]
                    if track_id in self.current_masks:
                        del self.current_masks[track_id]
                    # Remove from CUTIE objects
                    if track_id in self.cutie_objects:
                        self.cutie_objects.remove(track_id)
    
    def _handle_no_detections(self, frame: np.ndarray):
        """Handle frame with no detections."""
        # Still propagate masks to maintain state
        if self.cutie is not None and self.initialized:
            self._propagate_cutie(frame)
    
    def _mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def get_masks(self) -> Dict[int, np.ndarray]:
        """Get current masks for all active tracks."""
        return self.current_masks.copy()
    
    def get_track_info(self, track_id: int) -> Optional['TrackInfo']:
        """Get track information."""
        return self.active_tracks.get(track_id)
    
    def reset(self):
        """Reset tracker state."""
        self.next_id = 1
        self.active_tracks = {}
        self.lost_tracks = {}
        self.frame_count = 0
        self.current_masks = {}
        self.initialized = False
        self.cutie_objects = []
        
        if self.cutie is not None:
            self.cutie.clear_memory()


class TrackInfo:
    """Information about a tracked object."""
    
    def __init__(
        self,
        track_id: int,
        last_mask: np.ndarray,
        last_box: np.ndarray,
        last_seen: int,
        class_label: int,
        confidence: float
    ):
        self.track_id = track_id
        self.last_mask = last_mask
        self.last_box = last_box
        self.last_seen = last_seen
        self.class_label = class_label
        self.confidence = confidence
        self.team_id = -1


def create_cutie_tracker(config: CUTIETrackerConfig = None) -> CUTIEStandaloneTracker:
    """Factory function to create CUTIE standalone tracker."""
    if config is None:
        config = CUTIETrackerConfig()
    return CUTIEStandaloneTracker(config)
