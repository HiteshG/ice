"""
Visualization Module with Mask Support for Ice Hockey.
"""
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class Visualizer:
    """Draws tracking results with optional mask overlays for ice hockey."""
    
    def __init__(self, config):
        self.config = config
    
    def draw_frame(
        self,
        frame: np.ndarray,
        detections: Dict[str, Dict[int, Dict]],
        team_mapping: Dict[int, int],
        masks: Optional[Dict[int, np.ndarray]] = None
    ) -> np.ndarray:
        """Draw detections on frame."""
        annotated = frame.copy()
        
        # Draw masks first (underneath)
        if masks and self.config.show_masks:
            annotated = self._draw_masks(annotated, masks, team_mapping)
        
        # Draw players/goaltenders
        for class_name in ["Player", "Goaltender"]:
            if class_name not in detections:
                continue
            
            for obj_id, det in detections[class_name].items():
                bbox = det.get("bbox", det.get("box", [0,0,0,0]))
                bottom_center = det["bottom_center"]
                is_occluded = det.get("is_occluded", False)
                
                # Color
                if class_name == "Goaltender":
                    color = self.config.goaltender_color
                elif is_occluded:
                    color = self.config.occlusion_color
                else:
                    team_id = team_mapping.get(obj_id, 0)
                    color = self.config.team_colors.get(team_id, (255,255,255))
                
                # Bounding box
                if self.config.show_bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    thickness = 1 if is_occluded else 2
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
                    
                    if is_occluded:
                        cv2.putText(annotated, "OCC", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                   self.config.occlusion_color, 1)
                
                # Ellipse at bottom center
                x, y = bottom_center
                cv2.ellipse(annotated, (x, y), (35, 18), 0, -45, 235, color, 2)
                
                # ID
                if self.config.show_ids:
                    cv2.putText(annotated, str(obj_id), (x-10, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw puck
        if self.config.show_puck and "Puck" in detections:
            for puck_det in detections["Puck"].values():
                x, y = puck_det["bottom_center"]
                # Draw puck as a circle
                cv2.circle(annotated, (x, y), 8, self.config.puck_color, -1)
                cv2.circle(annotated, (x, y), 10, self.config.puck_color, 2)
        
        return annotated
    
    def _draw_masks(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        team_mapping: Dict[int, int]
    ) -> np.ndarray:
        """Draw mask overlays."""
        overlay = frame.copy()
        alpha = self.config.mask_alpha
        
        for obj_id, mask in masks.items():
            if mask is None:
                continue
            
            team_id = team_mapping.get(obj_id, 0)
            color = self.config.team_colors.get(team_id, (255,255,255))
            
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = color
        
        return cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    def draw_from_dataframe(
        self,
        frame: np.ndarray,
        frame_idx: int,
        df: pd.DataFrame,
        team_mapping: Dict[int, int],
        masks: Optional[Dict[int, np.ndarray]] = None
    ) -> np.ndarray:
        """Draw from DataFrame."""
        if frame_idx not in df.index:
            return frame
        
        annotated = frame.copy()
        
        # Draw masks
        if masks and self.config.show_masks:
            annotated = self._draw_masks(annotated, masks, team_mapping)
        
        row = df.loc[frame_idx]
        
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            
            x, y = int(val[0]), int(val[1])
            
            if col == "Puck":
                if self.config.show_puck:
                    # Draw puck as a circle
                    cv2.circle(annotated, (x, y), 8, self.config.puck_color, -1)
                    cv2.circle(annotated, (x, y), 10, self.config.puck_color, 2)
            else:
                parts = col.split("_")
                obj_type = parts[0]
                obj_id = int(parts[1])
                
                if obj_type == "Goaltender":
                    color = self.config.goaltender_color
                else:
                    team_id = team_mapping.get(obj_id, 0)
                    color = self.config.team_colors.get(team_id, (255,255,255))
                
                cv2.ellipse(annotated, (x, y), (35, 18), 0, -45, 235, color, 2)
                
                if self.config.show_ids:
                    cv2.putText(annotated, str(obj_id), (x-10, y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
    
    def create_annotated_video(
        self,
        frames: List[np.ndarray],
        df: pd.DataFrame,
        team_mapping: Dict[int, int],
        output_path: str,
        fps: int,
        masks_per_frame: Optional[List[Dict]] = None
    ) -> str:
        """Create annotated video."""
        print(f"Creating video: {output_path}")
        
        annotated_frames = []
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f"  Rendering {i}/{len(frames)}")
            
            masks = masks_per_frame[i] if masks_per_frame else None
            
            if i in df.index:
                annotated = self.draw_from_dataframe(frame, i, df, team_mapping, masks)
            else:
                annotated = frame.copy()
            
            annotated_frames.append(annotated)
        
        # Write
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for f in annotated_frames:
            out.write(f)
        
        out.release()
        print(f"✓ Saved: {output_path}")
        
        return output_path
