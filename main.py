"""
Football Tracker Main Pipeline
Supports 3 tracking modes: botsort, sam2_cutie, hybrid
"""
import os
import sys
import argparse
import time
from typing import List, Dict, Optional
import numpy as np


class FootballTracker:
    """
    Main tracking pipeline with mode switching support.
    
    Modes:
    - botsort: Fast, uses ReID (default)
    - sam2_cutie: Mask-based, better occlusion handling
    - hybrid: Both combined, best accuracy
    """
    
    def __init__(self, config=None):
        """
        Initialize tracker.
        
        Args:
            config: MainConfig object or None for defaults
        """
        from config import MainConfig
        
        self.config = config or MainConfig()
        
        print("\n" + "="*60)
        print("FOOTBALL TRACKER")
        print(f"Tracking Mode: {self.config.tracking.mode.upper()}")
        print("="*60 + "\n")
        
        # Initialize components
        self._init_detector()
        self._init_tracker()
        self._init_team_assigner()
        self._init_visualizer()
        
        # Processor initialized later with FPS
        self.processor = None
        
        print("\n✓ All components initialized!\n")
    
    def _init_detector(self):
        """Initialize object detector."""
        from detector import ObjectDetector
        self.detector = ObjectDetector(self.config.detector)
    
    def _init_tracker(self):
        """Initialize tracker based on mode."""
        from unified_tracker import UnifiedTracker, UnifiedTrackerConfig
        
        tracker_config = UnifiedTrackerConfig(
            mode=self.config.tracking.mode,
            device=self.config.tracking.device,
            reid_weights=self.config.tracking.reid_weights,
            sam2_model=self.config.tracking.sam2_model,
            cutie_model=self.config.tracking.cutie_model,
            cutie_mem_every=self.config.tracking.cutie_mem_every,
            cutie_top_k=self.config.tracking.cutie_top_k,
            mask_threshold=self.config.tracking.mask_threshold,
            min_mask_area=self.config.tracking.min_mask_area,
            iou_threshold=self.config.tracking.iou_threshold,
            max_lost_frames=self.config.tracking.max_lost_frames,
            mask_weight=self.config.tracking.mask_weight,
            use_amp=self.config.tracking.use_amp
        )
        
        self.tracker = UnifiedTracker(tracker_config)
    
    def _init_team_assigner(self):
        """Initialize team assigner."""
        from team_assigner import TeamAssigner
        self.team_assigner = TeamAssigner(self.config.team_assigner)
    
    def _init_visualizer(self):
        """Initialize visualizer."""
        from visualizer import Visualizer
        self.visualizer = Visualizer(self.config.visualizer)
    
    def process_video(self, video_path: str, output_dir: str = None) -> str:
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            Path to output directory
        """
        from utils import read_video, create_output_directory, save_tracking_data, print_summary
        from processor import DataProcessor
        
        # Create output directory
        if output_dir is None:
            output_dir = create_output_directory(video_path, self.config.output_dir)
        
        mode = self.config.tracking.mode.upper()
        print("\n" + "="*60)
        print(f"PROCESSING VIDEO [{mode} MODE]")
        print(f"Input: {os.path.basename(video_path)}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Step 1: Read video
        print("Step 1/5: Reading video...")
        frames, fps = read_video(video_path, self.config.fps)
        
        if not frames:
            raise ValueError("No frames read from video")
        
        print(f"  → {len(frames)} frames at {fps} FPS")
        
        # Initialize processor
        self.processor = DataProcessor(self.config.processor, fps)
        
        # Step 2: Detection and Tracking
        print(f"\nStep 2/5: Detecting and tracking [{mode}]...")
        detections_per_frame, masks_per_frame = self._detect_and_track(frames)
        
        # Step 3: Team Assignment
        print("\nStep 3/5: Assigning teams...")
        team_mapping = self.team_assigner.assign_teams(frames, detections_per_frame)
        
        # Step 4: Data Processing
        print("\nStep 4/5: Processing tracking data...")
        df, team_mapping = self.processor.process(detections_per_frame, team_mapping)
        
        # Step 5: Save Results
        print("\nStep 5/5: Saving results...")
        
        # Save tracking data
        save_tracking_data(df, team_mapping, output_dir, fps)
        
        # Save mode info
        self._save_tracking_info(output_dir, masks_per_frame)
        
        # Create annotated video
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        self.visualizer.create_annotated_video(
            frames, df, team_mapping, annotated_path, fps,
            masks_per_frame if self.config.visualizer.show_masks else None
        )
        
        # Print summary
        elapsed = time.time() - start_time
        print_summary(df, team_mapping, fps)
        
        print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"📁 Results: {output_dir}")
        
        return output_dir
    
    def _detect_and_track(self, frames: List[np.ndarray]) -> tuple:
        """Run detection and tracking on all frames."""
        detections_per_frame = []
        masks_per_frame = []
        
        total = len(frames)
        
        for i, frame in enumerate(frames):
            if i % 50 == 0 or i == total - 1:
                print(f"  Frame {i+1}/{total}")
            
            # Detect
            boxes, confidences, class_labels = self.detector.detect(frame)
            
            # Prepare for tracker
            detection_array = self.detector.get_detection_array(
                boxes, confidences, class_labels
            )
            
            # Track
            tracks = self.tracker.update(detection_array, frame)
            
            # Organize
            frame_detections = self.tracker.organize_tracks(
                tracks,
                self.detector.CLASS_NAMES,
                self.config.detector.confidence_threshold,
                frame.shape[:2]
            )
            
            # Get ball (not tracked)
            ball_dets = self.detector.filter_detections(
                boxes, confidences, class_labels, frame.shape[:2]
            ).get("Ball", {})
            frame_detections["Ball"] = ball_dets
            
            detections_per_frame.append(frame_detections)
            
            # Store masks
            masks = self.tracker.get_masks()
            masks_per_frame.append(masks)
        
        return detections_per_frame, masks_per_frame
    
    def _save_tracking_info(self, output_dir: str, masks_per_frame: List):
        """Save tracking mode info and statistics."""
        import json
        
        info = {
            "tracking_mode": self.config.tracking.mode,
            "tracker_stats": self.tracker.get_stats(),
            "has_masks": len(masks_per_frame) > 0 and any(masks_per_frame)
        }
        
        info_path = os.path.join(output_dir, "tracking_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
    
    def switch_mode(self, mode: str):
        """
        Switch tracking mode at runtime.
        
        Args:
            mode: "botsort", "sam2_cutie", or "hybrid"
        """
        self.config.tracking.mode = mode
        self.tracker.set_mode(mode)
        print(f"✓ Switched to {mode.upper()} mode")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Football Player Tracker with Multi-Mode Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tracking Modes:
  botsort      Fast, uses ReID appearance features (default)
  sam2_cutie   Mask-based, better occlusion handling  
  hybrid       Both combined, best accuracy

Examples:
  python main.py --video match.mp4 --mode botsort
  python main.py --video match.mp4 --mode sam2_cutie --show-masks
  python main.py --video match.mp4 --mode hybrid
        """
    )
    
    # Required
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="botsort",
        choices=["botsort", "sam2_cutie", "hybrid"],
        help="Tracking mode (default: botsort)"
    )
    
    # Optional
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--detector-conf", type=float, default=0.35)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--show-bboxes", action="store_true")
    parser.add_argument("--show-masks", action="store_true")
    
    # SAM2/CUTIE specific
    parser.add_argument("--sam2-model", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--cutie-model", type=str, default="cutie-base")
    
    args = parser.parse_args()
    
    # Create config
    from config import MainConfig
    
    config = MainConfig()
    config.fps = args.fps
    config.tracking.mode = args.mode
    config.detector.model_path = args.model
    config.detector.confidence_threshold = args.detector_conf
    config.visualizer.show_bboxes = args.show_bboxes
    config.visualizer.show_masks = args.show_masks
    
    if args.no_gpu:
        config.detector.device = "cpu"
        config.tracking.device = "cpu"
    
    if args.mode in ["sam2_cutie", "hybrid"]:
        config.tracking.sam2_model = args.sam2_model
        config.tracking.cutie_model = args.cutie_model
    
    # Run
    try:
        tracker = FootballTracker(config)
        output_dir = tracker.process_video(args.video, args.output_dir)
        
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\nOutput: {output_dir}")
        print("\nFiles:")
        print("  - annotated.mp4")
        print("  - raw_data.json")
        print("  - processed_data.json") 
        print("  - metadata.json")
        print("  - tracking_info.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
