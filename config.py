"""
Configuration with Tracking Mode Selection
Supports switching between: botsort, sam2_cutie, hybrid
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
import torch


def get_device() -> str:
    """Auto-detect available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DetectorConfig:
    """Configuration for object detector."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    low_confidence_threshold: float = 0.15
    device: Optional[str] = None
    class_names: Optional[dict] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = get_device()
        
        if self.class_names is None:
            self.class_names = {
                0: "Player",
                1: "Goalkeeper", 
                2: "Ball",
                3: "Referee",
                4: "Staff"
            }


@dataclass 
class TrackingConfig:
    """
    Unified tracking configuration.
    
    Modes:
    - "botsort": BoTSORT only - fast, uses ReID appearance features
    - "sam2_cutie": SAM2 + CUTIE only - mask-based, better occlusion handling
    - "hybrid": Both combined - best accuracy, slowest
    """
    # Mode selection - THE KEY SETTING
    mode: str = "botsort"  # Options: "botsort", "sam2_cutie", "hybrid"
    
    # Device
    device: Optional[str] = None
    
    # BoTSORT settings (used in botsort and hybrid modes)
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    
    # SAM2 settings (used in sam2_cutie and hybrid modes)
    sam2_model: str = "facebook/sam2-hiera-large"
    # Options: "facebook/sam2-hiera-large", "facebook/sam2-hiera-base-plus"
    
    # CUTIE settings (used in sam2_cutie and hybrid modes)
    cutie_model: str = "cutie-base"  # Options: "cutie-base", "cutie-small"
    cutie_mem_every: int = 5  # Add to memory every N frames
    cutie_top_k: int = 30  # Top-k memory frames
    
    # Tracking thresholds
    mask_threshold: float = 0.5
    min_mask_area: int = 100
    iou_threshold: float = 0.3
    max_lost_frames: int = 30
    
    # Hybrid mode settings
    mask_weight: float = 0.3
    
    # Performance
    use_amp: bool = True
    
    def __post_init__(self):
        if self.device is None:
            device = get_device()
            # BoTSORT doesn't support MPS
            self.device = "cpu" if device == "mps" else device
        
        # Validate mode
        valid_modes = ["botsort", "sam2_cutie", "hybrid"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class TeamAssignerConfig:
    """Configuration for team assignment."""
    team_method: str = "color"  # "color" or "embedding"
    n_clusters: int = 2
    overlap_threshold: float = 0.35
    memory_decay_frames: int = 150
    
    # Embedding method settings
    embedding_batch_size: int = 256
    stride: int = 3
    shrink_scale: float = 0.7
    
    color_ranges: dict = None
    
    def __post_init__(self):
        if self.color_ranges is None:
            self.color_ranges = {
                "red": [(0, 100, 100), (10, 255, 255)],
                "red2": [(160, 100, 100), (179, 255, 255)],
                "orange": [(11, 100, 100), (25, 255, 255)],
                "yellow": [(26, 100, 100), (35, 255, 255)],
                "green": [(36, 100, 100), (85, 255, 255)],
                "cyan": [(86, 100, 100), (95, 255, 255)],
                "blue": [(96, 100, 100), (125, 255, 255)],
                "purple": [(126, 100, 100), (145, 255, 255)],
                "magenta": [(146, 100, 100), (159, 255, 255)],
                "white": [(0, 0, 200), (180, 30, 255)],
                "gray": [(0, 0, 50), (180, 30, 200)],
                "black": [(0, 0, 0), (180, 255, 50)],
            }


@dataclass
class ProcessorConfig:
    """Configuration for data processing."""
    interpolate: bool = True
    smooth: bool = False
    temporal_threshold_seconds: float = 1.1
    spatial_threshold_per_frame: float = 10.0


@dataclass
class VisualizerConfig:
    """Configuration for visualization."""
    show_ids: bool = True
    show_bboxes: bool = True
    show_ball: bool = True
    show_masks: bool = False  # Show segmentation mask overlays
    mask_alpha: float = 0.3
    
    team_colors: dict = None
    goalkeeper_color: tuple = (0, 255, 0)
    ball_color: tuple = (0, 255, 0)
    occlusion_color: tuple = (0, 165, 255)  # Orange
    
    def __post_init__(self):
        if self.team_colors is None:
            self.team_colors = {
                0: (0, 0, 255),    # Red
                1: (255, 0, 0),    # Blue
            }


@dataclass
class MainConfig:
    """Main configuration combining all components."""
    detector: DetectorConfig = None
    tracking: TrackingConfig = None  # NEW: Unified tracking config
    team_assigner: TeamAssignerConfig = None
    processor: ProcessorConfig = None
    visualizer: VisualizerConfig = None
    
    # Video processing
    fps: int = 24
    output_dir: str = "output"
    
    def __post_init__(self):
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.tracking is None:
            self.tracking = TrackingConfig()
        if self.team_assigner is None:
            self.team_assigner = TeamAssignerConfig()
        if self.processor is None:
            self.processor = ProcessorConfig()
        if self.visualizer is None:
            self.visualizer = VisualizerConfig()
    
    # Convenience property for tracking mode
    @property
    def tracking_mode(self) -> str:
        return self.tracking.mode
    
    @tracking_mode.setter
    def tracking_mode(self, value: str):
        self.tracking.mode = value


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_fast_config() -> MainConfig:
    """
    Fast configuration - BoTSORT only.
    Best for: Quick processing, limited GPU, real-time applications
    """
    config = MainConfig()
    config.tracking.mode = "botsort"
    config.detector.model_path = "yolov8n.pt"
    config.fps = 15  # Lower FPS for speed
    return config


def get_accurate_config() -> MainConfig:
    """
    Accurate configuration - SAM2 + CUTIE only.
    Best for: Dense scenes, many occlusions, offline processing
    """
    config = MainConfig()
    config.tracking.mode = "sam2_cutie"
    config.detector.model_path = "yolov8m.pt"
    config.tracking.sam2_model = "facebook/sam2-hiera-large"
    config.tracking.cutie_model = "cutie-base"
    config.visualizer.show_masks = True
    return config


def get_best_config() -> MainConfig:
    """
    Best quality configuration - Hybrid mode.
    Best for: Maximum accuracy, benchmarking, research
    """
    config = MainConfig()
    config.tracking.mode = "hybrid"
    config.detector.model_path = "yolov8l.pt"
    config.tracking.sam2_model = "facebook/sam2-hiera-large"
    config.tracking.cutie_model = "cutie-base"
    config.visualizer.show_masks = True
    return config


def get_colab_config(mode: str = "botsort") -> MainConfig:
    """
    Optimized configuration for Google Colab (T4 GPU).
    
    Args:
        mode: "botsort", "sam2_cutie", or "hybrid"
    """
    config = MainConfig()
    config.tracking.mode = mode
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if mode == "botsort":
            # BoTSORT: Use larger detection model
            config.detector.model_path = "yolov8m.pt"
            
        elif mode == "sam2_cutie":
            # SAM2 + CUTIE: Adjust based on GPU memory
            if gpu_mem < 8:
                config.tracking.sam2_model = "facebook/sam2-hiera-base-plus"
                config.tracking.cutie_model = "cutie-small"
                config.tracking.cutie_mem_every = 10
                config.detector.model_path = "yolov8n.pt"
            else:
                config.tracking.sam2_model = "facebook/sam2-hiera-large"
                config.tracking.cutie_model = "cutie-base"
                config.detector.model_path = "yolov8n.pt"
            
        else:  # hybrid
            # Hybrid: Conservative settings
            if gpu_mem < 12:
                config.tracking.sam2_model = "facebook/sam2-hiera-base-plus"
                config.tracking.cutie_model = "cutie-small"
                config.detector.model_path = "yolov8n.pt"
            else:
                config.tracking.sam2_model = "facebook/sam2-hiera-large"
                config.tracking.cutie_model = "cutie-base"
                config.detector.model_path = "yolov8m.pt"
        
        config.tracking.use_amp = True
        
    else:
        # CPU: Force BoTSORT
        print("⚠️ No GPU: Forcing BoTSORT mode")
        config.tracking.mode = "botsort"
        config.detector.model_path = "yolov8n.pt"
    
    return config


# ============================================================================
# MODE COMPARISON HELPER
# ============================================================================

def print_mode_comparison():
    """Print comparison of tracking modes."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TRACKING MODE COMPARISON                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  MODE         │ SPEED    │ OCCLUSION │ REID  │ GPU MEM │ BEST FOR            ║
║  ─────────────┼──────────┼───────────┼───────┼─────────┼──────────────────── ║
║  botsort      │ ⚡ Fast  │ ⚠️ Basic  │ ✅ Yes│ ~2GB    │ General use, real-  ║
║               │          │           │       │         │ time, limited GPU   ║
║  ─────────────┼──────────┼───────────┼───────┼─────────┼──────────────────── ║
║  sam2_cutie   │ 🐢 Slow  │ ✅ Good   │ ❌ No │ ~6-8GB  │ Dense scenes,       ║
║               │          │           │       │         │ many occlusions     ║
║  ─────────────┼──────────┼───────────┼───────┼─────────┼──────────────────── ║
║  hybrid       │ 🐌 Slowest│ ✅ Best  │ ✅ Yes│ ~8-10GB │ Maximum accuracy,   ║
║               │          │           │       │         │ benchmarking        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    config = MainConfig()
    config.tracking.mode = "botsort"     # Fast
    config.tracking.mode = "sam2_cutie"  # Better occlusion handling
    config.tracking.mode = "hybrid"      # Best accuracy
""")
