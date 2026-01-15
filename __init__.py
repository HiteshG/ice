"""
Football Tracker with Multi-Mode Support

Tracking Modes:
- botsort: Fast, uses ReID appearance features
- sam2_cutie: Mask-based, better occlusion handling
- hybrid: Both combined, best accuracy

Usage:
    from football_tracker import FootballTracker, MainConfig
    
    config = MainConfig()
    config.tracking.mode = "botsort"  # or "sam2_cutie" or "hybrid"
    
    tracker = FootballTracker(config)
    output_dir = tracker.process_video("video.mp4")
"""

from .config import (
    MainConfig,
    DetectorConfig,
    TrackingConfig,
    TeamAssignerConfig,
    ProcessorConfig,
    VisualizerConfig,
    get_fast_config,
    get_accurate_config,
    get_best_config,
    get_colab_config,
    print_mode_comparison
)

from .main import FootballTracker
from .unified_tracker import UnifiedTracker, create_tracker
from .utils import read_video, write_video, load_tracking_data

__version__ = "2.0.0"
__all__ = [
    "FootballTracker",
    "MainConfig",
    "TrackingConfig",
    "UnifiedTracker",
    "create_tracker",
    "get_fast_config",
    "get_accurate_config", 
    "get_best_config",
    "get_colab_config",
    "print_mode_comparison"
]
