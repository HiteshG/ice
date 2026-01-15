# Ice Hockey Tracker

Multi-mode tracking system for ice hockey games with support for players, goaltenders, puck, and referees.

## 🏒 Features

- **Multi-Mode Tracking**: Choose between `botsort`, `sam2_cutie`, or `hybrid` modes
- **Ice Hockey Specific**: Optimized for hockey classes (Player, Goaltender, Puck, Referee, etc.)
- **Team Assignment**: Automatic team detection based on jersey colors
- **Puck Tracking**: Specialized handling for fast-moving puck
- **Occlusion Handling**: Advanced mask-based tracking for crowded scenes

## 📋 Class Names

The system is configured for the following ice hockey classes:

```python
CLASS_NAMES = {
    0: "Center Ice",
    1: "Faceoff",
    2: "Goalpost",
    3: "Goaltender",
    4: "Player",
    5: "Puck",
    6: "Referee"
}
```

## 🚀 Quick Start

### 1. Installation

```bash
# Core dependencies
pip install torch torchvision
pip install ultralytics
pip install boxmot
pip install opencv-python
pip install pandas scikit-learn

# For mask-based tracking (sam2_cutie and hybrid modes)
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/hkchengrex/Cutie.git
```

### 2. Test the System

```bash
python test_hockey.py
```

This will verify:
- ✓ Configuration is set up correctly
- ✓ All modules can be imported
- ✓ Tracker can be initialized

### 3. Run Tracking

#### Command Line

```bash
# Basic usage with BoTSORT (fastest)
python main.py --video hockey_game.mp4

# With SAM2 + CUTIE (better occlusion handling)
python main.py --video hockey_game.mp4 --mode sam2_cutie --show-masks

# Hybrid mode (best accuracy)
python main.py --video hockey_game.mp4 --mode hybrid

# Custom settings
python main.py --video hockey_game.mp4 \
    --mode botsort \
    --fps 25 \
    --detector-conf 0.35 \
    --model yolov8m.pt \
    --show-bboxes
```

#### Python API

```python
from config import MainConfig
from main import HockeyTracker

# Create configuration
config = MainConfig()
config.tracking.mode = "botsort"  # or "sam2_cutie" or "hybrid"
config.detector.model_path = "path/to/your/model.pt"
config.fps = 25

# Ice hockey classes (already configured by default)
config.detector.class_names = {
    0: "Center Ice",
    1: "Faceoff",
    2: "Goalpost",
    3: "Goaltender",
    4: "Player",
    5: "Puck",
    6: "Referee"
}

# Initialize and run
tracker = HockeyTracker(config)
output_dir = tracker.process_video("hockey_game.mp4")
```

## 📊 Tracking Modes

| Mode | Speed | Occlusion | ReID | GPU Memory | Best For |
|------|-------|-----------|------|------------|----------|
| `botsort` | ⚡ Fast | ⚠️ Basic | ✅ Yes | ~2GB | General use, real-time |
| `sam2_cutie` | 🐢 Slow | ✅ Good | ❌ No | ~6-8GB | Dense scenes, occlusions |
| `hybrid` | 🐌 Slowest | ✅ Best | ✅ Yes | ~8-10GB | Maximum accuracy |

### Mode Selection Guide

**Use `botsort` when:**
- You need fast processing
- You have limited GPU memory
- Players are well-separated
- You need real-time performance

**Use `sam2_cutie` when:**
- Players frequently overlap/occlude each other
- You have time for offline processing
- You have 6-8GB+ GPU memory
- Appearance-based tracking is less reliable

**Use `hybrid` when:**
- You need the best possible accuracy
- You have 8-10GB+ GPU memory
- Processing time is not a concern
- You're doing research or benchmarking

## 🎯 YOLO Model Requirements

Your YOLO model must be trained to detect ice hockey classes. The model should output the class IDs matching:

- Class 3: Goaltender
- Class 4: Player  
- Class 5: Puck
- Class 6: Referee (optional)

### Training Your Own Model

If you need to train a custom model:

1. Annotate ice hockey images with the above classes
2. Train using Ultralytics YOLO:
   ```bash
   yolo train data=hockey.yaml model=yolov8m.pt epochs=100
   ```
3. Place the trained model in your project directory
4. Update the model path in config

## 🔧 Configuration Options

### Detection Settings

```python
config.detector.model_path = "yolov8m.pt"  # Path to YOLO model
config.detector.confidence_threshold = 0.35  # Confidence threshold for tracking
config.detector.low_confidence_threshold = 0.15  # For initial detection
config.detector.device = "cuda"  # "cuda", "cpu", or "mps"
```

### Tracking Settings

```python
config.tracking.mode = "botsort"  # "botsort", "sam2_cutie", or "hybrid"
config.tracking.max_lost_frames = 30  # Frames before track is lost
config.tracking.iou_threshold = 0.3  # IoU threshold for matching
```

### Visualization Settings

```python
config.visualizer.show_ids = True  # Show player IDs
config.visualizer.show_bboxes = True  # Show bounding boxes
config.visualizer.show_puck = True  # Show puck tracking
config.visualizer.show_masks = False  # Show segmentation masks
config.visualizer.team_colors = {
    0: (0, 0, 255),  # Red for team 0
    1: (255, 0, 0),  # Blue for team 1
}
config.visualizer.goaltender_color = (0, 255, 0)  # Green
config.visualizer.puck_color = (0, 255, 0)  # Green
```

## 📁 Output Files

After processing, you'll get:

- `annotated.mp4` - Video with tracking visualization
- `raw_data.json` - Raw tracking data per frame
- `processed_data.json` - Processed and interpolated data
- `metadata.json` - Video info and team assignments
- `tracking_info.json` - Tracking mode and statistics

## 🐛 Troubleshooting

### "No detections possible" / Empty tracking results

**Problem**: YOLO model not detecting ice hockey classes

**Solutions**:
1. Verify your YOLO model is trained for ice hockey
2. Lower confidence threshold: `--detector-conf 0.2`
3. Test model separately:
   ```python
   from ultralytics import YOLO
   model = YOLO("your_model.pt")
   results = model("test_frame.jpg")
   results[0].plot()  # Check what's detected
   ```

### GPU Memory Errors

**Problem**: Out of memory with `sam2_cutie` or `hybrid` modes

**Solutions**:
1. Use smaller models:
   ```bash
   --sam2-model facebook/sam2-hiera-base-plus
   --cutie-model cutie-small
   ```
2. Reduce FPS: `--fps 15`
3. Use `botsort` mode instead
4. Process shorter video clips

### Incorrect Team Assignments

**Problem**: Players assigned to wrong teams

**Solutions**:
1. Check color ranges in `config.team_assigner.color_ranges`
2. Ensure good lighting in video
3. Try increasing `overlap_threshold` in team assignment
4. Manually verify detected colors match team jerseys

### Puck Not Detected

**Problem**: Puck missing in many frames

**Solutions**:
1. The puck moves very fast - this is normal
2. Lower puck detection confidence: `config.detector.confidence_threshold = 0.2`
3. Check if your model is trained to detect pucks
4. Puck interpolation should fill gaps automatically

### Slow Processing

**Problem**: Tracking takes too long

**Solutions**:
1. Use `botsort` mode (fastest)
2. Reduce FPS: `--fps 15`
3. Use smaller YOLO model: `yolov8n.pt`
4. Reduce video resolution before processing
5. Use GPU if available

## 🔬 Advanced Usage

### Google Colab

Use the provided `eaglev2.py` notebook:

```python
# Change the tracking mode in Cell 4
TRACKING_MODE = "botsort"  # or "sam2_cutie" or "hybrid"

# Update class names
CLASS_NAMES = {
    0: "Center Ice",
    1: "Faceoff",
    2: "Goalpost",
    3: "Goaltender",
    4: "Player",
    5: "Puck",
    6: "Referee"
}
```

### Comparing Modes

```python
from config import get_colab_config
from main import HockeyTracker
import time

modes = ["botsort", "sam2_cutie", "hybrid"]
results = {}

for mode in modes:
    config = get_colab_config(mode=mode)
    config.detector.model_path = "your_model.pt"
    
    tracker = HockeyTracker(config)
    start = time.time()
    output_dir = tracker.process_video("hockey.mp4")
    elapsed = time.time() - start
    
    results[mode] = {"time": elapsed, "output": output_dir}
    print(f"{mode}: {elapsed:.1f}s")
```

## 📝 Key Differences from Football Tracker

This has been adapted from a football tracker with these key changes:

1. **Class Names**: Updated to ice hockey specific classes
2. **Ball → Puck**: Renamed and adjusted center point calculation
3. **Goalkeeper → Goaltender**: Updated terminology
4. **Puck Handling**: Center point instead of bottom center (puck is on ice)
5. **Visualization**: Updated colors and markers for hockey
6. **Documentation**: All references updated to ice hockey

## 🤝 Support

If you encounter issues:

1. Run `python test_hockey.py` to diagnose setup issues
2. Check that your YOLO model detects the correct classes
3. Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
4. Try `botsort` mode first (simplest and most stable)
5. Review error messages in the console output

## 📄 License

Same as original Eagle Vision project.
