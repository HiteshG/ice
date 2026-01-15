# -*- coding: utf-8 -*-
"""
EagleV2 - Football Tracker with Multi-Mode Support

Tracking Modes:
┌────────────────┬───────────┬─────────────┬───────────┬────────────┐
│ MODE           │ SPEED     │ OCCLUSION   │ REID      │ GPU MEM    │
├────────────────┼───────────┼─────────────┼───────────┼────────────┤
│ botsort        │ ⚡ Fast   │ ⚠️ Basic    │ ✅ Yes    │ ~2GB       │
│ sam2_cutie     │ 🐢 Slow   │ ✅ Good     │ ❌ No     │ ~6-8GB     │
│ hybrid         │ 🐌 Slowest│ ✅ Best     │ ✅ Yes    │ ~8-10GB    │
└────────────────┴───────────┴─────────────┴───────────┴────────────┘

Switch modes by changing TRACKING_MODE below!
"""

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================================

print("🔧 Installing dependencies...\n")

# Core
!pip install -q torch torchvision
!pip install -q ultralytics
!pip install -q boxmot
!pip install -q opencv-python-headless
!pip install -q pandas scikit-learn

# Team assignment
!pip install -q transformers umap-learn pillow

# SAM2 + CUTIE (for sam2_cutie and hybrid modes)
print("\n📦 Installing SAM2 + CUTIE...")
!pip install -q git+https://github.com/facebookresearch/sam2.git
!pip install -q git+https://github.com/hkchengrex/Cutie.git

# Utilities
!pip install -q tqdm matplotlib

print("\n✅ All dependencies installed!")

# ============================================================================
# CELL 2: CLONE REPOSITORY
# ============================================================================

import os
import sys

GITHUB_REPO = "https://github.com/HiteshG/Eaglevision.git"

print("📦 Cloning repository...\n")

if os.path.exists('Eaglevision'):
    !rm -rf Eaglevision

!git clone {GITHUB_REPO}

sys.path.insert(0, '/content/Eaglevision')

print("\n✅ Repository cloned!")

# ============================================================================
# CELL 3: SET PATHS
# ============================================================================

MODEL_PATH = "/content/yolov11l.pt"
VIDEO_PATH = "/content/clip_2.mp4"

# Get video info
import cv2
cap = cv2.VideoCapture(VIDEO_PATH)
native_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / native_fps
cap.release()

print(f"📊 Video: {os.path.basename(VIDEO_PATH)}")
print(f"   Resolution: {width}x{height}")
print(f"   Duration: {duration:.1f}s ({frame_count} frames @ {native_fps:.1f} FPS)")

# ============================================================================
# CELL 4: CONFIGURATION - CHANGE MODE HERE!
# ============================================================================

import torch

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    🎯 TRACKING MODE SELECTION                            ║
# ║                                                                          ║
# ║  Change this to switch between tracking methods:                         ║
# ║                                                                          ║
# ║  "botsort"     → Fast, uses ReID (DEFAULT)                              ║
# ║  "sam2_cutie"  → Mask-based, better occlusion handling                  ║
# ║  "hybrid"      → Both combined, best accuracy                           ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRACKING_MODE = "botsort"  # 👈 CHANGE THIS: "botsort" | "sam2_cutie" | "hybrid"

# ============================================================================
# Other settings
# ============================================================================

PROCESSING_FPS = 25
CONFIDENCE = 0.35
TEAM_METHOD = "color"  # "color" or "embedding"

# Visualization
SHOW_BBOXES = True
SHOW_IDS = True
SHOW_MASKS = False  # Set True to see mask overlays (only for sam2_cutie/hybrid)

# SAM2 + CUTIE settings (only used in sam2_cutie/hybrid modes)
SAM2_MODEL = "facebook/sam2-hiera-large"  # or "facebook/sam2-hiera-base-plus"
CUTIE_MODEL = "cutie-base"  # or "cutie-small"
CUTIE_MEM_EVERY = 5
MAX_LOST_FRAMES = 30

# Class names (adjust for your YOLO model)
CLASS_NAMES = {
    0: "Ball",
    1: "Goalkeeper",
    2: "Player",
    3: "Referee"
}

# ============================================================================
# Create configuration
# ============================================================================

from config import MainConfig, get_colab_config, print_mode_comparison

# Show mode comparison
print_mode_comparison()

# Get optimized config for Colab
config = get_colab_config(mode=TRACKING_MODE)

# Override with our settings
config.detector.model_path = MODEL_PATH
config.detector.confidence_threshold = CONFIDENCE
config.detector.class_names = CLASS_NAMES
config.fps = PROCESSING_FPS
config.team_assigner.team_method = TEAM_METHOD
config.visualizer.show_bboxes = SHOW_BBOXES
config.visualizer.show_ids = SHOW_IDS
config.visualizer.show_masks = SHOW_MASKS

# SAM2/CUTIE settings
if TRACKING_MODE in ["sam2_cutie", "hybrid"]:
    config.tracking.sam2_model = SAM2_MODEL
    config.tracking.cutie_model = CUTIE_MODEL
    config.tracking.cutie_mem_every = CUTIE_MEM_EVERY
    config.tracking.max_lost_frames = MAX_LOST_FRAMES

# ============================================================================
# Display configuration
# ============================================================================

print("\n" + "="*60)
print("⚙️ CONFIGURATION")
print("="*60)

print(f"""
📹 Video: {os.path.basename(VIDEO_PATH)}
🎯 Model: {os.path.basename(MODEL_PATH)}

🔧 TRACKING MODE: {TRACKING_MODE.upper()}
""")

if TRACKING_MODE == "botsort":
    print("""   ⚡ Fast processing
   ✅ ReID appearance features
   ⚠️ Basic occlusion handling
""")
elif TRACKING_MODE == "sam2_cutie":
    print(f"""   🎭 Mask-based tracking
   ✅ Better occlusion handling
   ❌ No ReID features
   📦 SAM2: {SAM2_MODEL}
   📦 CUTIE: {CUTIE_MODEL}
""")
else:  # hybrid
    print(f"""   🎭 Combined tracking
   ✅ Best occlusion handling
   ✅ ReID features
   📦 SAM2: {SAM2_MODEL}
   📦 CUTIE: {CUTIE_MODEL}
""")

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    if TRACKING_MODE in ["sam2_cutie", "hybrid"] and gpu_mem < 6:
        print(f"⚠️ Warning: Low GPU memory for {TRACKING_MODE} mode")
        print(f"   Consider using 'botsort' mode or smaller models")
else:
    print("⚠️ No GPU - using CPU (slow)")
    if TRACKING_MODE != "botsort":
        print(f"   Forcing 'botsort' mode for CPU")
        config.tracking.mode = "botsort"

print("="*60)
print("\n✅ Ready to process!")

# ============================================================================
# CELL 5: PROCESS VIDEO
# ============================================================================

from main import FootballTracker
import time

print(f"\n🚀 Starting [{TRACKING_MODE.upper()}] tracking...")
print("="*60 + "\n")

start_time = time.time()

try:
    tracker = FootballTracker(config)
    output_dir = tracker.process_video(VIDEO_PATH)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"\n⏱️ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"📁 Output: {output_dir}")
    
    # List files
    import glob
    print("\n📊 Output files:")
    for f in sorted(glob.glob(f"{output_dir}/*")):
        size = os.path.getsize(f) / (1024*1024)
        print(f"   - {os.path.basename(f)} ({size:.2f} MB)")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n📋 Troubleshooting:")
    print("   1. Try mode='botsort' (fastest, most stable)")
    print("   2. Reduce FPS")
    print("   3. Use smaller models")
    import traceback
    traceback.print_exc()
    raise

# ============================================================================
# CELL 6: PREVIEW RESULTS
# ============================================================================

import matplotlib.pyplot as plt

annotated_path = os.path.join(output_dir, "annotated.mp4")
cap = cv2.VideoCapture(annotated_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [int(i * total_frames / 5) for i in range(1, 5)]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, frame_num in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(frame_rgb)
        axes[idx].set_title(f"Frame {frame_num}")
        axes[idx].axis('off')

cap.release()
plt.suptitle(f"Results [{TRACKING_MODE.upper()} Mode]", fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 7: VIEW STATISTICS
# ============================================================================

import json

# Load metadata
with open(os.path.join(output_dir, "metadata.json"), 'r') as f:
    metadata = json.load(f)

# Load tracking info
tracking_info_path = os.path.join(output_dir, "tracking_info.json")
if os.path.exists(tracking_info_path):
    with open(tracking_info_path, 'r') as f:
        tracking_info = json.load(f)
else:
    tracking_info = {}

print("📈 TRACKING STATISTICS")
print("="*60)
print(f"Mode: {tracking_info.get('tracking_mode', TRACKING_MODE).upper()}")
print(f"Frames: {metadata['num_frames']}")
print(f"FPS: {metadata['fps']}")
print(f"Duration: {metadata['num_frames'] / metadata['fps']:.1f}s")

team_mapping = metadata['team_mapping']
team_counts = {}
for pid, tid in team_mapping.items():
    team_counts[tid] = team_counts.get(tid, 0) + 1

print(f"\nPlayers: {len(team_mapping)}")
for tid, count in sorted(team_counts.items()):
    print(f"  Team {tid}: {count} players")

if tracking_info.get('has_masks'):
    print(f"\n🎭 Masks: Available")
else:
    print(f"\n🎭 Masks: Not generated (botsort mode)")

# ============================================================================
# CELL 8: DOWNLOAD RESULTS
# ============================================================================

from google.colab import files

print("📥 Preparing download...")
!zip -r results.zip {output_dir}

files.download("results.zip")
print("\n✅ Download started!")

# ============================================================================
# CELL 9: COMPARE MODES (OPTIONAL)
# ============================================================================

"""
# Uncomment to run comparison between all modes

MODES_TO_TEST = ["botsort", "sam2_cutie", "hybrid"]
results = {}

for mode in MODES_TO_TEST:
    print(f"\n{'='*60}")
    print(f"Testing mode: {mode.upper()}")
    print('='*60)
    
    # Create config for this mode
    test_config = get_colab_config(mode=mode)
    test_config.detector.model_path = MODEL_PATH
    test_config.fps = PROCESSING_FPS
    test_config.output_dir = f"output_{mode}"
    
    # Process
    start = time.time()
    tracker = FootballTracker(test_config)
    out_dir = tracker.process_video(VIDEO_PATH)
    elapsed = time.time() - start
    
    # Store results
    results[mode] = {
        "time": elapsed,
        "output": out_dir
    }
    
    print(f"✓ {mode}: {elapsed:.1f}s")

# Print comparison
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
for mode, data in results.items():
    print(f"{mode:12s}: {data['time']:.1f}s")
"""

# ============================================================================
# CELL 10: CLEANUP (OPTIONAL)
# ============================================================================

"""
# Uncomment to cleanup
!rm -rf output_*
!rm results.zip
print("✅ Cleanup complete")
"""
