"""Utility functions."""
import cv2
import os
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def read_video(video_path: str, fps: int = 24) -> Tuple[List[np.ndarray], int]:
    """Read video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    skip = max(1, int(native_fps / fps))
    actual_fps = native_fps / skip
    
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            frames.append(frame)
        count += 1
    
    cap.release()
    print(f"  Read {len(frames)} frames at {actual_fps:.1f} FPS")
    
    return frames, int(actual_fps)


def write_video(frames: List[np.ndarray], path: str, fps: int) -> str:
    """Write video file."""
    if not frames:
        raise ValueError("No frames")
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for f in frames:
        out.write(f)
    
    out.release()
    return path


def save_tracking_data(
    df: pd.DataFrame,
    team_mapping: Dict[int, int],
    output_dir: str,
    fps: int
):
    """Save tracking data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata
    meta = {
        "fps": fps,
        "num_frames": len(df),
        "team_mapping": {str(k): int(v) for k, v in team_mapping.items()}
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Raw data - convert tuples to lists for JSON
    raw_data = []
    for idx in df.index:
        row_data = {"frame": int(idx)}
        for col in df.columns:
            val = df.loc[idx, col]
            if pd.notna(val):
                row_data[col] = list(val) if isinstance(val, tuple) else val
        raw_data.append(row_data)
    
    with open(os.path.join(output_dir, "raw_data.json"), "w") as f:
        json.dump(raw_data, f, indent=2)
    
    # Processed data
    processed = []
    for idx in df.index:
        frame_data = {
            "frame": int(idx),
            "time": f"{idx // fps // 60:02d}:{idx // fps % 60:02d}",
            "detections": []
        }
        
        row = df.loc[idx]
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            
            if col == "Ball":
                det = {"id": "Ball", "type": "Ball", "x": float(val[0]), "y": float(val[1])}
            else:
                parts = col.split("_")
                oid = int(parts[1])
                det = {
                    "id": oid,
                    "type": parts[0],
                    "team": team_mapping.get(oid, -1),
                    "x": float(val[0]),
                    "y": float(val[1])
                }
            frame_data["detections"].append(det)
        
        processed.append(frame_data)
    
    with open(os.path.join(output_dir, "processed_data.json"), "w") as f:
        json.dump(processed, f, indent=2)


def load_tracking_data(output_dir: str) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """Load tracking data."""
    with open(os.path.join(output_dir, "metadata.json")) as f:
        meta = json.load(f)
    
    fps = meta["fps"]
    team_mapping = {int(k): int(v) for k, v in meta["team_mapping"].items()}
    
    with open(os.path.join(output_dir, "raw_data.json")) as f:
        raw_data = json.load(f)
    
    # Reconstruct DataFrame
    data = {}
    for row in raw_data:
        idx = row["frame"]
        data[idx] = {}
        for k, v in row.items():
            if k != "frame":
                data[idx][k] = tuple(v) if isinstance(v, list) else v
    
    df = pd.DataFrame(data).T
    
    return df, team_mapping, fps


def create_output_directory(video_path: str, base_dir: str = "output") -> str:
    """Create output directory."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def print_summary(df: pd.DataFrame, team_mapping: Dict[int, int], fps: int):
    """Print tracking summary."""
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    duration = len(df) / fps
    print(f"Duration: {duration:.1f}s ({len(df)} frames)")
    
    players = [c for c in df.columns if "Player" in c or "Goalkeeper" in c]
    print(f"Players: {len(players)}")
    
    teams = {}
    for col in players:
        pid = int(col.split("_")[1])
        tid = team_mapping.get(pid, -1)
        teams[tid] = teams.get(tid, 0) + 1
    
    for tid, count in sorted(teams.items()):
        print(f"  Team {tid}: {count}")
    
    if "Ball" in df.columns:
        ball_frames = df["Ball"].notna().sum()
        print(f"Ball: {ball_frames}/{len(df)} frames ({100*ball_frames/len(df):.1f}%)")
    
    print("="*50)
