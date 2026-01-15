"""Team Assignment Module for Ice Hockey."""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import Dict, List, Tuple


class TeamAssigner:
    """Color-based team assignment for ice hockey."""
    
    def __init__(self, config):
        self.config = config
        self.color_ranges = config.color_ranges
    
    def assign_teams(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[Dict]
    ) -> Dict[int, int]:
        """Assign teams to players."""
        print("Assigning teams...")
        
        player_color_counts = {}
        
        for frame, detections in zip(frames, detections_per_frame):
            if "Player" not in detections:
                continue
            
            players = detections["Player"]
            all_bboxes = [item["bbox"] for item in players.values()]
            
            for player_id, detection in players.items():
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                overlap = self._calc_overlap(bbox, all_bboxes)
                if overlap > self.config.overlap_threshold:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                colors = self._detect_colors(crop)
                
                if player_id not in player_color_counts:
                    player_color_counts[player_id] = {}
                
                for color, count in colors:
                    if color not in player_color_counts[player_id]:
                        player_color_counts[player_id][color] = 0
                    player_color_counts[player_id][color] += count * (1 - overlap)
        
        # Find dominant colors
        player_colors = {}
        for pid, counts in player_color_counts.items():
            if counts:
                player_colors[pid] = max(counts, key=counts.get)
        
        all_colors = list(player_colors.values())
        if len(all_colors) < 2:
            return {pid: 0 for pid in player_colors}
        
        freq = Counter(all_colors)
        team_colors = [c for c, _ in freq.most_common(2)]
        color_to_team = {c: i for i, c in enumerate(team_colors)}
        
        # Assign teams
        team_mapping = {}
        for pid, color in player_colors.items():
            if color in color_to_team:
                team_mapping[pid] = color_to_team[color]
            else:
                counts = player_color_counts[pid]
                team_counts = [(c, counts.get(c, 0)) for c in team_colors]
                if team_counts:
                    best = max(team_counts, key=lambda x: x[1])[0]
                    team_mapping[pid] = color_to_team.get(best, 0)
                else:
                    team_mapping[pid] = 0
        
        t0 = sum(1 for t in team_mapping.values() if t == 0)
        t1 = sum(1 for t in team_mapping.values() if t == 1)
        print(f"  Team 0: {t0}, Team 1: {t1}")
        
        return team_mapping
    
    def _calc_overlap(self, bbox, all_bboxes) -> float:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            return 0.0
        
        max_overlap = 0.0
        for other in all_bboxes:
            if other == bbox:
                continue
            ox1, oy1, ox2, oy2 = other
            ix = max(0, min(x2, ox2) - max(x1, ox1))
            iy = max(0, min(y2, oy2) - max(y1, oy1))
            overlap = (ix * iy) / area
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _detect_colors(self, image: np.ndarray) -> List[Tuple[str, int]]:
        if image.size == 0:
            return []
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=3)
        kmeans.fit(rgb.reshape(-1, 3))
        labels = kmeans.labels_.reshape(image.shape[:2])
        
        corners = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        bg = max(set(corners), key=corners.count)
        fg = 1 if bg == 0 else 0
        
        mask = (labels == fg).astype(np.uint8) * 255
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        counts = {}
        for name, (lo, hi) in self.color_ranges.items():
            lo = np.array(lo, dtype=np.uint8)
            hi = np.array(hi, dtype=np.uint8)
            cmask = cv2.inRange(hsv, lo, hi)
            cmask = cv2.bitwise_and(cmask, mask)
            c = cv2.countNonZero(cmask)
            if c > 0:
                counts[name] = c
        
        if "red2" in counts:
            counts["red"] = counts.get("red", 0) + counts.pop("red2")
        
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)
