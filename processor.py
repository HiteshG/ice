"""Data Processor Module."""
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple


class DataProcessor:
    """Processes tracking data."""
    
    def __init__(self, config, fps: int):
        self.config = config
        self.fps = fps
    
    def process(
        self,
        detections_per_frame: List[Dict],
        team_mapping: Dict[int, int]
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """Process detection data."""
        df = self._create_dataframe(detections_per_frame)
        
        if df.empty:
            return df, team_mapping
        
        if "Ball" in df.columns:
            df = self._interpolate(df, "Ball", fill=True)
        
        df, team_mapping = self._merge_ids(df, team_mapping)
        
        for col in df.columns:
            if col != "Ball":
                df = self._interpolate(df, col, fill=False)
                if self.config.smooth:
                    df = self._smooth(df, col)
        
        return df, team_mapping
    
    def _create_dataframe(self, detections_per_frame: List[Dict]) -> pd.DataFrame:
        data = {}
        
        for idx, dets in enumerate(detections_per_frame):
            frame_data = {}
            has_person = False
            
            for cls in ["Player", "Goalkeeper"]:
                if cls not in dets:
                    continue
                for oid, det in dets[cls].items():
                    col = f"{cls}_{oid}"
                    frame_data[col] = tuple(det["bottom_center"])
                    has_person = True
            
            if "Ball" in dets and dets["Ball"]:
                best = max(dets["Ball"].values(), key=lambda x: x["confidence"])
                frame_data["Ball"] = tuple(best["bottom_center"])
            
            if has_person:
                data[idx] = frame_data
        
        df = pd.DataFrame(data).T
        
        if len(df) > 0:
            thresh = 0.01 * len(df)
            df = df.loc[:, df.notna().sum() >= thresh]
        
        return df
    
    def _interpolate(self, df: pd.DataFrame, col: str, fill: bool) -> pd.DataFrame:
        if col not in df.columns:
            return df
        
        s = df[col]
        x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)
        
        if fill:
            x = x.interpolate().bfill().ffill()
            y = y.interpolate().bfill().ffill()
        else:
            x = x.interpolate(limit_area="inside")
            y = y.interpolate(limit_area="inside")
        
        combined = pd.Series([
            (xi, yi) if not (math.isnan(xi) or math.isnan(yi)) else np.nan
            for xi, yi in zip(x, y)
        ], index=s.index)
        
        df[col] = combined
        return df
    
    def _smooth(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            return df
        
        s = df[col]
        x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)
        
        x.iloc[::2] = np.nan
        y.iloc[::2] = np.nan
        
        x = x.interpolate(limit_area="inside")
        y = y.interpolate(limit_area="inside")
        
        combined = pd.Series([
            (xi, yi) if not (math.isnan(xi) or math.isnan(yi)) else np.nan
            for xi, yi in zip(x, y)
        ], index=s.index)
        
        df[col] = combined
        return df
    
    def _merge_ids(
        self,
        df: pd.DataFrame,
        team_mapping: Dict[int, int]
    ) -> Tuple[pd.DataFrame, Dict[int, int]]:
        # Merge goalkeepers with players if same ID
        gk_cols = [c for c in df.columns if "Goalkeeper" in c]
        for col in gk_cols:
            pid = col.split("_")[1]
            pcol = f"Player_{pid}"
            if pcol in df.columns:
                df[col] = df[pcol].combine_first(df[col])
                df.drop(columns=[pcol], inplace=True)
        
        return df, team_mapping
