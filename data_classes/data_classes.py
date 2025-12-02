
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    video_path: str
    output_path: str
    feature_type: str = 'default'

@dataclass 
class Features:
    keypoints: np.ndarray  # (N, 2) x,y coordinates
    descriptors: np.ndarray  # (N, D) descriptor vectors
    scores: np.ndarray = None  # (N,) confidence scores

@dataclass
class Frame:
    image: np.ndarray
    idx: int
    path: str = None

@dataclass
class Match:
    idx0: int  # frame index 0
    idx1: int  # frame index 1
    matches: np.ndarray  # (M, 2) matched keypoint indices
    confidence: np.ndarray = None  # (M,) match confidence

