import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from data_classes.data_classes import Frame

class VideoLoader:   
    def __init__(self, subsample: int = 1, blur_threshold: float = 100.0, max_size: int = None):
        self.subsample = subsample
        self.blur_threshold = blur_threshold
        self.max_size = max_size  # Resize images if larger than this (longest edge)
    
    def load_video(self, filepath: str) -> list[Frame]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(filepath)
        frames = []
        idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if idx % self.subsample == 0:
                if self._check_quality(frame):
                    frames.append(Frame(
                        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        idx=len(frames),
                        path=filepath
                    ))
            idx += 1
        
        cap.release()
        return frames
    
    def load_images(self, folder: str, extensions: tuple = ('.jpg', '.png', '.jpeg')) -> list[Frame]:
        """Load frames from image sequence"""
        folder = Path(folder)
        image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])
        
        frames = []
        for i, path in enumerate(image_paths[::self.subsample]):
            img = cv2.imread(str(path))
            if img is not None and self._check_quality(img):
                img = self._resize_if_needed(img)
                frames.append(Frame(
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    idx=len(frames),
                    path=str(path)
                ))
        
        print(f"Loaded {len(frames)} frames")
        return frames
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if larger than max_size"""
        if self.max_size is None:
            return image
        
        h, w = image.shape[:2]
        if max(h, w) <= self.max_size:
            return image
        
        scale = self.max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _check_quality(self, image: np.ndarray) -> bool:
        """Check if image passes quality threshold (blur detection)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var >= self.blur_threshold