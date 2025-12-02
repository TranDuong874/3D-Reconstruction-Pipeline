from kornia.feature import DISK
from data_classes.data_classes import Frame, Features
import torch
import numpy as np
from tqdm import tqdm

class FeatureExtractor:   
    def __init__(self, method: str = 'disk', max_features: int = 4096, device: str = 'cuda', batch_size: int = 4):
        import torch
        from kornia.feature import DISK
        
        self.method = method
        self.max_features = max_features
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        
        if method == 'disk':
            self.detector = DISK.from_pretrained('depth').to(self.device)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'disk'")
    
    def extract(self, frame: Frame) -> Features:
        """Extract features from a single frame"""
        import torch
        
        # Convert to tensor (B, C, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(frame.image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            feats_list = self.detector(img_tensor)
            feats = feats_list[0]  # DISK returns a list, get first element
            keypoints = feats.keypoints.cpu().numpy()  # (N, 2)
            descriptors = feats.descriptors.cpu().numpy()  # (N, D)
            scores = feats.detection_scores.cpu().numpy()  # (N,)
        
        return Features(keypoints=keypoints, descriptors=descriptors, scores=scores)
    
    def extract_batch(self, frames: list[Frame]) -> dict[int, Features]:
        """Extract features from multiple frames in batches (faster GPU processing)"""
        features = {}
        
        for i in tqdm(range(0, len(frames), self.batch_size), desc="Extracting"):
            batch_frames = frames[i:i+self.batch_size]
            
            # Stack frames into a batch
            img_tensors = []
            for frame in batch_frames:
                img_tensor = torch.from_numpy(frame.image).permute(2, 0, 1).float() / 255.0
                img_tensors.append(img_tensor)
            
            # Pad to same size if needed, or process individually
            img_batch = torch.stack(img_tensors).to(self.device)
            
            with torch.no_grad():
                feats_list = self.detector(img_batch)
            
            # Extract features from batch
            for j, frame in enumerate(batch_frames):
                feats = feats_list[j]
                keypoints = feats.keypoints.cpu().numpy()  # (N, 2)
                descriptors = feats.descriptors.cpu().numpy()  # (N, D)
                scores = feats.detection_scores.cpu().numpy()  # (N,)
                features[frame.idx] = Features(keypoints=keypoints, descriptors=descriptors, scores=scores)
        
        return features
    
    def extract_all(self, frames: list[Frame]) -> dict[int, Features]:
        """Extract features from all frames using batch processing"""
        return self.extract_batch(frames)
