import torch
import numpy as np
from tqdm import tqdm
from kornia.feature import DescriptorMatcher
from data_classes.data_classes import Features, Match


class FeatureMatcher:   
    def __init__(self, feature_type: str = 'disk', device: str = 'cuda', match_mode: str = 'snn', th: float = 0.8):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Use DescriptorMatcher with second nearest neighbor ratio test (GPU-accelerated)
        self.matcher = DescriptorMatcher(match_mode=match_mode, th=th).to(self.device)
    
    def match_pair(self, feat0: Features, feat1: Features, idx0: int, idx1: int) -> Match:
        """Match descriptors using GPU-accelerated nearest neighbor matching"""
        desc0 = torch.from_numpy(feat0.descriptors).float().to(self.device)
        desc1 = torch.from_numpy(feat1.descriptors).float().to(self.device)
        
        with torch.no_grad():
            # DescriptorMatcher returns (distances, match_indices)
            dists, match_idxs = self.matcher(desc0, desc1)
        
        # match_idxs is (num_desc0, 2) where [:, 0] is idx in desc0, [:, 1] is idx in desc1
        # Filter out invalid matches (marked as -1)
        valid_mask = match_idxs[:, 1] >= 0
        matches_np = match_idxs[valid_mask].cpu().numpy()
        confidence = dists[valid_mask].cpu().numpy().flatten() if dists.numel() > 0 else None
        
        return Match(idx0=idx0, idx1=idx1, matches=matches_np, confidence=confidence)
    
    def match_sequential(self, features: dict[int, Features], overlap: int = 10) -> list[Match]:
        """Match frames sequentially (for video)"""
        indices = sorted(features.keys())
        matches = []
        
        # Calculate total pairs for progress bar
        total_pairs = sum(min(overlap, len(indices) - i - 1) for i in range(len(indices)))
        
        with tqdm(total=total_pairs, desc="Matching") as pbar:
            for i, idx0 in enumerate(indices):
                for idx1 in indices[i+1 : i+1+overlap]:
                    match = self.match_pair(features[idx0], features[idx1], idx0, idx1)
                    if len(match.matches) > 0:
                        matches.append(match)
                    pbar.update(1)
        
        return matches
    
    def match_exhaustive(self, features: dict[int, Features]) -> list[Match]:
        """Match all pairs (for unordered images)"""
        indices = sorted(features.keys())
        matches = []
        
        for i, idx0 in enumerate(indices):
            for idx1 in indices[i+1:]:
                match = self.match_pair(features[idx0], features[idx1], idx0, idx1)
                if len(match.matches) > 0:
                    matches.append(match)
        
        return matches
