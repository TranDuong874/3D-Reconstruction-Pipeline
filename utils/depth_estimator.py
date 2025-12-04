"""
Depth Estimation Module

Uses Depth Anything V2 for metric depth estimation.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    """Depth Anything V2 Metric Depth Estimator"""
    
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
                 device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"  Loading Depth Anything V2 on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for a single image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Depth map as numpy array (H, W) in meters
        """
        H, W = image.shape[:2]
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        
        # Resize to original size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return depth
    
    def estimate_batch(self, images: list) -> list:
        """
        Estimate depth maps for multiple images.
        
        Args:
            images: List of RGB images as numpy arrays
        
        Returns:
            List of depth maps
        """
        depths = []
        for image in images:
            depth = self.estimate(image)
            depths.append(depth)
        return depths


def estimate_depth_scale(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                         sparse_points: np.ndarray, width: int, height: int) -> float:
    """
    Estimate scale factor to align predicted depth with SfM sparse points.
    Projects sparse points into image, compares depths, returns median scale.
    
    Args:
        depth: Predicted depth map (H, W)
        K: Camera intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        sparse_points: Sparse 3D points from SfM (N, 3)
        width, height: Image dimensions
    
    Returns:
        Scale factor
    """
    if len(sparse_points) == 0:
        return 1.0
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Transform sparse points to camera coordinates: P_cam = R @ P_world + t
    points_cam = (R @ sparse_points.T + t.reshape(3, 1)).T
    
    # Filter points in front of camera
    valid = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid]
    if len(points_cam) == 0:
        return 1.0
    
    # Project to image
    u = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    v = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    z_sfm = points_cam[:, 2]
    
    # Filter points within image bounds
    valid = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    u, v, z_sfm = u[valid], v[valid], z_sfm[valid]
    if len(u) < 5:
        return 1.0
    
    # Sample predicted depth at these locations
    z_pred = depth[v.astype(int), u.astype(int)]
    
    # Filter valid depths
    valid = (z_pred > 0.1) & np.isfinite(z_pred) & (z_sfm > 0.1)
    z_pred, z_sfm = z_pred[valid], z_sfm[valid]
    if len(z_pred) < 3:
        return 1.0
    
    # Scale: z_sfm = scale * z_pred, use median for robustness
    scale = np.median(z_sfm / z_pred)
    return scale
