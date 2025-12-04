"""
Utils module for 3D reconstruction pipeline.

Submodules:
- video_loader: Load and subsample video frames
- feature_extractor: Extract DISK/SIFT features
- feature_matcher: Match features across frames
- depth_estimator: Depth Anything V2 metric depth estimation
- segmentation: Mask2Former semantic segmentation
- reprojection: Depth to point cloud conversion
"""

from .video_loader import VideoLoader
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .depth_estimator import DepthEstimator, estimate_depth_scale
from .segmentation import (
    SemanticSegmenter,
    clean_segmentation,
    get_label_statistics,
    colorize_segmentation,
    ADE20K_COLORS,
    ADE20K_CLASSES
)
from .reprojection import (
    depth_to_pointcloud,
    save_ply,
    load_sparse_points,
    load_camera_intrinsics,
    load_poses,
    quat_to_rotation_matrix
)

__all__ = [
    # Video loading
    'VideoLoader',
    # Feature extraction
    'FeatureExtractor',
    # Feature matching
    'FeatureMatcher',
    # Depth estimation
    'DepthEstimator',
    'estimate_depth_scale',
    # Segmentation
    'SemanticSegmenter',
    'clean_segmentation',
    'get_label_statistics',
    'colorize_segmentation',
    'ADE20K_COLORS',
    'ADE20K_CLASSES',
    # Reprojection
    'depth_to_pointcloud',
    'save_ply',
    'load_sparse_points',
    'load_camera_intrinsics',
    'load_poses',
    'quat_to_rotation_matrix',
]
