"""
Depth Anything V2 Metric Depth + Mask2Former Segmentation + Reprojection

This script:
1. Loads registered images and camera poses from SfM output
2. Runs Depth Anything V2 metric depth model
3. Runs Mask2Former for semantic segmentation (with temporal consistency)
4. Estimates depth scale using sparse SfM points
5. Reprojects depth maps to 3D point clouds
6. Outputs RGB and semantic label PLY files

Temporal consistency is achieved by:
- Using first frame's Mask2Former output as reference
- Propagating labels using optical flow (RAFT) for subsequent frames
- Falling back to Mask2Former when flow confidence is low
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import Mask2FormerForUniversalSegmentation
import cv2
from tqdm import tqdm
import argparse
import struct
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


# ADE20K color palette (150 classes)
ADE20K_COLORS = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)

# ADE20K class names (150 classes)
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree",
    "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf",
    "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway",
    "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag",
]


def get_label_statistics(segmentation: np.ndarray) -> dict:
    """Get statistics of labels in a segmentation map."""
    unique, counts = np.unique(segmentation, return_counts=True)
    total = segmentation.size
    stats = {}
    for label_id, count in zip(unique, counts):
        if label_id < len(ADE20K_CLASSES):
            name = ADE20K_CLASSES[label_id]
        else:
            name = f"unknown_{label_id}"
        stats[name] = {
            "id": int(label_id),
            "count": int(count),
            "percentage": float(count / total * 100)
        }
    return stats


def clean_segmentation(segmentation: np.ndarray, image: np.ndarray = None,
                       min_area: int = 500, morph_size: int = 5,
                       use_bilateral: bool = True) -> np.ndarray:
    """
    Clean up segmentation artifacts using multiple techniques:
    1. Remove small connected components (noise)
    2. Morphological closing to fill small holes
    3. Morphological opening to remove small protrusions
    4. Optional: edge-aware bilateral filtering on label boundaries
    
    Args:
        segmentation: Input segmentation map (H, W), integer labels
        image: Original RGB image for edge-aware filtering (optional)
        min_area: Minimum area for a region to keep (pixels)
        morph_size: Size of morphological structuring element
        use_bilateral: Whether to use bilateral-like edge-aware smoothing
    
    Returns:
        Cleaned segmentation map
    """
    H, W = segmentation.shape
    cleaned = segmentation.copy()
    
    # Get unique labels
    unique_labels = np.unique(segmentation)
    
    # Process each label: remove small disconnected components
    for label in unique_labels:
        mask = (segmentation == label).astype(np.uint8)
        
        # Find connected components
        num_components, labels_map, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Remove small components (keep only large ones)
        for comp_id in range(1, num_components):  # Skip background (0)
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_area:
                # Replace small component with most common neighbor label
                comp_mask = (labels_map == comp_id)
                
                # Dilate to find neighbors
                dilated = cv2.dilate(comp_mask.astype(np.uint8), 
                                    np.ones((3, 3), np.uint8), iterations=2)
                neighbor_mask = dilated.astype(bool) & ~comp_mask
                
                if np.any(neighbor_mask):
                    neighbor_labels = segmentation[neighbor_mask]
                    # Exclude the current label
                    neighbor_labels = neighbor_labels[neighbor_labels != label]
                    if len(neighbor_labels) > 0:
                        # Get most common neighbor
                        most_common = np.bincount(neighbor_labels).argmax()
                        cleaned[comp_mask] = most_common
    
    # Morphological operations on each label to smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    
    # Create a smoothed version using mode filtering
    # This is more label-aware than simple morphology
    smoothed = cleaned.copy()
    
    # Apply median-like filtering using a sliding window approach
    # Use morphological closing then opening on each label
    for label in unique_labels:
        mask = (cleaned == label).astype(np.uint8)
        # Close (fill small holes)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Open (remove small protrusions)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        # Only update where this creates a valid region
        smoothed[opened == 1] = label
    
    # Edge-aware refinement using image edges (if image provided)
    if use_bilateral and image is not None:
        # Use guided filter-like approach: respect image edges
        # Convert to joint bilateral style by using image gradients
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create edge zone
        edge_zone = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        
        # In edge zones, prefer the original (unsmoothed) segmentation
        # as it may better respect object boundaries
        final = smoothed.copy()
        final[edge_zone] = cleaned[edge_zone]
        
        return final.astype(np.int32)
    
    return smoothed.astype(np.int32)


def apply_crf_refinement(segmentation: np.ndarray, image: np.ndarray,
                         n_classes: int = 150, 
                         sxy_gaussian: int = 3, compat_gaussian: int = 3,
                         sxy_bilateral: int = 80, srgb_bilateral: int = 13,
                         compat_bilateral: int = 10, n_iterations: int = 5) -> np.ndarray:
    """
    Apply Dense CRF refinement to segmentation.
    Uses pydensecrf if available, otherwise falls back to simple refinement.
    
    This encourages:
    - Spatially close pixels to have the same label
    - Pixels with similar colors to have the same label
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
        
        H, W = segmentation.shape
        n_labels = n_classes
        
        # Create unary potentials from current segmentation
        # Add small probability for other classes to allow refinement
        unary = unary_from_labels(segmentation.flatten(), n_labels, gt_prob=0.7, zero_unsure=False)
        
        # Create CRF
        d = dcrf.DenseCRF2D(W, H, n_labels)
        d.setUnaryEnergy(unary)
        
        # Pairwise Gaussian (spatial smoothness)
        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
        
        # Pairwise Bilateral (edge-aware smoothness)
        d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, 
                               rgbim=image.astype(np.uint8), compat=compat_bilateral)
        
        # Inference
        Q = d.inference(n_iterations)
        result = np.argmax(Q, axis=0).reshape((H, W))
        
        return result.astype(np.int32)
        
    except ImportError:
        # pydensecrf not available, use simpler refinement
        return clean_segmentation(segmentation, image)


def load_poses(poses_path: str) -> dict:
    """Load camera poses from poses.txt"""
    poses = {}
    with open(poses_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                parts = line.split()
            frame_idx = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            R = quat_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            poses[frame_idx] = (R, t)
    return poses


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_camera_intrinsics(sparse_path: str):
    """Load camera intrinsics from COLMAP cameras.bin"""
    cameras_bin = Path(sparse_path) / "cameras.bin"
    with open(cameras_bin, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('i', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            if model_id == 0:  # SIMPLE_PINHOLE
                params = struct.unpack('3d', f.read(24))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                params = struct.unpack('4d', f.read(32))
                fx, fy, cx, cy = params
            elif model_id == 2:  # SIMPLE_RADIAL
                params = struct.unpack('4d', f.read(32))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Unsupported camera model: {model_id}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            return K, int(width), int(height)
    raise ValueError("No camera found")


def load_sparse_points(sparse_path: str) -> np.ndarray:
    """Load sparse 3D points from COLMAP points3D.bin"""
    points3d_bin = Path(sparse_path) / "points3D.bin"
    points = []
    with open(points3d_bin, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('3d', f.read(24))
            rgb = struct.unpack('3B', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            track_length = struct.unpack('Q', f.read(8))[0]
            f.read(track_length * 8)
            points.append(xyz)
    return np.array(points) if points else np.zeros((0, 3))


def estimate_depth_scale(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                         sparse_points: np.ndarray, width: int, height: int) -> float:
    """
    Estimate scale factor to align predicted depth with SfM sparse points.
    Projects sparse points into image, compares depths, returns median scale.
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


def depth_to_pointcloud(depth: np.ndarray, image: np.ndarray, K: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float = 1.0,
                        downsample: int = 4, max_depth: float = 10.0,
                        segmentation: np.ndarray = None):
    """Convert depth map to 3D point cloud in world coordinates
    
    Returns: points, colors, labels (labels is None if segmentation not provided)
    """
    H, W = depth.shape
    
    # Create pixel grid (downsampled)
    u = np.arange(0, W, downsample)
    v = np.arange(0, H, downsample)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()
    
    # Get scaled depth values
    z = depth[v, u] * scale
    
    # Filter by depth
    valid = (z > 0.1) & (z < max_depth) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]
    
    # Get colors
    colors = image[v, u]
    
    # Get labels if segmentation provided
    labels = None
    if segmentation is not None:
        labels = segmentation[v, u]
    
    # Unproject to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    points_cam = np.stack([x_cam, y_cam, z], axis=1)
    
    # Transform to world: P_world = R^T @ (P_cam - t)
    points_world = (R.T @ (points_cam.T - t.reshape(3, 1))).T
    
    return points_world, colors, labels


def save_ply(points: np.ndarray, colors: np.ndarray, path: str):
    """Save point cloud as PLY file"""
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    print(f"  Saved {len(points):,} points to {path}")


def warp_segmentation_with_flow(prev_seg: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp previous segmentation to current frame using optical flow.
    
    Args:
        prev_seg: Previous frame's segmentation (H, W), integer labels
        flow: Optical flow from prev to current (H, W, 2) in pixels
    
    Returns:
        Warped segmentation (H, W)
    """
    H, W = prev_seg.shape
    
    # Create sampling grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Apply flow: where each pixel in current frame comes from in previous frame
    # flow is (current -> previous displacement), so we add it
    src_x = (x + flow[:, :, 0]).astype(np.float32)
    src_y = (y + flow[:, :, 1]).astype(np.float32)
    
    # Use cv2.remap for nearest neighbor interpolation (preserves labels)
    warped = cv2.remap(
        prev_seg.astype(np.float32), 
        src_x, src_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    ).astype(np.int32)
    
    return warped


def compute_flow_confidence(flow: np.ndarray, prev_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
    """
    Compute confidence mask based on forward-backward consistency.
    Simple version: use gradient magnitude of flow as proxy.
    
    Returns: confidence map (H, W), values 0-1
    """
    # Compute flow gradient (high gradient = uncertain)
    flow_dx = np.gradient(flow[:, :, 0], axis=1)
    flow_dy = np.gradient(flow[:, :, 1], axis=0)
    flow_mag = np.sqrt(flow_dx**2 + flow_dy**2)
    
    # Also consider flow magnitude (very large flow = uncertain)
    total_flow = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    
    # Confidence: low gradient, reasonable magnitude
    confidence = np.exp(-flow_mag / 2.0) * np.exp(-total_flow / 100.0)
    
    return confidence.clip(0, 1)


def blend_segmentations(warped_seg: np.ndarray, new_seg: np.ndarray, 
                        confidence: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Blend warped (temporally consistent) and new (Mask2Former) segmentations.
    Use warped where confidence is high, new where confidence is low.
    """
    result = np.where(confidence > threshold, warped_seg, new_seg)
    return result.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 + Mask2Former + Reprojection")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--max_depth", type=float, default=80)
    parser.add_argument("--num_images", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal consistency via optical flow")
    parser.add_argument("--flow_confidence", type=float, default=0.4, help="Flow confidence threshold")
    parser.add_argument("--clean", action="store_true", help="Clean segmentation artifacts")
    parser.add_argument("--clean_min_area", type=int, default=500, help="Min area for regions (pixels)")
    parser.add_argument("--clean_morph_size", type=int, default=5, help="Morphological kernel size")
    parser.add_argument("--crf", action="store_true", help="Apply CRF refinement (requires pydensecrf)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    registered_dir = output_dir / "registered_images"
    poses_path = output_dir / "poses.txt"
    sparse_path = output_dir / "sparse" / "0"
    
    if not registered_dir.exists() or not poses_path.exists():
        print("Error: Run main.py first to generate SfM output")
        return
    
    print("=" * 60)
    print("Depth Anything V2 + Mask2Former + Reprojection")
    if args.temporal:
        print("  [Temporal Consistency: ENABLED via RAFT optical flow]")
    if args.clean:
        print(f"  [Artifact Cleaning: ENABLED (min_area={args.clean_min_area}, morph={args.clean_morph_size})]")
    if args.crf:
        print("  [CRF Refinement: ENABLED]")
    print("=" * 60)
    
    # Load camera intrinsics
    print("\n[1/6] Loading camera intrinsics...")
    K, cam_width, cam_height = load_camera_intrinsics(str(sparse_path))
    print(f"  Camera: {cam_width}x{cam_height}, fx={K[0,0]:.1f}")
    
    # Load poses
    print("\n[2/6] Loading camera poses...")
    poses = load_poses(str(poses_path))
    print(f"  Loaded {len(poses)} poses")
    
    # Load sparse points for scale estimation
    print("\n[3/6] Loading sparse points...")
    sparse_points = load_sparse_points(str(sparse_path))
    print(f"  Loaded {len(sparse_points)} sparse points")
    
    # Get image files and subsample evenly
    image_files = sorted(list(registered_dir.glob("*.jpg")) + list(registered_dir.glob("*.png")))
    step = max(1, len(image_files) // args.num_images)
    selected_files = image_files[::step][:args.num_images]
    print(f"\n  Selected {len(selected_files)} images (every {step}th)")
    
    # Load Depth Anything V2 model
    print("\n[4/6] Loading Depth Anything V2...")
    depth_model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(args.device).eval()
    print(f"  Depth model on {args.device}")
    
    # Load Mask2Former model
    print("\n[5/7] Loading Mask2Former...")
    seg_model_name = "facebook/mask2former-swin-base-ade-semantic"
    seg_processor = AutoImageProcessor.from_pretrained(seg_model_name)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_model_name).to(args.device).eval()
    print(f"  Segmentation model on {args.device}")
    
    # Load RAFT optical flow model for temporal consistency
    flow_model = None
    if args.temporal:
        print("\n[6/7] Loading RAFT optical flow...")
        weights = Raft_Small_Weights.DEFAULT
        flow_model = raft_small(weights=weights).to(args.device).eval()
        flow_transforms = weights.transforms()
        print(f"  RAFT model on {args.device}")
    
    # Create output dirs
    rgb_ply_dir = output_dir / "ply_rgb"
    label_ply_dir = output_dir / "ply_label"
    depth_dir = output_dir / "depth_maps"
    seg_dir = output_dir / "segmentation"
    rgb_ply_dir.mkdir(exist_ok=True)
    label_ply_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    
    # Process images
    step_num = "7/7" if args.temporal else "6/6"
    print(f"\n[{step_num}] Processing {len(selected_files)} images...")
    all_points, all_colors, all_labels = [], [], []
    
    # Temporal consistency state
    prev_image_np = None
    prev_segmentation = None
    
    # Label statistics tracking
    all_frame_stats = {}
    global_label_counts = {}
    
    for i, img_path in enumerate(tqdm(selected_files, desc="Processing")):
        stem = img_path.stem
        frame_idx = int(stem.replace("frame_", "")) if stem.startswith("frame_") else i
        
        if frame_idx not in poses:
            continue
        
        R, t = poses[frame_idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        H, W = image_np.shape[:2]
        
        # Run depth model
        depth_inputs = depth_processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            depth = depth_model(**depth_inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        
        # Run segmentation model (Mask2Former)
        seg_inputs = seg_processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            seg_outputs = seg_model(**seg_inputs)
        raw_segmentation = seg_processor.post_process_semantic_segmentation(
            seg_outputs, target_sizes=[(H, W)]
        )[0].cpu().numpy()
        
        # Apply temporal consistency if enabled
        if args.temporal and prev_image_np is not None and prev_segmentation is not None:
            # Compute optical flow from previous to current frame
            prev_tensor = torch.from_numpy(prev_image_np).permute(2, 0, 1).float().unsqueeze(0)
            curr_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0)
            
            # Resize for RAFT (needs to be divisible by 8)
            h_raft = (H // 8) * 8
            w_raft = (W // 8) * 8
            prev_resized = F.interpolate(prev_tensor, size=(h_raft, w_raft), mode='bilinear', align_corners=False)
            curr_resized = F.interpolate(curr_tensor, size=(h_raft, w_raft), mode='bilinear', align_corners=False)
            
            # Apply RAFT transforms
            prev_raft, curr_raft = flow_transforms(prev_resized.to(args.device), curr_resized.to(args.device))
            
            with torch.no_grad():
                flow_list = flow_model(prev_raft, curr_raft)
                flow = flow_list[-1]  # Take final flow prediction
            
            # Resize flow back to original size
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            flow[:, 0] *= W / w_raft
            flow[:, 1] *= H / h_raft
            flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Warp previous segmentation
            warped_seg = warp_segmentation_with_flow(prev_segmentation, flow)
            
            # Compute confidence
            confidence = compute_flow_confidence(flow, prev_image_np, image_np)
            
            # Blend warped and new segmentation
            segmentation = blend_segmentations(warped_seg, raw_segmentation, confidence, args.flow_confidence)
        else:
            segmentation = raw_segmentation
        
        # Apply segmentation cleaning if enabled
        if args.clean:
            segmentation = clean_segmentation(
                segmentation, image_np,
                min_area=args.clean_min_area,
                morph_size=args.clean_morph_size,
                use_bilateral=True
            )
        
        # Apply CRF refinement if enabled
        if args.crf:
            segmentation = apply_crf_refinement(segmentation, image_np)
        
        # Update temporal state
        prev_image_np = image_np.copy()
        prev_segmentation = segmentation.copy()
        
        # Estimate scale from sparse points
        scale = estimate_depth_scale(depth, K, R, t, sparse_points, cam_width, cam_height)
        
        # Save depth visualization
        depth_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(depth_dir / f"{stem}_depth.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
        
        # Get label statistics for this frame
        frame_stats = get_label_statistics(segmentation)
        all_frame_stats[stem] = frame_stats
        
        # Accumulate global label counts
        for name, info in frame_stats.items():
            if name not in global_label_counts:
                global_label_counts[name] = {"id": info["id"], "count": 0}
            global_label_counts[name]["count"] += info["count"]
        
        # Save segmentation visualization with legend
        seg_colored = ADE20K_COLORS[segmentation % len(ADE20K_COLORS)]
        
        # Create visualization with legend
        seg_vis = seg_colored.copy()
        # Sort by percentage (top labels)
        sorted_labels = sorted(frame_stats.items(), key=lambda x: x[1]["percentage"], reverse=True)[:10]
        
        # Add legend on the image
        legend_h = 25 * min(len(sorted_labels), 10) + 10
        legend_w = 200
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        for j, (name, info) in enumerate(sorted_labels[:10]):
            color = ADE20K_COLORS[info["id"] % len(ADE20K_COLORS)]
            y = 5 + j * 25
            cv2.rectangle(legend, (5, y), (25, y + 20), color.tolist(), -1)
            cv2.putText(legend, f"{name}: {info['percentage']:.1f}%", (30, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Composite legend onto segmentation
        if H > legend_h and W > legend_w:
            seg_vis[:legend_h, :legend_w] = cv2.addWeighted(
                seg_vis[:legend_h, :legend_w], 0.3, legend, 0.7, 0)
        
        cv2.imwrite(str(seg_dir / f"{stem}_seg.png"), cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
        
        # Reproject to 3D with scale
        points, colors, labels = depth_to_pointcloud(
            depth, image_np, K, R, t, scale=scale,
            downsample=args.downsample, max_depth=args.max_depth,
            segmentation=segmentation
        )
        
        # Get label colors from palette
        label_colors = ADE20K_COLORS[labels % len(ADE20K_COLORS)]
        
        # Save individual RGB PLY
        save_ply(points, colors, str(rgb_ply_dir / f"{i+1:02d}_{stem}_rgb.ply"))
        
        # Save individual Label PLY
        save_ply(points, label_colors, str(label_ply_dir / f"{i+1:02d}_{stem}_label.ply"))
        
        all_points.append(points)
        all_colors.append(colors)
        all_labels.append(label_colors)
    
    # Combine and save
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        combined_labels = np.vstack(all_labels)
        
        print(f"\nTotal: {len(combined_points):,} points")
        
        # Save combined RGB point cloud
        save_ply(combined_points, combined_colors, str(output_dir / "dense_rgb.ply"))
        
        # Save combined Label point cloud
        save_ply(combined_points, combined_labels, str(output_dir / "dense_label.ply"))
        
        # Calculate global percentages and save label statistics
        total_pixels = sum(info["count"] for info in global_label_counts.values())
        sorted_global = sorted(global_label_counts.items(), 
                              key=lambda x: x[1]["count"], reverse=True)
        
        # Save label statistics to file
        stats_path = output_dir / "label_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SEMANTIC LABEL STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total pixels analyzed: {total_pixels:,}\n")
            f.write(f"Number of unique classes: {len(global_label_counts)}\n\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Rank':<6}{'Label Name':<25}{'ID':<6}{'Count':<15}{'%':<8}\n")
            f.write("-" * 60 + "\n")
            for rank, (name, info) in enumerate(sorted_global, 1):
                pct = info["count"] / total_pixels * 100
                f.write(f"{rank:<6}{name:<25}{info['id']:<6}{info['count']:<15,}{pct:<8.2f}\n")
            f.write("-" * 60 + "\n")
        
        print(f"  Saved label statistics to {stats_path}")
        
        # Print top labels to console
        print("\n  Top 10 detected labels:")
        for rank, (name, info) in enumerate(sorted_global[:10], 1):
            pct = info["count"] / total_pixels * 100
            print(f"    {rank:2d}. {name:<20} ({pct:.1f}%)")
        
        print("\n" + "=" * 60)
        print("Done!")
        print(f"  RGB point clouds:")
        print(f"    - {output_dir}/dense_rgb.ply")
        print(f"    - {rgb_ply_dir}/ ({len(all_points)} files)")
        print(f"  Label point clouds:")
        print(f"    - {output_dir}/dense_label.ply")
        print(f"    - {label_ply_dir}/ ({len(all_points)} files)")
        print(f"  Visualizations:")
        print(f"    - {depth_dir}/")
        print(f"    - {seg_dir}/")
        print(f"  Statistics:")
        print(f"    - {stats_path}")


if __name__ == "__main__":
    main()
