"""
Reprojection Module

Converts depth maps to 3D point clouds and handles PLY file I/O.
"""

import numpy as np
import struct
from pathlib import Path


def depth_to_pointcloud(depth: np.ndarray, image: np.ndarray, K: np.ndarray,
                        R: np.ndarray, t: np.ndarray, scale: float = 1.0,
                        downsample: int = 4, max_depth: float = 10.0,
                        segmentation: np.ndarray = None):
    """
    Convert depth map to 3D point cloud in world coordinates.
    
    Args:
        depth: Depth map (H, W)
        image: RGB image (H, W, 3)
        K: Camera intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3) - camera to world
        t: Translation vector (3,) - camera to world
        scale: Depth scale factor
        downsample: Downsampling factor for points
        max_depth: Maximum depth to include
        segmentation: Optional segmentation map (H, W)
    
    Returns:
        points: 3D points in world coordinates (N, 3)
        colors: RGB colors (N, 3)
        labels: Segmentation labels (N,) or None
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
    """
    Save point cloud as PLY file.
    
    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3), values 0-255
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    
    print(f"  Saved {len(points):,} points to {path}")


def load_sparse_points(sparse_path: str) -> np.ndarray:
    """
    Load sparse 3D points from COLMAP points3D.bin.
    
    Args:
        sparse_path: Path to COLMAP sparse directory (containing points3D.bin)
    
    Returns:
        3D points as numpy array (N, 3)
    """
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
            f.read(track_length * 8)  # Skip track data
            points.append(xyz)
    
    return np.array(points) if points else np.zeros((0, 3))


def load_camera_intrinsics(sparse_path: str):
    """
    Load camera intrinsics from COLMAP cameras.bin.
    
    Args:
        sparse_path: Path to COLMAP sparse directory
    
    Returns:
        K: Intrinsic matrix (3, 3)
        width: Image width
        height: Image height
    """
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


def load_poses(poses_path: str) -> dict:
    """
    Load camera poses from poses.txt.
    
    Args:
        poses_path: Path to poses.txt file
    
    Returns:
        Dictionary mapping frame_idx to (R, t) tuple
    """
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
