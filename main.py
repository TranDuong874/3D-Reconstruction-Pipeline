import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from utils.video_loader import VideoLoader
from utils.feature_extractor import FeatureExtractor
from utils.feature_matcher import FeatureMatcher
from utils.depth_estimator import DepthEstimator, estimate_depth_scale
from utils.segmentation import (
    SemanticSegmenter, clean_segmentation, get_label_statistics,
    colorize_segmentation, ADE20K_COLORS
)
from utils.reprojection import (
    depth_to_pointcloud, save_ply, load_sparse_points, load_camera_intrinsics, load_poses
)
from data_classes.data_classes import Frame, Features, Match
from tqdm import tqdm
import time

import pycolmap


def timeit(name):
    """Context manager for timing code blocks"""
    class Timer:
        def __init__(self, name):
            self.name = name
            self.start = None
            self.elapsed = None
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"{self.name}: {self.elapsed:.2f}s")
    return Timer(name)


@dataclass
class Camera:
    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3,) translation vector
    K: np.ndarray  # (3, 3) intrinsic matrix
    image_id: int


@dataclass
class SfMResult:
    cameras: dict  # frame_idx -> Camera
    points3d: np.ndarray  # (N, 3) 3D points
    colors: np.ndarray  # (N, 3) RGB colors
    point_errors: np.ndarray  # (N,) reprojection errors
    num_registered: int
    num_total: int


class SfMReconstructor:
    """Stage 4: Incremental SfM via pycolmap"""
    
    def __init__(self, image_size: tuple, focal_length: float = None):
        """
        Args:
            image_size: (width, height)
            focal_length: if None, estimate from image size
        """
        self.width, self.height = image_size
        self.focal_length = focal_length or max(image_size) * 1.2
        
    def reconstruct(self, frames: list, features: dict, matches: list, output_dir: str = "output") -> SfMResult:
        """
        Run incremental SfM
        
        Args:
            frames: list of Frame objects
            features: dict[frame_idx, Features]
            matches: list of Match objects
            output_dir: directory to save COLMAP output (cameras.bin, images.bin, points3D.bin)
        """
        # Create output directory structure (permanent)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = output_dir / "database.db"
        image_path = output_dir / "images"
        sparse_path = output_dir / "sparse" / "0"  # COLMAP convention: sparse/0/
        
        image_path.mkdir(parents=True, exist_ok=True)
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        # Remove old database if exists
        if db_path.exists():
            db_path.unlink()
        
        try:
            # Save images to output folder
            print("  Saving images to temp folder...")
            frame_to_name = {}
            for frame in tqdm(frames, desc="  Saving images"):
                img_name = f"frame_{frame.idx:06d}.jpg"
                frame_to_name[frame.idx] = img_name
                img_bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path / img_name), img_bgr)
            
            # Create database
            print("  Creating database...")
            db = pycolmap.Database.open(str(db_path))
            
            # Add camera (shared intrinsics)
            camera = pycolmap.Camera(
                model='PINHOLE',
                width=self.width,
                height=self.height,
                params=[self.focal_length, self.focal_length, self.width/2, self.height/2]
            )
            camera_id = db.write_camera(camera)
            
            # Add images and keypoints
            print("  Adding images and keypoints...")
            image_ids = {}
            for frame in tqdm(frames, desc="  Adding images"):
                img_name = frame_to_name[frame.idx]
                image = pycolmap.Image(
                    name=img_name,
                    camera_id=camera_id
                )
                img_id = db.write_image(image)
                image_ids[frame.idx] = img_id
                
                # Add keypoints
                kpts = features[frame.idx].keypoints.astype(np.float32)
                db.write_keypoints(img_id, kpts)
            
            # Add matches
            print("  Adding matches...")
            pairs = []
            for match in tqdm(matches, desc="  Adding matches"):
                img_id0 = image_ids[match.idx0]
                img_id1 = image_ids[match.idx1]
                
                # Filter invalid matches
                valid = (match.matches[:, 0] >= 0) & (match.matches[:, 1] >= 0)
                valid_matches = match.matches[valid].astype(np.uint32)
                
                if len(valid_matches) >= 15:
                    db.write_matches(img_id0, img_id1, valid_matches)
                    pairs.append((frame_to_name[match.idx0], frame_to_name[match.idx1]))
            
            # Write pairs file for verification
            pairs_path = output_dir / "pairs.txt"
            with open(pairs_path, 'w') as f:
                for name0, name1 in pairs:
                    f.write(f"{name0} {name1}\n")
            
            db.close()
            
            # Verify matches with geometric verification
            print("  Verifying matches (geometric verification)...")
            pycolmap.verify_matches(
                database_path=str(db_path),
                pairs_path=str(pairs_path),
                options=pycolmap.TwoViewGeometryOptions()
            )
            
            # Run reconstruction
            print("  Running incremental SfM...")
            opts = pycolmap.IncrementalPipelineOptions()
            opts.min_num_matches = 15
            opts.ba_global_max_num_iterations = 50
            
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(db_path),
                image_path=str(image_path),
                output_path=str(sparse_path.parent),  # pycolmap adds /0 automatically
                options=opts
            )
            
            if not reconstructions:
                raise RuntimeError("SfM failed - no reconstruction produced")
            
            # Take best reconstruction (most registered images)
            best_idx = max(reconstructions.keys(), key=lambda k: reconstructions[k].num_reg_images())
            recon = reconstructions[best_idx]
            
            print(f"  Reconstruction complete: {recon.num_reg_images()}/{len(frames)} images registered")
            print(f"  3D points: {len(recon.points3D)}")
            
            # Save COLMAP binary files
            print(f"  Saving COLMAP files to {sparse_path}...")
            recon.write_binary(str(sparse_path))
            
            return self._extract_result(recon, image_ids, frames, features, sparse_path)
            
        except Exception as e:
            raise e
    
    def _extract_result(self, recon, image_ids, frames, features, output_path) -> SfMResult:
        """Extract results from pycolmap reconstruction"""
        
        # Invert image_ids mapping
        id_to_frame = {v: k for k, v in image_ids.items()}
        
        # Get registered image IDs
        reg_ids = set(recon.reg_image_ids())
        
        # Extract cameras from registered images
        cameras = {}
        for img_id in reg_ids:
            image = recon.images[img_id]
            frame_idx = id_to_frame.get(img_id)
            if frame_idx is not None:
                cam = recon.cameras[image.camera_id]
                # Get pose via image's frame
                try:
                    pose = image.cam_from_world()  # Returns Rigid3d
                    cameras[frame_idx] = Camera(
                        R=pose.rotation.matrix(),
                        t=np.array(pose.translation),
                        K=cam.calibration_matrix(),
                        image_id=img_id
                    )
                except Exception as e:
                    print(f"  Warning: Could not get pose for image {img_id}: {e}")
        
        # Extract 3D points with colors
        points3d = []
        colors = []
        errors = []
        
        for pt3d in recon.points3D.values():
            points3d.append(pt3d.xyz)
            colors.append(pt3d.color)
            errors.append(pt3d.error)
        
        # Also export PLY directly via pycolmap (as backup)
        try:
            ply_path = output_path / "points.ply"
            recon.export_PLY(str(ply_path))
            print(f"  Exported PLY to {ply_path}")
        except Exception as e:
            print(f"  Warning: Could not export PLY: {e}")
        
        return SfMResult(
            cameras=cameras,
            points3d=np.array(points3d) if points3d else np.zeros((0, 3)),
            colors=np.array(colors) if colors else np.zeros((0, 3), dtype=np.uint8),
            point_errors=np.array(errors) if errors else np.zeros(0),
            num_registered=recon.num_reg_images(),
            num_total=len(frames)
        )


def save_point_cloud(result: SfMResult, path: str):
    """Save point cloud as PLY"""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(result.points3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, col in zip(result.points3d, result.colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(col[0])} {int(col[1])} {int(col[2])}\n")
    
    print(f"Saved {len(result.points3d)} points to {path}")


def save_registered_images(result: SfMResult, frames: list, output_dir: str):
    """
    Save only images that were successfully registered (have poses).
    Also exports camera poses to a text file.
    """
    output_dir = Path(output_dir)
    registered_dir = output_dir / "registered_images"
    registered_dir.mkdir(parents=True, exist_ok=True)
    
    # Get registered frame indices
    registered_indices = set(result.cameras.keys())
    
    # Filter frames
    registered_frames = [f for f in frames if f.idx in registered_indices]
    unregistered_frames = [f for f in frames if f.idx not in registered_indices]
    
    print(f"  Registered: {len(registered_frames)} images")
    print(f"  Unregistered (filtered out): {len(unregistered_frames)} images")
    
    # Save registered images
    for frame in registered_frames:
        img_name = f"frame_{frame.idx:06d}.jpg"
        img_bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(registered_dir / img_name), img_bgr)
    
    # Save poses to text file (for easy inspection)
    poses_path = output_dir / "poses.txt"
    with open(poses_path, 'w') as f:
        f.write("# frame_idx, qw, qx, qy, qz, tx, ty, tz\n")
        f.write("# Rotation as quaternion (w, x, y, z), Translation (x, y, z)\n")
        for frame_idx in sorted(result.cameras.keys()):
            cam = result.cameras[frame_idx]
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(cam.R).as_quat()  # x, y, z, w
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = cam.t
            f.write(f"{frame_idx}, {qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}, {tx:.6f}, {ty:.6f}, {tz:.6f}\n")
    
    print(f"  Saved poses to {poses_path}")
    print(f"  Saved {len(registered_frames)} registered images to {registered_dir}")
    
    return registered_frames


def run_dense_reconstruction(output_dir: str, num_images: int = 30, 
                             downsample: int = 4, max_depth: float = 10.0,
                             enable_temporal: bool = False,
                             enable_cleaning: bool = True,
                             device: str = "cuda"):
    """
    Run dense reconstruction using Depth Anything V2 + Mask2Former.
    
    Args:
        output_dir: Directory containing SfM output
        num_images: Number of images to process
        downsample: Point cloud downsampling factor
        max_depth: Maximum depth in meters
        enable_temporal: Enable temporal consistency for segmentation
        enable_cleaning: Clean segmentation artifacts
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    registered_dir = output_dir / "registered_images"
    poses_path = output_dir / "poses.txt"
    sparse_path = output_dir / "sparse" / "0"
    
    if not registered_dir.exists() or not poses_path.exists():
        raise FileNotFoundError("Run SfM first to generate registered images and poses")
    
    print("=" * 60)
    print("DENSE RECONSTRUCTION")
    print(f"  Temporal consistency: {'ENABLED' if enable_temporal else 'DISABLED'}")
    print(f"  Artifact cleaning: {'ENABLED' if enable_cleaning else 'DISABLED'}")
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
    step = max(1, len(image_files) // num_images)
    selected_files = image_files[::step][:num_images]
    print(f"\n  Selected {len(selected_files)} images (every {step}th)")
    
    # Load models
    print("\n[4/6] Loading depth model...")
    depth_estimator = DepthEstimator(device=device)
    
    print("\n[5/6] Loading segmentation model...")
    segmenter = SemanticSegmenter(device=device, enable_temporal=enable_temporal)
    
    # Create output directories
    rgb_ply_dir = output_dir / "ply_rgb"
    label_ply_dir = output_dir / "ply_label"
    depth_dir = output_dir / "depth_maps"
    seg_dir = output_dir / "segmentation"
    
    for d in [rgb_ply_dir, label_ply_dir, depth_dir, seg_dir]:
        d.mkdir(exist_ok=True)
    
    # Process images
    print(f"\n[6/6] Processing {len(selected_files)} images...")
    all_points, all_colors, all_labels = [], [], []
    global_label_counts = {}
    
    for i, img_path in enumerate(tqdm(selected_files, desc="Dense reconstruction")):
        stem = img_path.stem
        frame_idx = int(stem.replace("frame_", "")) if stem.startswith("frame_") else i
        
        if frame_idx not in poses:
            continue
        
        R, t = poses[frame_idx]
        
        # Load image
        image_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = image_np.shape[:2]
        
        # Run depth estimation
        depth = depth_estimator.estimate(image_np)
        
        # Run segmentation
        segmentation = segmenter.segment(image_np)
        
        # Clean segmentation
        if enable_cleaning:
            segmentation = clean_segmentation(segmentation, image_np)
        
        # Scale intrinsics if image resolution differs from camera model
        depth_H, depth_W = depth.shape[:2]
        if depth_W != cam_width or depth_H != cam_height:
            scale_x = depth_W / cam_width
            scale_y = depth_H / cam_height
            K_img = K.copy()
            K_img[0, 0] *= scale_x  # fx
            K_img[1, 1] *= scale_y  # fy
            K_img[0, 2] *= scale_x  # cx
            K_img[1, 2] *= scale_y  # cy
        else:
            K_img = K
        
        # Estimate scale from sparse points
        scale = estimate_depth_scale(depth, K_img, R, t, sparse_points, depth_W, depth_H)

        # Save depth visualization
        depth_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(depth_dir / f"{stem}_depth.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
        
        # Save segmentation visualization
        seg_colored = colorize_segmentation(segmentation)
        cv2.imwrite(str(seg_dir / f"{stem}_seg.png"), cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR))
        
        # Accumulate label counts
        frame_stats = get_label_statistics(segmentation)
        for name, info in frame_stats.items():
            if name not in global_label_counts:
                global_label_counts[name] = {"id": info["id"], "count": 0}
            global_label_counts[name]["count"] += info["count"]
        
        # Reproject to 3D (use scaled intrinsics)
        points, colors, labels = depth_to_pointcloud(
            depth, image_np, K_img, R, t, scale=scale,
            downsample=downsample, max_depth=max_depth,
            segmentation=segmentation
        )
        
        # Get label colors
        label_colors = ADE20K_COLORS[labels % len(ADE20K_COLORS)]
        
        # Save individual PLY files
        save_ply(points, colors, str(rgb_ply_dir / f"{i+1:02d}_{stem}_rgb.ply"))
        save_ply(points, label_colors, str(label_ply_dir / f"{i+1:02d}_{stem}_label.ply"))
        
        all_points.append(points)
        all_colors.append(colors)
        all_labels.append(label_colors)
    
    # Combine and save
    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        combined_labels = np.vstack(all_labels)
        
        print(f"\n  Total: {len(combined_points):,} points")
        
        # Save combined point clouds
        save_ply(combined_points, combined_colors, str(output_dir / "dense_rgb.ply"))
        save_ply(combined_points, combined_labels, str(output_dir / "dense_label.ply"))
        
        # Save label statistics
        total_pixels = sum(info["count"] for info in global_label_counts.values())
        sorted_global = sorted(global_label_counts.items(), key=lambda x: x[1]["count"], reverse=True)
        
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
        
        # Print top labels
        print("\n  Top 10 detected labels:")
        for rank, (name, info) in enumerate(sorted_global[:10], 1):
            pct = info["count"] / total_pixels * 100
            print(f"    {rank:2d}. {name:<20} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Dense reconstruction complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/dense_rgb.ply      (combined RGB point cloud)")
    print(f"  - {output_dir}/dense_label.ply    (combined semantic point cloud)")
    print(f"  - {rgb_ply_dir}/                  (per-frame RGB)")
    print(f"  - {label_ply_dir}/                (per-frame semantic)")
    print(f"  - {depth_dir}/                    (depth visualizations)")
    print(f"  - {seg_dir}/                      (segmentation visualizations)")


def run_colmap_pipeline(input_dir: str, output_dir: str) -> SfMResult:
    """
    Fallback: Run full COLMAP pipeline using pycolmap's native SIFT extraction and matching.
    Used when DISK-based SfM fails.
    """
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = output_dir / "database.db"
    image_path = output_dir / "images"
    sparse_path = output_dir / "sparse" / "0"
    
    image_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    # Remove old database
    if db_path.exists():
        db_path.unlink()
    
    # Copy images to output folder
    print("  Copying images...")
    input_path = Path(input_dir)
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy(img_file, image_path)
    
    # Feature extraction (pycolmap's SIFT)
    print("  Extracting features (COLMAP SIFT)...")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_path),
    )
    
    # Exhaustive matching
    print("  Matching features...")
    pycolmap.match_exhaustive(database_path=str(db_path))
    
    # Incremental SfM
    print("  Running incremental SfM...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_path),
        output_path=str(sparse_path.parent),
    )
    
    if not reconstructions:
        raise RuntimeError("COLMAP SfM failed - no reconstruction produced")
    
    # Get best reconstruction
    best_idx = max(reconstructions.keys(), key=lambda k: reconstructions[k].num_reg_images())
    recon = reconstructions[best_idx]
    
    print(f"  Reconstruction complete: {recon.num_reg_images()} images registered")
    print(f"  3D points: {len(recon.points3D)}")
    
    # Save COLMAP binary files
    recon.write_binary(str(sparse_path))
    
    # Export PLY
    try:
        recon.export_PLY(str(sparse_path / "points.ply"))
    except:
        pass
    
    # Extract result
    cameras = {}
    for img_id in recon.reg_image_ids():
        image = recon.images[img_id]
        cam = recon.cameras[image.camera_id]
        pose = image.cam_from_world()
        
        # Parse frame index from filename
        name = image.name
        import re
        nums = re.findall(r'\d+', name)
        frame_idx = int(nums[0]) if nums else img_id
        
        cameras[frame_idx] = Camera(
            R=pose.rotation.matrix(),
            t=np.array(pose.translation),
            K=cam.calibration_matrix(),
            image_id=img_id
        )
    
    points3d = np.array([pt.xyz for pt in recon.points3D.values()]) if recon.points3D else np.zeros((0, 3))
    colors = np.array([pt.color for pt in recon.points3D.values()]) if recon.points3D else np.zeros((0, 3), dtype=np.uint8)
    errors = np.array([pt.error for pt in recon.points3D.values()]) if recon.points3D else np.zeros(0)
    
    return SfMResult(
        cameras=cameras,
        points3d=points3d,
        colors=colors,
        point_errors=errors,
        num_registered=recon.num_reg_images(),
        num_total=len(list(image_path.glob("*")))
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--skip_sfm", action="store_true", help="Skip SfM, run only dense reconstruction")
    parser.add_argument("--skip_dense", action="store_true", help="Skip dense reconstruction")
    parser.add_argument("--num_images", type=int, default=30, help="Number of images for dense reconstruction")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal consistency")
    parser.add_argument("--no_clean", action="store_true", help="Disable segmentation cleaning")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--input", type=str, default="data/rgbd_dataset_freiburg1_room/rgb", help="Input image directory")
    parser.add_argument("--subsample", type=int, default=10, help="Frame subsampling rate")
    args = parser.parse_args()
    
    timings = {}
    total_start = time.perf_counter()
    output_dir = args.output
    
    if not args.skip_sfm:
        print("="*60)
        print("STAGE 1: Loading frames...")
        print("="*60)
        with timeit("Frame loading") as t:
            loader = VideoLoader(subsample=args.subsample, blur_threshold=150.0, max_size=1024)
            frames = loader.load_images(args.input)
        timings['1_loading'] = t.elapsed

        print("\n" + "="*60)
        print("STAGE 2: Extracting features (GPU-accelerated)...")
        print("="*60)
        with timeit("Feature extraction (DISK)") as t:
            extractor = FeatureExtractor(method='disk', max_features=8192, device='cuda', batch_size=4)
            features = extractor.extract_all(frames)
        timings['2_extraction'] = t.elapsed
        print(f"Extracted features from {len(features)} frames")

        print("\n" + "="*60)
        print("STAGE 3: Matching features...")
        print("="*60)
        with timeit("Feature matching (GPU)") as t:
            matcher = FeatureMatcher(device='cuda', match_mode='snn', th=0.9)
            matches = matcher.match_exhaustive(features)
        timings['3_matching'] = t.elapsed
        print(f"Found {len(matches)} matching pairs")
        
        print("\n" + "="*60)
        print("STAGE 4: Structure from Motion...")
        print("="*60)
        h, w = frames[0].image.shape[:2]
        sfm = SfMReconstructor(image_size=(w, h))
        
        sfm_success = False
        try:
            with timeit("SfM reconstruction") as t:
                result = sfm.reconstruct(frames, features, matches, output_dir=output_dir)
            timings['4_sfm'] = t.elapsed
            
            print(f"\nSfM Results:")
            print(f"  Registered images: {result.num_registered}/{result.num_total}")
            print(f"  3D points: {len(result.points3d)}")
            print(f"  Mean reprojection error: {result.point_errors.mean():.3f} px" if len(result.point_errors) > 0 else "  No points")
            sfm_success = True
            
        except Exception as e:
            print(f"\nDISK-based SfM failed: {e}")
            print("\n" + "="*60)
            print("Falling back to vanilla COLMAP pipeline...")
            print("="*60)
            
            try:
                with timeit("COLMAP fallback") as t:
                    result = run_colmap_pipeline(args.input, output_dir)
                timings['4_colmap_fallback'] = t.elapsed
                
                print(f"\nCOLMAP Results:")
                print(f"  Registered images: {result.num_registered}/{result.num_total}")
                print(f"  3D points: {len(result.points3d)}")
                print(f"  Mean reprojection error: {result.point_errors.mean():.3f} px" if len(result.point_errors) > 0 else "  No points")
                
                # Reload frames from COLMAP output for consistency
                loader = VideoLoader(subsample=1, blur_threshold=0, max_size=None)
                frames = loader.load_images(f"{output_dir}/images")
                sfm_success = True
                
            except Exception as e2:
                print(f"\nCOLMAP fallback also failed: {e2}")
                import traceback
                traceback.print_exc()
        
        if sfm_success:
            print("\n" + "="*60)
            print("Saving SfM outputs...")
            print("="*60)
            with timeit("Save PLY") as t:
                save_point_cloud(result, f"{output_dir}/point_cloud.ply")
            timings['5_save_ply'] = t.elapsed
            
            print("\n" + "="*60)
            print("Filtering registered images...")
            print("="*60)
            with timeit("Filter registered") as t:
                registered_frames = save_registered_images(result, frames, output_dir)
            timings['6_filter'] = t.elapsed
        else:
            args.skip_dense = True  # Can't run dense without SfM
    
    # Run dense reconstruction
    if not args.skip_dense:
        print("\n" + "="*60)
        print("STAGE 5: Dense Reconstruction...")
        print("="*60)
        with timeit("Dense reconstruction") as t:
            run_dense_reconstruction(
                output_dir=output_dir,
                num_images=args.num_images,
                enable_temporal=args.temporal,
                enable_cleaning=not args.no_clean,
                device=args.device
            )
        timings['7_dense'] = t.elapsed
    
    total_elapsed = time.perf_counter() - total_start
    
    print("\n" + "="*60)
    print("TIME PROFILE")
    print("="*60)
    for name, elapsed in timings.items():
        pct = (elapsed / total_elapsed) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {name:20s}: {elapsed:6.2f}s ({pct:5.1f}%) {bar}")
    print(f"  {'TOTAL':20s}: {total_elapsed:6.2f}s")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print(f"\nAll outputs saved to '{output_dir}/'")
    print(f"\nSparse reconstruction:")
    print(f"  - sparse/0/                 (COLMAP format)")
    print(f"  - point_cloud.ply           (sparse point cloud)")
    print(f"  - poses.txt                 (camera poses)")
    if not args.skip_dense:
        print(f"\nDense reconstruction:")
        print(f"  - dense_rgb.ply             (dense RGB point cloud)")
        print(f"  - dense_label.ply           (dense semantic point cloud)")
        print(f"  - depth_maps/               (depth visualizations)")
        print(f"  - segmentation/             (segmentation visualizations)")