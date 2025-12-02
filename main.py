import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from utils.video_loader import VideoLoader
from data_classes.data_classes import Frame, Features, Match
from utils.feature_extractor import FeatureExtractor
from utils.feature_matcher import FeatureMatcher
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



if __name__ == '__main__':
    timings = {}
    total_start = time.perf_counter()
    
    print("="*60)
    print("STAGE 1: Loading frames...")
    print("="*60)
    with timeit("Frame loading") as t:
        loader = VideoLoader(subsample=10, blur_threshold=150.0, max_size=1024)
        frames = loader.load_images('data/rgbd_dataset_freiburg1_room/rgb')
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
    output_dir = "output"
    
    try:
        with timeit("SfM reconstruction") as t:
            result = sfm.reconstruct(frames, features, matches, output_dir=output_dir)
        timings['4_sfm'] = t.elapsed
        
        print(f"\nSfM Results:")
        print(f"  Registered images: {result.num_registered}/{result.num_total}")
        print(f"  3D points: {len(result.points3d)}")
        print(f"  Mean reprojection error: {result.point_errors.mean():.3f} px" if len(result.point_errors) > 0 else "  No points")
        
        print("\n" + "="*60)
        print("Saving outputs...")
        print("="*60)
        with timeit("Save PLY") as t:
            save_point_cloud(result, f"{output_dir}/point_cloud.ply")
        timings['5_save_ply'] = t.elapsed
        
        print("\n" + "="*60)
        print("Filtering registered images...")
        print("="*60)
        with timeit("Filter registered") as t:
            save_registered_images(result, frames, output_dir)
        timings['6_filter'] = t.elapsed
        
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
        print(f"\nOutput files saved to '{output_dir}/':")
        print(f"  - sparse/0/cameras.bin      (camera intrinsics)")
        print(f"  - sparse/0/images.bin       (camera poses - COLMAP format)")
        print(f"  - sparse/0/points3D.bin     (3D points)")
        print(f"  - images/                   (all input images)")
        print(f"  - registered_images/        (only images with poses)")
        print(f"  - poses.txt                 (camera poses as quaternion+translation)")
        print(f"  - database.db               (COLMAP database)")
        print(f"  - point_cloud.ply           (point cloud for viewing)")
        
    except Exception as e:
        print(f"\nSfM failed: {e}")
        import traceback
        traceback.print_exc()