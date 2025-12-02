"""
Debug visualization script for feature tracks.
Usage: python visualize_tracks.py
"""
import cv2
import numpy as np
from tqdm import tqdm
from utils.video_loader import VideoLoader
from utils.feature_extractor import FeatureExtractor
from utils.feature_matcher import FeatureMatcher


def build_tracks(features: dict, matcher: FeatureMatcher) -> dict:
    """Build feature tracks by chaining matches across consecutive frames."""
    indices = sorted(features.keys())
    
    # Track structure: track_id -> list of (frame_idx, keypoint_xy)
    tracks = {}
    next_track_id = 0
    
    # Map: (frame_idx, keypoint_idx) -> track_id
    kp_to_track = {}
    
    for i in tqdm(range(len(indices) - 1), desc="Building tracks"):
        idx0, idx1 = indices[i], indices[i + 1]
        match = matcher.match_pair(features[idx0], features[idx1], idx0, idx1)
        
        for m in match.matches:
            kp_idx0, kp_idx1 = int(m[0]), int(m[1])
            pt0 = features[idx0].keypoints[kp_idx0]
            pt1 = features[idx1].keypoints[kp_idx1]
            
            key0 = (idx0, kp_idx0)
            key1 = (idx1, kp_idx1)
            
            if key0 in kp_to_track:
                # Extend existing track
                track_id = kp_to_track[key0]
                tracks[track_id].append((idx1, pt1))
                kp_to_track[key1] = track_id
            else:
                # Start new track
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [(idx0, pt0), (idx1, pt1)]
                kp_to_track[key0] = track_id
                kp_to_track[key1] = track_id
    
    return tracks


def visualize_tracks(frames: list, tracks: dict, output_path: str = 'tracks.mp4', 
                     min_track_length: int = 3, fps: int = 10, point_radius: int = 2,
                     show_trails: bool = False):
    """
    Visualize feature tracks on video.
    
    Args:
        frames: List of Frame objects
        tracks: Dictionary of track_id -> list of (frame_idx, keypoint_xy)
        output_path: Output video path
        min_track_length: Minimum track length to display
        fps: Output video FPS
        point_radius: Radius of feature points
        show_trails: Whether to show trajectory trails
    """
    # Filter tracks by length
    long_tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_length}
    print(f"Tracks with {min_track_length}+ frames: {len(long_tracks)}")
    
    # Assign colors to tracks
    np.random.seed(42)
    track_colors = {track_id: tuple(map(int, np.random.randint(0, 255, 3))) 
                    for track_id in long_tracks.keys()}
    
    # Output video
    h, w = frames[0].image.shape[:2]
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in tqdm(frames, desc="Writing video"):
        img = frame.image.copy()
        
        for track_id, track_points in long_tracks.items():
            color = track_colors[track_id]
            
            if show_trails:
                # Get points up to current frame
                visible_points = [(fidx, pt) for fidx, pt in track_points if fidx <= frame.idx]
                
                # Draw trail
                if len(visible_points) >= 2:
                    for j in range(len(visible_points) - 1):
                        pt1 = tuple(map(int, visible_points[j][1]))
                        pt2 = tuple(map(int, visible_points[j + 1][1]))
                        cv2.line(img, pt1, pt2, color, 1)
                
                # Draw current point
                if visible_points:
                    last_fidx, last_pt = visible_points[-1]
                    if last_fidx == frame.idx:
                        cv2.circle(img, tuple(map(int, last_pt)), point_radius, color, -1)
            else:
                # Just draw points in current frame
                for fidx, pt in track_points:
                    if fidx == frame.idx:
                        cv2.circle(img, tuple(map(int, pt)), point_radius, color, -1)
                        break
        
        out_video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    out_video.release()
    print(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    # Load frames
    print("Loading frames...")
    loader = VideoLoader(subsample=2, blur_threshold=50.0)
    frames = loader.load_images('data/rgbd_dataset_freiburg1_xyz/rgb')
    print(f"Loaded {len(frames)} frames")
    
    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor(method='disk', max_features=4096, device='cuda', batch_size=4)
    features = extractor.extract_all(frames)
    print(f"Extracted features from {len(features)} frames")
    
    # Initialize matcher
    print("Initializing matcher...")
    matcher = FeatureMatcher(device='cuda', match_mode='snn', th=0.8)
    
    # Build tracks
    print("Building tracks...")
    tracks = build_tracks(features, matcher)
    print(f"Built {len(tracks)} tracks")
    
    # Visualize
    visualize_tracks(frames, tracks, output_path='tracks.mp4', 
                     min_track_length=3, point_radius=2, show_trails=False)
