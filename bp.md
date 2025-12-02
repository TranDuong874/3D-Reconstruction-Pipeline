┌─────────────────────────────────────────────────────────────────┐
│                    ROBUST SfM PIPELINE                          │
│                   (Video → Point Cloud)                         │
└─────────────────────────────────────────────────────────────────┘

INPUT: Video OR Image sequence
OUTPUT: Colored point cloud + Camera poses

┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PREPARATION                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Video Input                Image Sequence                       │
│      ↓                             ↓                             │
│  [Frame Extraction]            [Validation]                      │
│      ↓                             ↓                             │
│  ┌─────────────────────────────────────┐                        │
│  │ Frames (subsampled)                 │                        │
│  │ - Quality check                     │                        │
│  │ - Blur detection                    │                        │
│  │ - Resolution check                  │                        │
│  └─────────────────────────────────────┘                        │
│                  ↓                                               │
└──────────────────────────────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2: FEATURE EXTRACTION                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Decision Point: Which Features?                                │
│                                                                  │
│  ┌─────────────────┐          ┌──────────────────┐            │
│  │ Option A: SIFT  │          │ Option B: Modern │            │
│  │ (COLMAP native) │          │ (SuperPoint/     │            │
│  │                 │    OR    │  ALIKED/DISK +   │            │
│  │ - Fast          │          │  LightGlue)      │            │
│  │ - Simple        │          │                  │            │
│  │ - CPU/GPU       │          │ - Robust         │            │
│  └────────┬────────┘          │ - GPU required   │            │
│           │                    └────────┬─────────┘            │
│           └─────────┬──────────────────┘                       │
│                     ↓                                           │
│  ┌──────────────────────────────────────┐                      │
│  │ Features Per Image:                  │                      │
│  │ - Keypoints (x,y)                    │                      │
│  │ - Descriptors (128D or 256D)         │                      │
│  │ - Scores (confidence)                │                      │
│  └──────────────────────────────────────┘                      │
│                     ↓                                           │
└──────────────────────────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE MATCHING                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Decision Point: Matching Strategy                              │
│                                                                  │
│  Video (Sequential)        Images (Exhaustive)                  │
│      ↓                            ↓                              │
│  Sequential Matcher          Exhaustive Matcher                 │
│  - Match frame i with        - Match all pairs                  │
│    frames [i+1...i+K]        - (N×(N-1))/2 pairs                │
│  - K = overlap (10-20)       - Slower but thorough              │
│  - Fast!                     - Need vocab tree if >500 imgs     │
│      ↓                            ↓                              │
│  ┌─────────────────────────────────────┐                        │
│  │ Match Verification:                 │                        │
│  │ - Geometric verification (RANSAC)   │                        │
│  │ - Fundamental matrix estimation     │                        │
│  │ - Outlier rejection                 │                        │
│  └─────────────────────────────────────┘                        │
│                     ↓                                            │
│  ┌─────────────────────────────────────┐                        │
│  │ Match Database:                     │                        │
│  │ - Image pairs                       │                        │
│  │ - Matched keypoint indices          │                        │
│  │ - Inlier counts                     │                        │
│  └─────────────────────────────────────┘                        │
│                     ↓                                            │
└──────────────────────────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 4: STRUCTURE FROM MOTION (SfM)                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │ 4a. Initialization                  │                        │
│  │ - Find best initial pair            │                        │
│  │ - Triangulate initial points        │                        │
│  │ - Estimate camera intrinsics        │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 4b. Image Registration (Iterative)  │                        │
│  │ - Find next best image to register  │                        │
│  │ - PnP to estimate camera pose        │                        │
│  │ - Triangulate new points            │                        │
│  │ - Bundle adjustment (local)         │                        │
│  │ - Repeat until all images done      │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 4c. Global Bundle Adjustment        │                        │
│  │ - Refine all cameras + points       │                        │
│  │ - Minimize reprojection error       │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ Sparse Reconstruction:              │                        │
│  │ - Registered images: M/N            │                        │
│  │ - 3D points: ~10K-100K              │                        │
│  │ - Camera poses (R, t)               │                        │
│  │ - Camera intrinsics (K)             │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
└──────────────────────────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 5: DENSE RECONSTRUCTION (Optional)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Decision Point: Need dense?                                    │
│                                                                  │
│  YES (more points)           NO (sparse enough)                 │
│      ↓                              ↓                            │
│  ┌─────────────────┐           Skip to Stage 6                  │
│  │ 5a. Undistort   │                                            │
│  │ Images          │                                            │
│  └────────┬────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ 5b. Stereo      │                                            │
│  │ Matching        │                                            │
│  │ (PatchMatch)    │                                            │
│  └────────┬────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ 5c. Depth Maps  │                                            │
│  │ Fusion          │                                            │
│  └────────┬────────┘                                            │
│           ↓                                                      │
│  ┌─────────────────┐                                            │
│  │ Dense Point     │                                            │
│  │ Cloud           │                                            │
│  │ (~1M-10M pts)   │                                            │
│  └─────────────────┘                                            │
│           ↓                                                      │
└──────────────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6: POINT CLOUD PROCESSING                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │ 6a. Cleaning                        │                        │
│  │ - Remove statistical outliers       │                        │
│  │ - Remove low-confidence points      │                        │
│  │ - Radius filtering                  │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 6b. Downsampling (optional)         │                        │
│  │ - Voxel grid filter                 │                        │
│  │ - For faster processing             │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 6c. Normal Estimation               │                        │
│  │ - Compute point normals             │                        │
│  │ - For visualization/meshing         │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ Final Point Cloud:                  │                        │
│  │ - XYZ coordinates                   │                        │
│  │ - RGB colors                        │                        │
│  │ - Normals (optional)                │                        │
│  │ Format: PLY, PCD, etc.              │                        │
│  └─────────────────────────────────────┘                        │
│                 ↓                                                │
└──────────────────────────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 7: EVALUATION (if GT available)                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │ 7a. Pose Evaluation                 │                        │
│  │ - Load GT poses (TUM/ScanNet)       │                        │
│  │ - Align trajectories (Sim3)         │                        │
│  │ - Compute ATE (Abs Traj Error)      │                        │
│  │ - Compute RPE (Rel Pose Error)      │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 7b. Reconstruction Quality          │                        │
│  │ - Registration rate (% success)     │                        │
│  │ - Track length (avg obs per point)  │                        │
│  │ - Reprojection error                │                        │
│  │ - Number of 3D points               │                        │
│  └──────────────┬──────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ 7c. Geometric Accuracy (if GT mesh) │                        │
│  │ - Chamfer distance                  │                        │
│  │ - Precision/Recall                  │                        │
│  └─────────────────────────────────────┘                        │
│                 ↓                                                │
│  ┌─────────────────────────────────────┐                        │
│  │ Evaluation Report:                  │                        │
│  │ - Metrics summary                   │                        │
│  │ - Visualization plots               │                        │
│  │ - Comparison with baseline          │                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘