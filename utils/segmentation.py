"""
Semantic Segmentation Module

Uses Mask2Former for semantic segmentation with optional temporal consistency via RAFT optical flow.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2


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


class SemanticSegmenter:
    """Mask2Former Semantic Segmentation with optional temporal consistency"""
    
    def __init__(self, model_name: str = "facebook/mask2former-swin-base-ade-semantic",
                 device: str = "cuda", enable_temporal: bool = False):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            enable_temporal: Enable temporal consistency via RAFT optical flow
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.enable_temporal = enable_temporal
        
        print(f"  Loading Mask2Former on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()
        
        # Load RAFT for temporal consistency
        self.flow_model = None
        self.flow_transforms = None
        if enable_temporal:
            print(f"  Loading RAFT optical flow on {self.device}...")
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            weights = Raft_Small_Weights.DEFAULT
            self.flow_model = raft_small(weights=weights).to(self.device).eval()
            self.flow_transforms = weights.transforms()
        
        # State for temporal consistency
        self.prev_image = None
        self.prev_segmentation = None
    
    def segment(self, image: np.ndarray, flow_confidence_threshold: float = 0.4) -> np.ndarray:
        """
        Segment a single image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            flow_confidence_threshold: Threshold for using flow-warped labels
        
        Returns:
            Segmentation map as numpy array (H, W) with class indices
        """
        H, W = image.shape[:2]
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Run Mask2Former
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        raw_segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)]
        )[0].cpu().numpy()
        
        # Apply temporal consistency if enabled
        if self.enable_temporal and self.prev_image is not None and self.prev_segmentation is not None:
            segmentation = self._apply_temporal_consistency(
                image, raw_segmentation, flow_confidence_threshold
            )
        else:
            segmentation = raw_segmentation
        
        # Update state
        self.prev_image = image.copy()
        self.prev_segmentation = segmentation.copy()
        
        return segmentation.astype(np.int32)
    
    def _apply_temporal_consistency(self, curr_image: np.ndarray, raw_seg: np.ndarray,
                                     threshold: float) -> np.ndarray:
        """Apply temporal consistency using optical flow"""
        H, W = curr_image.shape[:2]
        
        # Compute optical flow from previous to current frame
        prev_tensor = torch.from_numpy(self.prev_image).permute(2, 0, 1).float().unsqueeze(0)
        curr_tensor = torch.from_numpy(curr_image).permute(2, 0, 1).float().unsqueeze(0)
        
        # Resize for RAFT (needs to be divisible by 8)
        h_raft = (H // 8) * 8
        w_raft = (W // 8) * 8
        prev_resized = F.interpolate(prev_tensor, size=(h_raft, w_raft), mode='bilinear', align_corners=False)
        curr_resized = F.interpolate(curr_tensor, size=(h_raft, w_raft), mode='bilinear', align_corners=False)
        
        # Apply RAFT transforms
        prev_raft, curr_raft = self.flow_transforms(prev_resized.to(self.device), curr_resized.to(self.device))
        
        with torch.no_grad():
            flow_list = self.flow_model(prev_raft, curr_raft)
            flow = flow_list[-1]
        
        # Resize flow back to original size
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        flow[:, 0] *= W / w_raft
        flow[:, 1] *= H / h_raft
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Warp previous segmentation
        warped_seg = self._warp_segmentation(self.prev_segmentation, flow)
        
        # Compute confidence
        confidence = self._compute_flow_confidence(flow)
        
        # Blend warped and new segmentation
        result = np.where(confidence > threshold, warped_seg, raw_seg)
        return result.astype(np.int32)
    
    def _warp_segmentation(self, prev_seg: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp previous segmentation using optical flow"""
        H, W = prev_seg.shape
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        src_x = (x + flow[:, :, 0]).astype(np.float32)
        src_y = (y + flow[:, :, 1]).astype(np.float32)
        warped = cv2.remap(
            prev_seg.astype(np.float32), 
            src_x, src_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        ).astype(np.int32)
        return warped
    
    def _compute_flow_confidence(self, flow: np.ndarray) -> np.ndarray:
        """Compute confidence mask based on flow gradient"""
        flow_dx = np.gradient(flow[:, :, 0], axis=1)
        flow_dy = np.gradient(flow[:, :, 1], axis=0)
        flow_mag = np.sqrt(flow_dx**2 + flow_dy**2)
        total_flow = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        confidence = np.exp(-flow_mag / 2.0) * np.exp(-total_flow / 100.0)
        return confidence.clip(0, 1)
    
    def reset_temporal_state(self):
        """Reset temporal consistency state (call when processing new sequence)"""
        self.prev_image = None
        self.prev_segmentation = None


def clean_segmentation(segmentation: np.ndarray, image: np.ndarray = None,
                       min_area: int = 500, morph_size: int = 5,
                       use_bilateral: bool = True) -> np.ndarray:
    """
    Clean up segmentation artifacts.
    
    Args:
        segmentation: Input segmentation map (H, W)
        image: Original RGB image for edge-aware filtering (optional)
        min_area: Minimum area for a region to keep
        morph_size: Morphological structuring element size
        use_bilateral: Use edge-aware smoothing
    
    Returns:
        Cleaned segmentation map
    """
    H, W = segmentation.shape
    cleaned = segmentation.copy()
    unique_labels = np.unique(segmentation)
    
    # Remove small connected components
    for label in unique_labels:
        mask = (segmentation == label).astype(np.uint8)
        num_components, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for comp_id in range(1, num_components):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_area:
                comp_mask = (labels_map == comp_id)
                dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
                neighbor_mask = dilated.astype(bool) & ~comp_mask
                
                if np.any(neighbor_mask):
                    neighbor_labels = segmentation[neighbor_mask]
                    neighbor_labels = neighbor_labels[neighbor_labels != label]
                    if len(neighbor_labels) > 0:
                        most_common = np.bincount(neighbor_labels).argmax()
                        cleaned[comp_mask] = most_common
    
    # Morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    smoothed = cleaned.copy()
    
    for label in unique_labels:
        mask = (cleaned == label).astype(np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        smoothed[opened == 1] = label
    
    # Edge-aware refinement
    if use_bilateral and image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_zone = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        final = smoothed.copy()
        final[edge_zone] = cleaned[edge_zone]
        return final.astype(np.int32)
    
    return smoothed.astype(np.int32)


def get_label_statistics(segmentation: np.ndarray) -> dict:
    """Get statistics of labels in a segmentation map"""
    unique, counts = np.unique(segmentation, return_counts=True)
    total = segmentation.size
    stats = {}
    for label_id, count in zip(unique, counts):
        name = ADE20K_CLASSES[label_id] if label_id < len(ADE20K_CLASSES) else f"unknown_{label_id}"
        stats[name] = {
            "id": int(label_id),
            "count": int(count),
            "percentage": float(count / total * 100)
        }
    return stats


def colorize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Convert segmentation indices to RGB colors"""
    return ADE20K_COLORS[segmentation % len(ADE20K_COLORS)]
