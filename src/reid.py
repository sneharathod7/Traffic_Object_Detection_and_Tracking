"""
reid.py — Vehicle-Specific Appearance Feature Extractor (OSNet)

This module implements a hybrid appearance descriptor for multi-object tracking,
combining:
  1. **OSNet-x0.25** deep embeddings — a lightweight Omni-Scale Network
     specifically designed for re-identification tasks.  OSNet uses multi-scale
     feature aggregation gates that learn to combine local fine-grained patterns
     (e.g., car logo, headlight shape) with global shape at multiple scales.
     Pre-trained on vehicle/person re-identification datasets, it produces
     far more discriminative embeddings for vehicles than a generic ImageNet
     backbone (ResNet50).
  2. **3D HSV Color Histograms** — a robust, illumination-tolerant colour
     signature that captures the dominant colour distribution of the vehicle
     crop.  Colour is the single strongest cue for distinguishing vehicles
     (a red sedan vs. a white sedan) and is completely free of neural-network
     inference cost.

The combined descriptor maintains identity across long intervals, occlusions,
and dense traffic clusters.

ARCHITECTURE CHOICE — WHY OSNET-x0.25
=======================================
  • Omni-Scale Feature Learning:  learns to aggregate features at multiple
    receptive-field scales simultaneously (unlike ResNet which has a single
    fixed receptive field at each layer).
  • Lightweight:  only 0.98M parameters (vs. ResNet50's 25M) — 25× fewer,
    yet produces more discriminative vehicle embeddings.
  • De-facto standard:  used by BoT-SORT, StrongSORT, Deep OC-SORT, and
    all top MOT methods.
  • 512-D output:  richer feature space than the previous 128-D projection.

FALLBACK CHAIN
===============
  1. OSNet-x0.25 via torchreid (primary — best accuracy, smallest model)
  2. ResNet50 with ImageNet weights (fallback if torchreid unavailable)
  3. SimpleCNN with random weights (emergency — no internet required)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)

# OSNet input size — standard re-identification crop dimensions
# Height > Width preserves vehicle aspect ratio in aerial/frontal views
OSNET_INPUT_H = 256
OSNET_INPUT_W = 128


class SimpleCNN(nn.Module):
    """
    Fallback lightweight CNN for feature extraction when neither OSNet
    nor pre-trained ResNet weights can be loaded.
    """
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class ReIDExtractor:
    """
    Extracts high-quality appearance embeddings using OSNet-x0.25.

    Combines unit-normalized OSNet deep embeddings (512-D) and
    3D HSV Color Histograms (512-D) for robust vehicle re-identification.

    The OSNet model is loaded via the ``torchreid`` library.  If torchreid
    is not installed, falls back to ResNet50 (ImageNet) → SimpleCNN.
    """

    def __init__(self, device: Optional[str] = None, embedding_dim: int = 512) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.model: nn.Module
        self._input_h = OSNET_INPUT_H
        self._input_w = OSNET_INPUT_W
        self._model_name = "unknown"

        # ---- Attempt 1: OSNet-x0.25 via torchreid ----------------------------
        try:
            from torchreid.utils import FeatureExtractor
            logger.info("Loading OSNet-x0.25 via torchreid for vehicle ReID...")
            self._feature_extractor = FeatureExtractor(
                model_name='osnet_x0_25',
                model_path='',  # empty string triggers auto-download of pretrained weights
                device=self.device,
            )
            # Override embedding_dim to match OSNet output
            self.embedding_dim = 512
            self._model_name = "osnet_x0_25"
            self._use_torchreid = True
            logger.info(
                "Successfully loaded OSNet-x0.25 (0.98M params, 512-D embeddings) on %s.",
                self.device,
            )
            return  # Done — OSNet is ready
        except ImportError:
            logger.warning(
                "torchreid not installed. Install with: pip install torchreid. "
                "Falling back to ResNet50."
            )
        except Exception as e:
            logger.warning(
                "Could not load OSNet-x0.25 via torchreid (%s). "
                "Falling back to ResNet50.",
                e,
            )

        self._use_torchreid = False
        self._feature_extractor = None

        # ---- Attempt 2: ResNet50 with ImageNet pre-training ------------------
        try:
            logger.info("Loading pre-trained ResNet50 for ReID (fallback)...")
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
            self.model = backbone
            self._model_name = "resnet50_imagenet"
            logger.info("Successfully loaded pre-trained ResNet50 for ReID.")
        except Exception as e:
            logger.warning(
                "Could not load pre-trained ResNet50 (%s). "
                "Falling back to lightweight SimpleCNN.",
                e,
            )
            self.model = SimpleCNN(embedding_dim=embedding_dim)
            self._model_name = "simple_cnn_random"

        self.model.to(self.device)
        self.model.eval()

    def extract_cnn_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract unit-normalized deep CNN features from BGR image crops.

        For OSNet: uses the torchreid FeatureExtractor directly.
        For fallback models: uses the standard PyTorch forward pass.
        """
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._use_torchreid and self._feature_extractor is not None:
            # torchreid FeatureExtractor accepts a list of PIL-like images
            # or numpy arrays.  We resize to 256×128 and convert BGR→RGB.
            processed = []
            for crop in crops:
                if crop.size == 0:
                    crop = np.zeros((self._input_h, self._input_w, 3), dtype=np.uint8)
                img = cv2.resize(crop, (self._input_w, self._input_h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed.append(img)

            with torch.no_grad():
                embeddings = self._feature_extractor(processed)
                # torchreid returns a torch.Tensor of shape (N, 512)
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy()

        # ---- Fallback: manual PyTorch inference (ResNet50 / SimpleCNN) --------
        tensors = []
        for crop in crops:
            if crop.size == 0:
                crop = np.zeros((self._input_h, self._input_w, 3), dtype=np.uint8)
            img = cv2.resize(crop, (self._input_w, self._input_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            # Normalize with standard ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()

    def extract_color_hist(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute a normalized 3D HSV Color Histogram.
        Bins: 8 H bins, 8 S bins, 8 V bins = 512 dimensions.
        """
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_combined(self, crops: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Extract combined CNN and Color features from list of BGR image crops.
        """
        if not crops:
            return []

        cnn_embs = self.extract_cnn_features(crops)
        results = []
        for i, crop in enumerate(crops):
            color_emb = self.extract_color_hist(crop)
            results.append({
                "cnn": cnn_embs[i],
                "color": color_emb,
            })
        return results


def compute_appearance_distance(
    emb1: Dict[str, np.ndarray],
    emb2: Dict[str, np.ndarray],
    w_cnn: float = 0.7,
    w_color: float = 0.3,
) -> float:
    """
    Compute a combined appearance distance between two appearance descriptors.

    Uses cosine distance for both CNN features and HSV Color Histograms.
    Weights are 70% CNN / 30% Color to leverage the stronger OSNet backbone.

    Cosine Distance = 1.0 - Cosine Similarity
    """
    # CNN Cosine Similarity (dot product since they are unit-normalized)
    cnn_sim = float(np.dot(emb1["cnn"], emb2["cnn"]))
    d_cnn = 1.0 - max(-1.0, min(1.0, cnn_sim))

    # Color Cosine Similarity
    color_norm1 = np.linalg.norm(emb1["color"]) + 1e-7
    color_norm2 = np.linalg.norm(emb2["color"]) + 1e-7
    color_sim = float(np.dot(emb1["color"], emb2["color"])) / (color_norm1 * color_norm2)
    d_color = 1.0 - max(-1.0, min(1.0, color_sim))

    return w_cnn * d_cnn + w_color * d_color
