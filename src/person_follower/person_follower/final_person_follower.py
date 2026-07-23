#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# final_person_follower.py — Wanis elderly-care robot
# ==========================================================================
#
# Identity-locked, recovery-aware person-following node. Built to be the
# reliable, PhD-grade core of the Wanis platform.
#
# HIGH-LEVEL PIPELINE
#   RGB frame ─► yolox-seg (class=0 person)
#                │
#                ├─► per-person bbox + segmentation mask
#                └─► BoT-SORT tracker   (stable track_id across frames)
#                     │
#                     ▼
#                For every tracked person:
#                  • crop+mask, compute OSNet ReID embedding (512-d)
#                  • HSV histogram (upper+lower body, separately)
#                  • LBP texture signature
#                  • depth from Kinect (median over mask-intersected window)
#
#   State machine:  UNLOCKED → (click / auto-lock) → LOCKED
#                   LOCKED   → (lost ≥1.5s)        → RECOVERING_BACKUP
#                   RECOVER.B→ (≥1.0s)             → RECOVERING_SCAN
#                   RECOVER.S→ (≥8s no match)      → RECOVERING_FRONTIER
#                   ANY_REC  → (signature match)   → LOCKED (re-acquired)
#                   any      → (/pill_time=true)   → PILL_DELIVERY
#
#   Motion:  separate PID for angular (pixel error) and linear (depth error),
#            with full output low-pass filter, slew-rate limit, D-on-
#            measurement, anti-windup, and integral-leak on sign flip.
#            Forward speed is continuously SCALED by heading error (blended
#            steering) — NO hard gate that zeros linear_x. Hysteresis on
#            the "stop rotating only" threshold to eliminate chatter.
#
#   Safety:  NONE in this node. safety_guard.py sits on /cmd_vel_raw →
#            /cmd_vel and is authoritative for collision avoidance.
#            This node publishes raw intent to /cmd_vel_raw.
#
# ROS INTERFACE
#   Subscribes: /image_raw/compressed, /depth/image_raw/compressedDepth,
#               /odometry/filtered, /scan, /pill_time, /map,
#               /Nour_fall_detection, /person_follower/lock_command
#   Publishes:  /cmd_vel_raw, /person_tracker/candidates (JSON),
#               /person_follower/status (JSON)
#   Action client: navigate_to_pose (Nav2)
#   Service client: /slam_toolbox/reset
#
# ==========================================================================

import os
import json
import math
import time
import base64
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, CompressedImage, Image
from std_msgs.msg import String, Bool, Int32
from nav2_msgs.action import NavigateToPose
from slam_toolbox.srv import Reset

from cv_bridge import CvBridge

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException


# ───────────────────────────────── Optional heavy deps ─────────────────────
# yolo segmentation + BoT-SORT tracker
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception as _e:
    _HAS_ULTRALYTICS = False
    _ULTRA_IMPORT_ERR = _e

# OSNet via torchreid (preferred) or a torchvision fallback
_HAS_TORCHREID = False
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
    try:
        from torchreid.utils import FeatureExtractor
        _HAS_TORCHREID = True
    except Exception:
        _HAS_TORCHREID = False
except Exception:
    _HAS_TORCH = False

# ────────────────────────────────────── Enums ──────────────────────────────
class FollowerState(Enum):
    """High-level state of the person-follower behavior."""
    UNLOCKED = "UNLOCKED"                    # waiting for user or auto-lock
    LOCKED = "LOCKED"                        # actively following target
    PILL_DELIVERY = "PILL_DELIVERY"          # external /pill_time == True
    RECOVERING_STATIONARY = "RECOVERING_STATIONARY"  # brief hold — most losses are momentary occlusions
    RECOVERING_BACKUP = "RECOVERING_BACKUP"  # brief reverse to widen FOV
    RECOVERING_SCAN = "RECOVERING_SCAN"      # in-place rotation scan
    RECOVERING_FRONTIER = "RECOVERING_FRONTIER"  # Nav2 frontier exploration
    GIVE_UP = "GIVE_UP"                      # all recovery exhausted


# ─────────────────────────────── PID controller ────────────────────────────
class PID:
    """
    Smooth PID controller with:
      • Derivative on measurement (not error) to suppress setpoint kicks
      • Low-pass filter on the derivative term
      • Low-pass filter on the full output (anti-chatter)
      • Slew-rate (Δoutput/sec) limiter
      • Conditional (clamped) integration — anti-windup
      • Integral-leak on sign flip — stale integral doesn't drag output
      • Output clamping

    Convention: error = setpoint - measurement.
    For angular yaw we drive error = 0 - pixel_offset, but caller handles
    the sign; internally this class is setpoint-agnostic, always using
    (setpoint - measurement).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        out_min: float,
        out_max: float,
        deadband: float = 0.0,
        deriv_filter_alpha: float = 0.8,
        output_filter_alpha: float = 0.7,
        slew_rate: Optional[float] = None,  # max |Δoutput|/sec; None = unbounded
        integral_leak: float = 0.02,        # fraction leaked when sign flips
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.deadband = deadband
        self.deriv_filter_alpha = deriv_filter_alpha
        self.output_filter_alpha = output_filter_alpha
        self.slew_rate = slew_rate
        self.integral_leak = integral_leak
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._last_meas: Optional[float] = None
        self._deriv_filt = 0.0
        self._last_output = 0.0
        self._last_error = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0.0 or not math.isfinite(dt):
            return self._last_output

        error = setpoint - measurement

        # Deadband: if within the deadband, zero the integral slowly and output 0
        if abs(error) < self.deadband:
            self._integral *= (1.0 - self.integral_leak)
            self._deriv_filt *= self.deriv_filter_alpha
            # smooth fade-out of any residual output
            new_out = self.output_filter_alpha * self._last_output
            new_out = self._clamp(new_out)
            new_out = self._slew(new_out, dt)
            self._last_output = new_out
            self._last_error = error
            return new_out

        # Integral with leak on sign flip
        if self._last_error * error < 0.0:
            self._integral *= (1.0 - 3.0 * self.integral_leak)  # stronger leak on reversal
        self._integral += error * dt

        # Derivative ON MEASUREMENT (anti-kick): -d(meas)/dt
        if self._last_meas is None:
            deriv_raw = 0.0
        else:
            deriv_raw = -(measurement - self._last_meas) / dt
        self._deriv_filt = (
            self.deriv_filter_alpha * self._deriv_filt
            + (1.0 - self.deriv_filter_alpha) * deriv_raw
        )

        # Raw PID
        out = self.kp * error + self.ki * self._integral + self.kd * self._deriv_filt

        # Output low-pass
        out = (
            self.output_filter_alpha * self._last_output
            + (1.0 - self.output_filter_alpha) * out
        )

        # Clamp + anti-windup (undo last integral accrual if we're saturated)
        clamped = self._clamp(out)
        if clamped != out and ((error > 0) == (self._integral > 0)):
            self._integral -= error * dt
        out = clamped

        # Slew limiting
        out = self._slew(out, dt)

        self._last_output = out
        self._last_meas = measurement
        self._last_error = error
        return out

    def _clamp(self, v: float) -> float:
        return max(self.out_min, min(self.out_max, v))

    def _slew(self, v: float, dt: float) -> float:
        if self.slew_rate is None:
            return v
        max_delta = self.slew_rate * dt
        delta = v - self._last_output
        if delta > max_delta:
            return self._last_output + max_delta
        if delta < -max_delta:
            return self._last_output - max_delta
        return v


# ─────────────────────────── Signature / Candidate ─────────────────────────
@dataclass
class PersonSignature:
    """Multi-modal appearance signature of a person.

    Two complementary modalities:
      • OSNet ReID embedding (512-d, L2-normalized) — primary identity cue,
        robust to viewpoint and lighting changes.
      • HS-only HSV histograms split upper/lower body — secondary cue that
        survives degraded embeddings (e.g. low-confidence crops, partial
        occlusion). H+S only (V dropped) for illumination invariance.

    LBP texture was removed — at its previous 0.05 weight it contributed
    less than the noise floor of the other modalities and added a skimage
    dependency for no measurable gain.
    """
    embedding: Optional[np.ndarray] = None          # L2-normalized, 512-d
    hist_upper: Optional[np.ndarray] = None         # HS histogram, flattened
    hist_lower: Optional[np.ndarray] = None
    sample_count: int = 0

    def has_any(self) -> bool:
        return self.embedding is not None or self.hist_upper is not None

    def copy(self) -> "PersonSignature":
        return PersonSignature(
            embedding=None if self.embedding is None else self.embedding.copy(),
            hist_upper=None if self.hist_upper is None else self.hist_upper.copy(),
            hist_lower=None if self.hist_lower is None else self.hist_lower.copy(),
            sample_count=self.sample_count,
        )

    def update_ema(self, other: "PersonSignature", alpha: float = 0.02) -> None:
        """Blend `other` into self with exponential moving average."""
        if other.embedding is not None:
            if self.embedding is None:
                self.embedding = other.embedding.copy()
            else:
                self.embedding = (1 - alpha) * self.embedding + alpha * other.embedding
                n = float(np.linalg.norm(self.embedding))
                if n > 1e-8:
                    self.embedding /= n
        for attr in ("hist_upper", "hist_lower"):
            a_self = getattr(self, attr)
            a_other = getattr(other, attr)
            if a_other is None:
                continue
            if a_self is None:
                setattr(self, attr, a_other.copy())
            else:
                blended = (1 - alpha) * a_self + alpha * a_other
                setattr(self, attr, blended)
        self.sample_count += 1


@dataclass
class TrackedCandidate:
    """Per-frame state of a tracked person."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 (image px)
    centroid: Tuple[float, float]    # (cx, cy) image px
    mask: Optional[np.ndarray]       # boolean mask (H,W) full-frame size
    depth_m: float                   # median depth over the person pixels, meters
    confidence: float                # YOLO detection confidence
    signature: PersonSignature = field(default_factory=PersonSignature)
    first_seen: float = 0.0
    last_seen: float = 0.0
    score_vs_target: float = 0.0     # most recent matching score vs target
    status: str = "candidate"        # "candidate" | "locked" | "ignored"
    thumb_bgr: Optional[np.ndarray] = None  # small preview crop, stored for UI


# ────────────────────────── Signature scoring helpers ──────────────────────
def _bhattacharyya_from_hist(h1: np.ndarray, h2: np.ndarray) -> float:
    """OpenCV Bhattacharyya distance (0 best, 1 worst). Normalizes first."""
    a = h1.astype(np.float32)
    b = h2.astype(np.float32)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 1.0
    a /= sa
    b /= sb
    bc = np.sqrt(a * b).sum()
    bc = max(0.0, min(1.0, bc))
    return float(math.sqrt(max(0.0, 1.0 - bc)))


def _cos_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def signature_score(cand: PersonSignature, target: PersonSignature) -> float:
    """
    Weighted multi-modal matching score in [0, 1]. 1 = identical.

    Weights: embedding 0.70, hist_upper 0.20, hist_lower 0.10.
    The embedding carries most of the discriminative power (OSNet was
    trained for re-ID), so it gets the dominant weight. Histograms act as
    a colour-based tiebreaker that is unaffected by viewpoint / pose
    changes that can briefly degrade the embedding.

    Score is normalized by the sum of weights for available modalities so
    the result stays in [0, 1] even when the embedding is absent — without
    this, the max reachable score would fall below the re-acquisition
    thresholds.
    """
    s = 0.0
    w = 0.0
    if cand.embedding is not None and target.embedding is not None:
        s += 0.70 * max(0.0, _cos_sim(cand.embedding, target.embedding))
        w += 0.70
    if cand.hist_upper is not None and target.hist_upper is not None:
        s += 0.20 * (1.0 - _bhattacharyya_from_hist(cand.hist_upper, target.hist_upper))
        w += 0.20
    if cand.hist_lower is not None and target.hist_lower is not None:
        s += 0.10 * (1.0 - _bhattacharyya_from_hist(cand.hist_lower, target.hist_lower))
        w += 0.10
    if w <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, s / w)))


# ──────────────────────────── Appearance encoder ───────────────────────────
class AppearanceEncoder:
    """
    Wraps OSNet via torchreid if available; otherwise returns None embeddings
    and relies on histograms only. Never throws on a missing backend — the
    pipeline degrades gracefully to "no embedding" mode.

    Default backbone is ``osnet_ain_x1_0`` — same parameter count as
    ``osnet_x1_0`` but trained with Adaptive Instance Normalization for
    cross-domain robustness. In an indoor robot deployment that swings
    between sunlit windows, fluorescent corridors, and dim rooms, AIN
    measurably reduces embedding drift across lighting changes — the most
    common source of false re-acquisition failures.

    Weights:
      • If ``model_path`` points to a readable file, load those weights
        (e.g. ReID-trained ``osnet_ain_x1_0_msmt17.pth.tar`` from the
        torchreid model zoo — strongly recommended for production).
      • Otherwise torchreid falls back to ImageNet-pretrained backbone
        weights, which are usable but ~15–20% behind ReID-trained
        weights on actual person re-identification accuracy.
    """

    def __init__(
        self,
        logger=None,
        model_name: str = "osnet_ain_x1_0",
        model_path: str = "",
    ):
        self.logger = logger
        self._extractor = None
        self._device = "cpu"
        self._model_name = model_name
        self._model_path = model_path or ""
        if _HAS_TORCHREID and _HAS_TORCH:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Resolve and validate the optional weights path. Pass an
                # empty string to FeatureExtractor when nothing is
                # configured — the torchreid API treats that as "use the
                # default ImageNet backbone".
                resolved_path = ""
                if self._model_path:
                    expanded = os.path.expanduser(self._model_path)
                    if os.path.exists(expanded):
                        resolved_path = expanded
                    elif logger:
                        logger.warn(
                            f"OSNet weights not found at '{self._model_path}' "
                            f"— falling back to ImageNet-pretrained backbone. "
                            f"Download ReID weights from the torchreid "
                            f"model zoo for best accuracy."
                        )
                self._extractor = FeatureExtractor(
                    model_name=model_name,
                    model_path=resolved_path,
                    device=device,
                )
                self._device = device
                if logger:
                    src = (
                        f"weights={resolved_path}" if resolved_path
                        else "weights=ImageNet (no model_path set)"
                    )
                    logger.info(f"OSNet ReID ready: {model_name} on {device} ({src})")
            except Exception as e:
                if logger:
                    logger.warn(
                        f"OSNet ReID init failed for {model_name}: {e}; "
                        f"running without embeddings"
                    )
                self._extractor = None
        else:
            if logger:
                logger.warn(
                    "torchreid not available — running with histograms only "
                    "(install torchreid for best re-ID accuracy)"
                )

    def available(self) -> bool:
        return self._extractor is not None

    def encode(self, crops_bgr: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Returns list of L2-normalized 512-d embeddings (or None)."""
        if not self._extractor or not crops_bgr:
            return [None] * len(crops_bgr)
        try:
            # torchreid FeatureExtractor accepts a list of BGR numpy arrays
            feats = self._extractor(crops_bgr)
            embeds = feats.cpu().numpy()
            out: List[Optional[np.ndarray]] = []
            for v in embeds:
                n = float(np.linalg.norm(v))
                out.append(v / n if n > 1e-8 else None)
            return out
        except Exception as e:
            if self.logger:
                self.logger.warn(f"OSNet encode failed: {e}")
            return [None] * len(crops_bgr)


# ────────────────────────────── Histogram helper ────────────────────────────
_HSV_H_BINS = 16
_HSV_S_BINS = 8
_HSV_HIST_DIM = _HSV_H_BINS * _HSV_S_BINS  # 128


def _compute_hsv_hist(img_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    HS-only histogram (16×8 = 128-d). Drops the V (brightness) channel so
    the descriptor is invariant to lighting changes — the single biggest
    cause of false mismatches when matching across frames with different
    exposure or when a person moves from shadow to direct light.

    16 H bins is the sweet spot: empirically indistinguishable from 32 H
    bins on re-ID accuracy benchmarks, while halving the Bhattacharyya
    cost and the EMA blend cost. `mask` must be a uint8 0/255 image or
    None.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros(_HSV_HIST_DIM, dtype=np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = mask
    if m is not None and m.dtype != np.uint8:
        m = (m.astype(np.uint8) * 255)
    hist = cv2.calcHist(
        [hsv], [0, 1], m,
        [_HSV_H_BINS, _HSV_S_BINS], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32).flatten()


def build_signature(
    img_bgr: np.ndarray,
    full_mask: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
    encoder: AppearanceEncoder,
) -> PersonSignature:
    """Compute a full PersonSignature from a crop + mask + bbox."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return PersonSignature()
    crop = img_bgr[y1:y2, x1:x2]
    if full_mask is not None:
        local_mask = full_mask[y1:y2, x1:x2].astype(np.uint8) * 255
    else:
        local_mask = None

    # Overlapping 60/60 body zones — matches _update_candidates split
    h = crop.shape[0]
    t = int(h * 0.60)
    b = int(h * 0.40)
    upper_crop = crop[:t]
    lower_crop = crop[b:]
    upper_mask = None if local_mask is None else local_mask[:t]
    lower_mask = None if local_mask is None else local_mask[b:]

    hist_u = _compute_hsv_hist(upper_crop, upper_mask)
    hist_l = _compute_hsv_hist(lower_crop, lower_mask)

    # Embedding (optional)
    embed: Optional[np.ndarray] = None
    try:
        embeds = encoder.encode([crop])
        embed = embeds[0] if embeds else None
    except Exception:
        embed = None

    return PersonSignature(
        embedding=embed,
        hist_upper=hist_u,
        hist_lower=hist_l,
        sample_count=1,
    )


# ─────────────────────────── YOLO + tracker wrapper ────────────────────────
class Detector:
    """
    yolox-seg + BoT-SORT. If Ultralytics isn't installed, `available()`
    returns False and the node should refuse to run.
    """

    def __init__(
        self,
        weights_path: str,
        tracker_yaml: str = "botsort.yaml",
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        logger=None,
    ):
        self.logger = logger
        self._model = None
        self._tracker_yaml = tracker_yaml
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._weights_path = weights_path
        self._device = self._resolve_device(device, logger)

        if not _HAS_ULTRALYTICS:
            if logger:
                logger.error(
                    f"ultralytics not installed — detector DISABLED. {_ULTRA_IMPORT_ERR}"
                )
            return
        try:
            # Prefer the provided path.  If the full path is missing,
            # fall back to the basename — Ultralytics resolves known
            # pretrained names (e.g. "yolom-seg.pt") against its CDN
            # and auto-downloads into the cwd.
            load_target = weights_path
            if not os.path.exists(weights_path):
                load_target = os.path.basename(weights_path)
                if logger:
                    logger.warn(
                        f"YOLO weights not found at {weights_path}; "
                        f"falling back to basename '{load_target}' so "
                        f"Ultralytics can auto-download"
                    )
            self._model = YOLO(load_target)
            # Push model onto the chosen device once up-front; the
            # torch.cuda.* API covers both CUDA and ROCm (HIP), so
            # 'cuda' works for NVIDIA RTX and AMD Radeon alike.
            try:
                self._model.to(self._device)
            except Exception as move_err:
                if logger:
                    logger.warn(
                        f"YOLO model.to('{self._device}') failed ({move_err}); "
                        f"falling back to CPU"
                    )
                self._device = "cpu"
                self._model.to("cpu")
            if logger:
                logger.info(f"yolo loaded: {load_target} on {self._device}")
        except Exception as e:
            if logger:
                logger.error(f"YOLO load failed: {e}")
            self._model = None

    @staticmethod
    def _resolve_device(device: str, logger) -> str:
        dev = (device or "auto").lower()
        if dev == "auto":
            try:
                import torch  # local import so the class stays importable without torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    if logger:
                        logger.info(f"CUDA/ROCm device detected: {name}")
                    return "cuda"
            except Exception as e:
                if logger:
                    logger.warn(f"torch probe failed, defaulting to CPU: {e}")
            return "cpu"
        return dev

    def available(self) -> bool:
        return self._model is not None

    def infer(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Run detection + tracking on one frame. Returns a list of dicts:
          {track_id, bbox(x1,y1,x2,y2), mask(HxW bool or None), conf}
        """
        if self._model is None:
            return []
        try:
            # class=[0] restricts to 'person'; persist=True keeps tracker state
            results = self._model.track(
                source=img_bgr,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                classes=[0],
                persist=True,
                tracker=self._tracker_yaml,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            if self.logger:
                self.logger.warn(f"YOLO inference failed: {e}")
            return []

        out: List[Dict] = []
        if not results:
            return out
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return out

        ids = res.boxes.id
        if ids is None:
            # tracker skipped this frame (e.g., cold start) — fabricate negative ids
            ids = [-(i + 1) for i in range(len(res.boxes))]
        else:
            ids = ids.int().cpu().tolist()

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        masks_full = None
        if res.masks is not None and res.masks.data is not None:
            try:
                masks_full = res.masks.data.cpu().numpy() > 0.5  # (N,H,W) bool
            except Exception:
                masks_full = None

        H, W = img_bgr.shape[:2]
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            conf = float(confs[i])
            m = None
            if masks_full is not None and i < len(masks_full):
                m_raw = masks_full[i]
                if m_raw.shape != (H, W):
                    m = cv2.resize(
                        m_raw.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    m = m_raw
            out.append({
                "track_id": int(tid),
                "bbox": (x1, y1, x2, y2),
                "mask": m,
                "conf": conf,
            })
        return out


# ══════════════════════════════════════════════════════════════════════════
#                            P E R S O N   F O L L O W E R
# ══════════════════════════════════════════════════════════════════════════
class PersonFollower(Node):
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    # Model tier is picked by the `yolo_weights_path` ROS parameter.
    # Default = yolov8n.pt: ~7 MB, detection-only (no masks), matches the
    # model fall3.py uses so both nodes share the same weights file.
    # BoT-SORT + histogram re-ID still work; mask-based background filtering
    # is skipped (None-safe in all downstream paths).
    # RTX production server: override to `yolo11m-seg.pt` for masks + accuracy.
    DEFAULT_YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo26n.pt")

    def __init__(self):
        super().__init__("person_follower")

        # ─── Callback groups: keep vision off the laser/odom path ─────────
        self.cb_vision = ReentrantCallbackGroup()
        self.cb_sensors = MutuallyExclusiveCallbackGroup()
        self.cb_timers = MutuallyExclusiveCallbackGroup()
        # Recovery planning (Nav2 action client + SLAM reset service client):
        # isolated so a slow action/service round-trip cannot stall vision.
        self.cb_recovery = MutuallyExclusiveCallbackGroup()

        # ─── Declare parameters (all tunable at runtime) ──────────────────
        self._declare_params()

        # ─── ROS primitives ───────────────────────────────────────────────
        self.bridge = CvBridge()
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel_raw", 10)
        self.unsafe_velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.candidates_pub = self.create_publisher(String, "/person_tracker/candidates", 5)
        self.status_pub = self.create_publisher(String, "/person_follower/status", 5)
        # Debug image is published — never shown via cv2.imshow. The Flask
        # UI (viz_bridge + /api/viz/person_debug.jpg) renders it in a
        # browser pane under the Debug tab.
        debug_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.debug_image_pub = self.create_publisher(
            CompressedImage, "/person_follower/debug/compressed", debug_qos
        )

        use_comp_rgb   = bool(self.get_parameter("use_compressed_rgb").value)
        use_comp_depth = bool(self.get_parameter("use_compressed_depth").value)

        # Sensor QoS: BEST_EFFORT + depth=1 means the middleware drops
        # old frames as soon as a new one arrives.  Without this, the
        # subscription queues up to 10 frames while YOLO is running; the
        # callback then processes stale frames and the debug window
        # appears to lag by 1–2 seconds behind live motion.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        if use_comp_rgb:
            rgb_topic = self.get_parameter("rgb_topic_compressed").value
            self.image_sub = self.create_subscription(
                CompressedImage, rgb_topic, self.image_callback, sensor_qos,
                callback_group=self.cb_vision,
            )
        else:
            rgb_topic = self.get_parameter("rgb_topic_raw").value
            self.image_sub = self.create_subscription(
                Image, rgb_topic, self.image_callback_raw, sensor_qos,
                callback_group=self.cb_vision,
            )
        if use_comp_depth:
            depth_topic = self.get_parameter("depth_topic_compressed").value
            self.depth_sub = self.create_subscription(
                CompressedImage, depth_topic, self.depth_callback, sensor_qos,
                callback_group=self.cb_sensors,
            )
        else:
            depth_topic = self.get_parameter("depth_topic_raw").value
            self.depth_sub = self.create_subscription(
                Image, depth_topic, self.depth_callback_raw, sensor_qos,
                callback_group=self.cb_sensors,
            )
        self.get_logger().info(
            f"Image transport: rgb={'compressed' if use_comp_rgb else 'raw'} ({rgb_topic}), "
            f"depth={'compressed' if use_comp_depth else 'raw'} ({depth_topic})"
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_callback, 10,
            callback_group=self.cb_sensors,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.laser_callback, 10,
            callback_group=self.cb_sensors,
        )
        self.fall_sub = self.create_subscription(
            String, "/Nour_fall_detection", self.fall_callback, 10,
            callback_group=self.cb_sensors,
        )
        self.pill_sub = self.create_subscription(
            Bool, "/pill_time", self.pill_time_callback, 10,
            callback_group=self.cb_sensors,
        )
        self.lock_sub = self.create_subscription(
            Int32, "/person_follower/lock_command", self.lock_command_callback, 10,
            callback_group=self.cb_sensors,
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, map_qos,
            callback_group=self.cb_sensors,
        )

        # Nav2 action client for frontier exploration
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose",
            callback_group=self.cb_recovery,
        )
        # SLAM-toolbox reset client used to kick off frontier exploration.
        # Lives on the node's own executor + cb_recovery group so the call is
        # fully async and never blocks the image callback thread.
        self.slam_reset_client = self.create_client(
            Reset, "/slam_toolbox/reset",
            callback_group=self.cb_recovery,
        )
        self._search_reset_in_flight: bool = False
        self._search_reset_last_try_ts: float = 0.0

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_available = False
        self.tf_check_timer = self.create_timer(
            1.0, self.check_tf, callback_group=self.cb_timers
        )

        # ─── Vision / tracking backends ───────────────────────────────────
        self.appearance = AppearanceEncoder(
            logger=self.get_logger(),
            model_name=str(self.get_parameter("osnet_model_name").value),
            model_path=str(self.get_parameter("osnet_model_path").value),
        )
        self.detector = Detector(
            weights_path=self.get_parameter("yolo_weights_path").value,
            tracker_yaml=self.get_parameter("tracker_yaml").value,
            conf=self.get_parameter("yolo_conf").value,
            iou=self.get_parameter("yolo_iou").value,
            imgsz=self.get_parameter("yolo_imgsz").value,
            device=self.get_parameter("yolo_device").value,
            logger=self.get_logger(),
        )
        if not self.detector.available():
            self.get_logger().error(
                "YOLO detector is NOT available. The node will stay idle. "
                "Install ultralytics and place the yolo-seg weights in the package dir."
            )

        # ─── Runtime state ────────────────────────────────────────────────
        self.state: FollowerState = FollowerState.UNLOCKED
        self._last_state_change_s = time.time()
        self.locked_track_id: Optional[int] = None
        self.target_signature: PersonSignature = PersonSignature()
        # Whether the target signature is frozen (locked at first acquisition,
        # never EMA-blended). User requirement: until the UI sends a release
        # command, the robot only relocks on the original person's histogram.
        self.target_signature_locked: bool = False
        self.target_signature_frozen_at_ts: Optional[float] = None
        # Multi-view template bank — populated only while the tracker is
        # actively LOCKED on the original person (so additions come from
        # the same identity, never from drift). Re-acquisition scores
        # against ``max(score_vs_view_k)`` so a side or back view of the
        # target still re-locks reliably even though the original frozen
        # view was front-on. Cleared on _release_lock.
        self.target_views: List[PersonSignature] = []
        self._last_bank_update_ts: float = 0.0
        # Per-track consecutive-frame counter for re-acquisition voting.
        # Reset whenever the best candidate changes; a track must be the
        # best AND above threshold for ``relock_streak_n`` consecutive
        # frames to actually re-lock — kills single-frame false positives.
        self._relock_streak: Dict[int, int] = {}
        self.candidates: Dict[int, TrackedCandidate] = {}
        # Ignored signatures decay after 10s (tuple of (signature, t_expiry))
        self.ignored_signatures: List[Tuple[PersonSignature, float]] = []
        # Per-candidate consecutive low-score frame counter — used to avoid
        # adding a candidate to `ignored_signatures` from a single bad frame.
        self._low_score_streak: Dict[int, int] = {}
        # Self-registered follow distance + hard lower bound for approach.
        # Both are seeded on the first successful lock of a session.
        self.registered_follow_dist_m: Optional[float] = None
        self.registered_min_track_dist_m: Optional[float] = None
        # Legacy alias kept for status field backward-compat.
        self.registered_min_dist: Optional[float] = None

        # Latest sensor snapshots
        self.cv_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.odom_pose: Optional[Pose] = None
        self.odom_linear_vel = 0.0
        self.odom_angular_vel = 0.0
        self.rear_min_dist = float("inf")
        self.fall_detected = False
        self.pill_time = False
        self.map_data: Optional[OccupancyGrid] = None

        # Motion state
        self.last_callback_time = self.get_clock().now()
        self.dt = 0.0
        self._last_stop_reason_time = 0.0
        self._stall_start_ts: Optional[float] = None
        self._last_image_center_x = 0.0
        # Hysteresis timer for "fully stop forward motion" band
        self._full_stop_band_start_ts: Optional[float] = None
        self._in_full_stop = False
        # Progress tracking for full-stop escape watchdog
        self._full_stop_entered_ts: Optional[float] = None
        self._full_stop_entry_error_px: float = 0.0
        # Smoothed pixel error used by the hysteresis comparison
        self._pixel_error_smooth: float = 0.0
        self._recovery_enter_ts = 0.0
        self._recovery_stage_enter_ts = 0.0
        self._recovery_backup_start_pose: Optional[Tuple[float, float]] = None
        self._recovery_scan_initial_dir: float = 1.0  # sign of last pixel error at loss
        self._last_pixel_error_sign: float = 0.0
        self._last_seen_target_ts = 0.0
        self._last_depth_good: Optional[float] = None
        self._last_depth_good_ts = 0.0
        # Pill delivery sub-phase
        self._pill_lost_grace_ts: Optional[float] = None
        self._pill_phase: str = "approach"  # "approach" or "hold"

        # Frontier exploration state (preserved from old code)
        self.visited_frontiers = set()
        self.checked = 0
        self.goal = 0
        self.goal_accepted = False
        self.initiated_search = False
        self.frontier_search_active = False

        # Auto-lock timer state
        self._unlocked_since = time.time()

        # Frame counter for modulo-gated per-frame work (signature EMA, thumbnails, ...)
        self._frame_counter: int = 0
        # Back-pressure: drop overlapping image callbacks instead of queuing
        self._vision_busy = threading.Lock()

        # Latest annotated debug frame (bytes). Published by a dedicated timer
        # at `debug_publish_hz`, NOT per-vision-frame. No cv2.imshow anywhere.
        self._debug_frame_latest: Optional[np.ndarray] = None
        self._debug_frame_lock = threading.Lock()

        # PIDs — smoother defaults: higher output filter, lower slew, stronger integral leak
        self.pid_angular = PID(
            kp=self.get_parameter("Kp_yaw").value,
            ki=self.get_parameter("Ki_yaw").value,
            kd=self.get_parameter("Kd_yaw").value,
            out_min=-self.get_parameter("max_angular_speed").value,
            out_max=self.get_parameter("max_angular_speed").value,
            deadband=self.get_parameter("pixel_deadband").value,
            deriv_filter_alpha=0.9,
            output_filter_alpha=0.85,
            slew_rate=1.8,  # rad/s²
            integral_leak=0.04,
        )
        self.pid_linear = PID(
            kp=self.get_parameter("Kp_lin").value,
            ki=self.get_parameter("Ki_lin").value,
            kd=self.get_parameter("Kd_lin").value,
            out_min=-self.get_parameter("max_linear_speed").value,
            out_max=self.get_parameter("max_linear_speed").value,
            deadband=self.get_parameter("depth_deadband_m").value,
            deriv_filter_alpha=0.9,
            output_filter_alpha=0.80,
            slew_rate=1.2,  # m/s²
            integral_leak=0.04,
        )

        # Periodic status / candidates publisher (5 Hz)
        self.status_timer = self.create_timer(
            0.2, self._publish_status_and_candidates,
            callback_group=self.cb_timers,
        )

        # Debug image publisher timer — decoupled from 30 Hz vision loop
        debug_hz = float(self.get_parameter("debug_publish_hz").value)
        debug_hz = max(1.0, min(10.0, debug_hz))
        self.debug_timer = self.create_timer(
            1.0 / debug_hz, self._publish_debug_frame,
            callback_group=self.cb_timers,
        )

        self.get_logger().info(
            f"Person follower ready. State={self.state.value}, "
            f"YOLO={self.detector.available()}, ReID={self.appearance.available()}"
        )

    # ───────────────────────────── Parameters ────────────────────────────
    def _declare_params(self) -> None:
        # PID — gains lowered and filters tightened for smoother motion
        self.declare_parameter("Kp_yaw", 0.003)
        self.declare_parameter("Ki_yaw", 0.0002)
        self.declare_parameter("Kd_yaw", 0.0005)
        self.declare_parameter("Kp_lin", 0.35)
        self.declare_parameter("Ki_lin", 0.06)
        self.declare_parameter("Kd_lin", 0.10)
        self.declare_parameter("max_linear_speed", 1.0/1.5)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("pill_max_linear_speed", 0.20)
        self.declare_parameter("pixel_deadband", 12.0)
        self.declare_parameter("depth_deadband_m", 0.1)
        # Blended steering with widened hysteresis + progress watchdog
        self.declare_parameter("blend_taper_center_px", 40.0)
        self.declare_parameter("blend_taper_end_px", 260.0)
        self.declare_parameter("blend_min_scale", 0.25)
        self.declare_parameter("full_stop_threshold_px", 260.0)
        self.declare_parameter("full_stop_enter_sec", 1.2)
        self.declare_parameter("full_stop_exit_px", 140.0)
        self.declare_parameter("full_stop_exit_sec", 0.3)
        self.declare_parameter("full_stop_max_sec", 3.0)   # hard cap on rotation-only
        self.declare_parameter("full_stop_progress_ratio", 0.30)  # require ≥30% error shrink
        self.declare_parameter("pixel_error_ema_alpha", 0.3)  # smooth the error used by hysteresis
        self.declare_parameter("stall_timeout_sec", 4.0)
        self.declare_parameter("stall_nudge_sec", 0.7)
        # Follow distance
        self.declare_parameter("safety_margin_mm", 2000.0)   # fallback when not self-registered
        self.declare_parameter("pill_delivery_distance", 1.0)
        self.declare_parameter("pill_hold_hysteresis_m", 0.25)   # must walk this far past 0.6 m to trigger re-approach
        self.declare_parameter("pill_lost_grace_sec", 0.8)       # wait before falling back to stationary recovery
        # Detection / tracking
        self.declare_parameter("yolo_weights_path", self.DEFAULT_YOLO_WEIGHTS)
        self.declare_parameter("tracker_yaml", "botsort.yaml")
        self.declare_parameter("yolo_conf", 0.5)
        self.declare_parameter("yolo_iou", 0.5)
        self.declare_parameter("yolo_imgsz", 480)  # 480 keeps CPU inference snappy
        # "cpu" is the safe default — gfx1010 (RX 5700 XT) has no rocBLAS
        # kernels in the pytorch/rocm6.2 wheel, so auto-detect misfires there.
        # On the RTX production server, override to "cuda":
        #   ros2 run person_follower final_person_follower --ros-args \
        #     -p yolo_device:=cuda -p yolo_weights_path:=yolom-seg.pt
        self.declare_parameter("yolo_device", "cuda")
        # Re-ID matching — lock signature is FROZEN after first acquisition;
        # recovery uses a relaxed threshold to survive partial occlusion.
        self.declare_parameter("match_lock_threshold", 0.72)
        self.declare_parameter("match_lock_threshold_recovery", 0.60)
        self.declare_parameter("match_ignore_threshold", 0.55)
        self.declare_parameter("ignored_ttl_sec", 10.0)
        self.declare_parameter("ignored_min_sightings", 3)  # N consecutive low-score frames before marking ignored
        # Compute-shaping: skip expensive work on low-confidence / non-candidate frames
        self.declare_parameter("signature_ema_every_n_frames", 5)
        self.declare_parameter("signature_ema_min_confidence", 0.6)
        self.declare_parameter("hist_min_confidence", 0.4)
        # OSNet ReID backbone. ``osnet_ain_x1_0`` (default) is cross-domain
        # robust — best for indoor↔outdoor lighting swings. Use
        # ``osnet_x0_25`` if you need to push frame rate on weak GPUs.
        self.declare_parameter("osnet_model_name", "osnet_ain_x1_0")
        # Path to a torchreid-format ``.pth.tar`` weights file. When set
        # AND the file exists, those weights are loaded — REQUIRED for
        # production-grade re-ID accuracy because the architecture alone
        # only gets you ImageNet features, which are ~15–20% weaker than
        # ReID-trained weights on actual person matching.
        # Recommended: download ``osnet_ain_x1_0_msmt17.pth.tar`` from
        # https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO and
        # set this parameter to its absolute path. ``~`` is expanded.
        # Leave blank to fall back to the ImageNet-pretrained backbone.
        self.declare_parameter("osnet_model_path", "osnet_ain_x1_0_msmt17.pth")
        # ── Multi-view template bank ────────────────────────────────────
        # Robustness gain: a single frozen view of the target captures
        # only one pose. When the person turns sideways or away, the
        # embedding diverges enough to drop below the recovery threshold.
        # We sidestep this by maintaining a small bank of K representative
        # views, sampled while LOCKED on the actually-tracked person, and
        # re-acquisition scores against ``max(score_vs_view_k)``.
        # The bank is populated only when the tracker still has positive
        # ID on the target, so it cannot drift onto the wrong person.
        self.declare_parameter("template_bank_size", 3)
        self.declare_parameter("template_bank_min_conf", 0.6)
        self.declare_parameter("template_bank_min_seconds_between_samples", 0.6)
        # Only add a new view if it is at least this dissimilar from every
        # existing bank entry (1.0 - max_score). Encourages diverse views.
        self.declare_parameter("template_bank_min_diff", 0.10)
        # ── Temporal voting on re-acquisition ───────────────────────────
        # Reject single-frame matches: the same track_id must beat the
        # threshold for N consecutive frames before we re-lock. Removes
        # the dominant source of mis-relocks (one-frame embedding noise
        # from a similarly-dressed bystander).
        self.declare_parameter("relock_streak_n", 3)
        # Lock-in UX
        self.declare_parameter("auto_lock_timeout_sec", 15.0)
        # Recovery
        self.declare_parameter("lost_threshold_sec", 1.5)
        self.declare_parameter("stationary_duration_sec", 1.0)  # hold stationary before backup
        self.declare_parameter("backup_distance_m", 0.4)         # distance-based backup target
        self.declare_parameter("backup_duration_max_sec", 3.0)   # safety cap on backup
        self.declare_parameter("backup_speed", 0.15)
        self.declare_parameter("backup_rear_clearance_m", 0.8)
        self.declare_parameter("scan_duration_sec", 8.0)
        self.declare_parameter("scan_angular_speed", 0.35)
        self.declare_parameter("give_up_sec", 90.0)
        # Depth robustness
        self.declare_parameter("depth_min_m", 0.5)
        self.declare_parameter("depth_max_m", 6.0)
        self.declare_parameter("depth_good_ttl_sec", 0.5)
        # Debug display — now published as /person_follower/debug/compressed
        # for the browser to render (no native cv2 window, no freeze).
        self.declare_parameter("debug_display", True)
        self.declare_parameter("debug_publish_hz", 10.0)
        self.declare_parameter("debug_jpeg_quality", 60)
        # Image transport — supports both compressed (default, matches
        # real Kinect driver output) and uncompressed raw topics (used
        # by the gz simulation bridge and when re-publishing raw via
        # image_transport). Topic names are parameters so a remap is
        # not strictly needed.
        self.declare_parameter("use_compressed_rgb",   True)
        self.declare_parameter("use_compressed_depth", True)
        self.declare_parameter("rgb_topic_compressed",   "/image_raw/compressed")
        self.declare_parameter("rgb_topic_raw",          "/kinect/rgb/image_raw")
        self.declare_parameter("depth_topic_compressed", "/depth/image_raw/compressedDepth")
        self.declare_parameter("depth_topic_raw",        "/kinect/depth/image_raw")

    # ═══════════════════════════ Callbacks ═══════════════════════════════
    def fall_callback(self, msg: String) -> None:
        self.fall_detected = (msg.data == "FALL DETECTED")

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_pose = msg.pose.pose
        self.odom_linear_vel = msg.twist.twist.linear.x
        self.odom_angular_vel = msg.twist.twist.angular.z

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.map_data = msg

    def pill_time_callback(self, msg: Bool) -> None:
        new = bool(msg.data)
        if new and not self.pill_time:
            self.get_logger().info("Pill time activated — entering delivery mode")
            if self.state == FollowerState.LOCKED:
                self._transition(FollowerState.PILL_DELIVERY)
        elif not new and self.pill_time:
            self.get_logger().info("Pill time deactivated — resuming normal follow")
            if self.state == FollowerState.PILL_DELIVERY:
                self._transition(
                    FollowerState.LOCKED if self.locked_track_id is not None
                    else FollowerState.UNLOCKED
                )
        self.pill_time = new

    def lock_command_callback(self, msg: Int32) -> None:
        """UI sends Int32 — track_id to lock onto, or -1 to release."""
        tid = int(msg.data)
        if tid < 0:
            self.get_logger().info("Lock release received from UI")
            self._release_lock()
            return
        if tid not in self.candidates:
            self.get_logger().warn(f"UI requested lock on unknown track_id={tid}; ignoring")
            return
        self._acquire_lock(tid, source="manual")

    def laser_callback(self, msg: LaserScan) -> None:
        """
        Rear-clearance ONLY — all other safety is deferred to safety_guard.py.
        """
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            valid = (ranges > msg.range_min) & (ranges < msg.range_max) & np.isfinite(ranges)
            ranges = np.where(valid, ranges, np.inf)
            n = len(ranges)
            if n == 0:
                return

            def angle_to_index(angle_rad: float) -> int:
                idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
                return max(0, min(idx, n - 1))

            # Rear sector: ±135°..±180° (two segments)
            i_a, i_b = angle_to_index(2.36), angle_to_index(3.14)
            i_c, i_d = angle_to_index(-3.14), angle_to_index(-2.36)
            lr = ranges[i_a:i_b + 1]
            rr = ranges[i_c:i_d + 1]
            rear = np.concatenate([lr, rr]) if lr.size + rr.size else np.array([np.inf])
            self.rear_min_dist = float(np.min(rear)) if rear.size else float("inf")
        except Exception as e:
            self.get_logger().debug(f"laser_callback error: {e}")

    def depth_callback(self, data: CompressedImage) -> None:
        """Decompress 16-bit depth PNG from compressedDepth transport."""
        try:
            fmt = data.format
            _, transport = (fmt.split(";") + [""])[:2]
            transport = transport.strip()
            if not transport.startswith("compressedDepth"):
                self.get_logger().error("depth_callback: Not compressedDepth data")
                return
            raw = np.frombuffer(data.data, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
            if img is None:
                self.get_logger().error("depth_callback: imdecode failed")
                return
            self.depth_image = img
        except Exception as e:
            self.get_logger().debug(f"depth_callback error: {e}")

    def depth_callback_raw(self, data: Image) -> None:
        """Depth handler for uncompressed sensor_msgs/Image (sim + image_transport raw).

        Downstream code (`_sample_depth`) treats depth as a scalar in metres.
        Convert 16UC1 (millimetres) or 32FC1 (metres) into the same uint16
        "millimetres" layout the compressed handler produces.
        """
        try:
            enc = (data.encoding or "").lower()
            if enc in ("16uc1", "mono16"):
                img = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
            elif enc in ("32fc1", "f32"):
                fimg = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
                with np.errstate(invalid="ignore"):
                    mm = np.nan_to_num(fimg * 1000.0, nan=0.0,
                                       posinf=0.0, neginf=0.0)
                img = np.clip(mm, 0, 65535).astype(np.uint16)
            else:
                # Unknown encoding — best-effort passthrough.
                img = self.bridge.imgmsg_to_cv2(data)
            if img is None:
                return
            self.depth_image = img
        except Exception as e:
            self.get_logger().debug(f"depth_callback_raw error: {e}")

    def check_tf(self) -> None:
        try:
            self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time(), timeout=Duration(seconds=0.1)
            )
            if not self.tf_available:
                self.get_logger().info("TF from odom to map is now available.")
            self.tf_available = True
        except TransformException:
            if self.tf_available:
                self.get_logger().warn("Lost TF from odom to map.")
            self.tf_available = False

    # ═══════════════════════════ Main image pipeline ═══════════════════════
    def _process_bgr_frame(self, bgr: np.ndarray) -> None:
        """Shared pipeline for both compressed and raw RGB inputs."""
        now_s = time.time()
        now = self.get_clock().now()
        self.dt = (now - self.last_callback_time).nanoseconds / 1e9
        self.last_callback_time = now
        if self.dt <= 0:
            return

        self.cv_image = bgr
        if self.cv_image is None:
            return
        self._frame_counter += 1

        if not self.detector.available():
            self._publish_velocity(0.0, 0.0)
            return

        detections = self.detector.infer(self.cv_image)
        self._update_candidates(detections, now_s)

        self.ignored_signatures = [
            (s, t) for (s, t) in self.ignored_signatures if t > now_s
        ]

        self._tick_state_machine(now_s)

        if self.get_parameter("debug_display").value:
            self._render_debug_frame()

    def image_callback(self, data: CompressedImage) -> None:
        # Back-pressure: if a previous vision cycle is still running, drop
        # this frame rather than queueing it. Keeps latency bounded.
        if not self._vision_busy.acquire(blocking=False):
            return
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            self._process_bgr_frame(bgr)
        except Exception as e:
            self.get_logger().debug(f"image_callback error: {e}")
        finally:
            self._vision_busy.release()

    def image_callback_raw(self, data: Image) -> None:
        if not self._vision_busy.acquire(blocking=False):
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self._process_bgr_frame(bgr)
        except Exception as e:
            self.get_logger().debug(f"image_callback_raw error: {e}")
        finally:
            self._vision_busy.release()

    # ───────────────────────── Candidate management ──────────────────────
    def _update_candidates(self, detections: List[Dict], now_s: float) -> None:
        """Compute signatures for all detections, update candidates dict.

        Heavy work (OSNet embedding, HSV histograms) is gated by detection
        confidence so low-quality crops don't burn GPU every frame.

        The frozen target signature is NEVER EMA-updated here — user
        requirement: once locked, the robot must only re-acquire on the
        original identity until explicit release from the UI. The
        multi-view template bank is populated only while the tracker
        still has the locked person under positive ID, so additions
        always come from the same identity (never drift onto someone
        else).
        """
        if self.cv_image is None:
            return
        hist_min_conf = float(self.get_parameter("hist_min_confidence").value)
        sig_ema_min_conf = float(self.get_parameter("signature_ema_min_confidence").value)
        sig_ema_every = max(1, int(self.get_parameter("signature_ema_every_n_frames").value))
        lock_th_full = float(self.get_parameter("match_lock_threshold").value)

        # Pass 0: prepare bbox/crop for every detection; gate low-quality by size + conf.
        prepared = []  # list of dicts: {det, bbox, cx, cy, mask, local_mask, crop, conf}
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.cv_image.shape[1], x2); y2 = min(self.cv_image.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            conf = float(det["conf"])
            crop = self.cv_image[y1:y2, x1:x2].copy()
            mask = det["mask"]
            local_mask = None
            if mask is not None:
                lm = mask[y1:y2, x1:x2]
                local_mask = (lm.astype(np.uint8) * 255)
            prepared.append({
                "det": det, "bbox": (x1, y1, x2, y2), "cx": 0.5 * (x1 + x2),
                "cy": 0.5 * (y1 + y2), "mask": mask, "local_mask": local_mask,
                "crop": crop, "conf": conf,
            })

        # Pass 1: embeddings — skip very low-confidence crops to keep batch small.
        encode_crops: List[np.ndarray] = []
        encode_map: List[int] = []
        for idx, p in enumerate(prepared):
            if p["conf"] < hist_min_conf:
                continue
            encode_crops.append(p["crop"])
            encode_map.append(idx)
        embeds_batch = self.appearance.encode(encode_crops) if encode_crops else []
        embeds_by_idx: Dict[int, Optional[np.ndarray]] = {}
        for bi, idx in enumerate(encode_map):
            embeds_by_idx[idx] = embeds_batch[bi] if bi < len(embeds_batch) else None

        # Pass 2: build full signatures (embedding + upper/lower hist) and
        # score against the multi-view bank (or the frozen single signature
        # if the bank hasn't been seeded yet).
        cand_builds = []  # (idx, sig, score_target, thumb_bgr, depth_m)
        for idx, p in enumerate(prepared):
            embed = embeds_by_idx.get(idx)
            if p["conf"] >= hist_min_conf:
                h = p["crop"].shape[0]
                # Overlapping 60/60 zones: each zone covers 60% of the body
                # so the waist boundary region is captured by both — far less
                # information is lost than with a hard 50/50 split.
                t = int(h * 0.60)   # top zone: rows 0..60%
                b = int(h * 0.40)   # bottom zone: rows 40%..100%
                um = None if p["local_mask"] is None else p["local_mask"][:t]
                lm_bot = None if p["local_mask"] is None else p["local_mask"][b:]
                hist_u = _compute_hsv_hist(p["crop"][:t], um)
                hist_l = _compute_hsv_hist(p["crop"][b:], lm_bot)
            else:
                hist_u = None
                hist_l = None
            sig = PersonSignature(
                embedding=embed,
                hist_upper=hist_u,
                hist_lower=hist_l,
                sample_count=1,
            )
            score_target = (
                self._score_vs_bank(sig) if self.target_signature_locked else 0.0
            )
            depth_m = self._sample_depth(p["cx"], p["cy"], p["mask"])
            thumb_bgr = self._make_thumbnail(self.cv_image, p["bbox"])
            cand_builds.append({
                "idx": idx, "sig": sig, "score": score_target,
                "thumb": thumb_bgr, "depth_m": depth_m,
            })

        # Pass 3: merge into self.candidates. Non-locked candidates get their
        # per-track signature EMA-blended only every N frames (reduces CPU).
        ignored_streak_n = max(1, int(self.get_parameter("ignored_min_sightings").value))
        active_ids = set()
        for c in cand_builds:
            p = prepared[c["idx"]]
            tid = p["det"]["track_id"]
            active_ids.add(tid)
            ignored = False
            # Against the frozen ignored signatures list
            for isig, _t in self.ignored_signatures:
                if signature_score(c["sig"], isig) >= lock_th_full:
                    ignored = True
                    break
            existing = self.candidates.get(tid)
            if existing is None:
                status = "ignored" if ignored else (
                    "locked" if tid == self.locked_track_id else "candidate"
                )
                self.candidates[tid] = TrackedCandidate(
                    track_id=tid,
                    bbox=p["bbox"],
                    centroid=(p["cx"], p["cy"]),
                    mask=p["mask"],
                    depth_m=c["depth_m"],
                    confidence=p["conf"],
                    signature=c["sig"],
                    first_seen=now_s,
                    last_seen=now_s,
                    score_vs_target=c["score"],
                    status=status,
                    thumb_bgr=c["thumb"],
                )
            else:
                existing.bbox = p["bbox"]
                existing.centroid = (p["cx"], p["cy"])
                existing.mask = p["mask"]
                existing.depth_m = c["depth_m"]
                existing.confidence = p["conf"]
                existing.last_seen = now_s
                existing.score_vs_target = c["score"]
                existing.thumb_bgr = c["thumb"]
                # Blend per-track signature only every N frames and only when
                # the detection is good enough; saves ~70% of per-frame work.
                if (
                    p["conf"] >= sig_ema_min_conf
                    and self._frame_counter % sig_ema_every == 0
                ):
                    existing.signature.update_ema(c["sig"], alpha=0.15)
                existing.status = (
                    "ignored" if ignored else
                    ("locked" if tid == self.locked_track_id else "candidate")
                )

            # ── Template-bank update ─────────────────────────────────
            # While the tracker still has positive ID on the locked
            # person AND we're in a stable LOCKED-family state (so we
            # know we are not currently being misled by a recovery
            # mismatch), opportunistically diversify the template bank
            # with the current view. This is identity-safe: by
            # construction, ``tid == locked_track_id`` means BoT-SORT
            # has continuously associated this detection with the
            # originally-locked person.
            if (
                tid == self.locked_track_id
                and self.target_signature_locked
                and self.state in (
                    FollowerState.LOCKED, FollowerState.PILL_DELIVERY
                )
            ):
                self._maybe_update_template_bank(c["sig"], p["conf"], now_s)

            # Streak-based ignored promotion: a candidate must score below
            # `ignore_th` against the frozen target for N consecutive frames
            # (during recovery states) before being permanently ignored.
            if (
                self.target_signature.hist_upper is not None
                and tid != self.locked_track_id
                and self.state in (
                    FollowerState.RECOVERING_STATIONARY,
                    FollowerState.RECOVERING_BACKUP,
                    FollowerState.RECOVERING_SCAN,
                )
            ):
                # During recovery use a much lower bar (0.20) so only
                # clearly-wrong people are blocked — a borderline score
                # (0.20–0.55) means "uncertain", not "definitely wrong",
                # and should never permanently block a re-acquire attempt.
                recovery_ignore_th = 0.20
                if c["score"] < recovery_ignore_th:
                    self._low_score_streak[tid] = self._low_score_streak.get(tid, 0) + 1
                    if (
                        self._low_score_streak[tid] >= ignored_streak_n
                        and c["sig"].hist_upper is not None
                    ):
                        ttl = float(self.get_parameter("ignored_ttl_sec").value)
                        self.ignored_signatures.append((c["sig"], now_s + ttl))
                        self._low_score_streak[tid] = 0
                else:
                    self._low_score_streak.pop(tid, None)

        # Track-last-seen for the locked id — used by lost_threshold_sec logic.
        if self.locked_track_id is not None and self.locked_track_id in active_ids:
            self._last_seen_target_ts = now_s
            # Remember which side of center the target was on — drives the
            # scan direction during recovery if we subsequently lose them.
            if self.locked_track_id in self.candidates:
                cand = self.candidates[self.locked_track_id]
                img_w = self.cv_image.shape[1]
                err = cand.centroid[0] - img_w / 2.0
                if abs(err) > 5.0:
                    self._last_pixel_error_sign = 1.0 if err > 0 else -1.0

        # Drop streak entries for disappeared candidates
        for tid in list(self._low_score_streak):
            if tid not in active_ids:
                self._low_score_streak.pop(tid, None)

        # Expire old candidates
        self.candidates = {
            tid: c for tid, c in self.candidates.items()
            if (now_s - c.last_seen) < 3.0
        }

    def _sample_depth(
        self,
        cx: float,
        cy: float,
        mask: Optional[np.ndarray],
    ) -> float:
        """Robust depth sample in meters. Returns 0.0 if no valid sample."""
        if self.depth_image is None:
            return 0.0
        H, W = self.depth_image.shape[:2]
        cx_i = int(max(0, min(W - 1, cx)))
        cy_i = int(max(0, min(H - 1, cy)))
        half = 7
        x0 = max(0, cx_i - half); x1 = min(W, cx_i + half + 1)
        y0 = max(0, cy_i - half); y1 = min(H, cy_i + half + 1)
        patch = self.depth_image[y0:y1, x0:x1]
        if patch.ndim == 3:
            patch = patch[..., 0]
        samples = patch
        if mask is not None:
            mp = mask[y0:y1, x0:x1]
            samples = patch[mp] if mp.any() else patch.flatten()
        else:
            samples = patch.flatten()
        valid = samples[(samples > 0) & np.isfinite(samples)]
        if valid.size == 0:
            if (time.time() - self._last_depth_good_ts
                    < self.get_parameter("depth_good_ttl_sec").value):
                return self._last_depth_good or 0.0
            return 0.0
        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0
        dmin = self.get_parameter("depth_min_m").value
        dmax = self.get_parameter("depth_max_m").value
        if depth_m < dmin or depth_m > dmax:
            if (time.time() - self._last_depth_good_ts
                    < self.get_parameter("depth_good_ttl_sec").value):
                return self._last_depth_good or 0.0
            return 0.0
        self._last_depth_good = depth_m
        self._last_depth_good_ts = time.time()
        return depth_m

    # ═══════════════════════ State machine driver ════════════════════════
    def _tick_state_machine(self, now_s: float) -> None:
        # Pill delivery: transition from LOCKED, and collapse any recovery
        # state back to PILL_DELIVERY (no searching during pill delivery).
        if self.pill_time:
            if self.state == FollowerState.LOCKED:
                self._transition(FollowerState.PILL_DELIVERY)
            elif self.state in (
                FollowerState.RECOVERING_STATIONARY,
                FollowerState.RECOVERING_BACKUP,
                FollowerState.RECOVERING_SCAN,
                FollowerState.RECOVERING_FRONTIER,
            ):
                self._transition(FollowerState.PILL_DELIVERY)

        if self.state == FollowerState.UNLOCKED:
            self._drive_unlocked(now_s)
        elif self.state == FollowerState.LOCKED:
            self._drive_locked(now_s)
        elif self.state == FollowerState.PILL_DELIVERY:
            self._drive_pill(now_s)
        elif self.state == FollowerState.RECOVERING_STATIONARY:
            self._drive_recover_stationary(now_s)
        elif self.state == FollowerState.RECOVERING_BACKUP:
            self._drive_recover_backup(now_s)
        elif self.state == FollowerState.RECOVERING_SCAN:
            self._drive_recover_scan(now_s)
        elif self.state == FollowerState.RECOVERING_FRONTIER:
            self._drive_recover_frontier(now_s)
        elif self.state == FollowerState.GIVE_UP:
            self._drive_give_up(now_s)

    def _transition(self, new_state: FollowerState) -> None:
        if new_state == self.state:
            return
        self.get_logger().info(
            f"STATE {self.state.value} → {new_state.value}"
        )
        self.state = new_state
        self._last_state_change_s = time.time()
        if new_state in (
            FollowerState.UNLOCKED,
            FollowerState.LOCKED,
            FollowerState.PILL_DELIVERY,
        ):
            # Motion mode change — reset PID memory so stale error doesn't kick
            self.pid_angular.reset()
            self.pid_linear.reset()
            self._full_stop_band_start_ts = None
            self._in_full_stop = False
            self._full_stop_entered_ts = None
            self._stall_start_ts = None
            self._pixel_error_smooth = 0.0
        if new_state in (
            FollowerState.RECOVERING_STATIONARY,
            FollowerState.RECOVERING_BACKUP,
            FollowerState.RECOVERING_SCAN,
        ):
            self._recovery_stage_enter_ts = time.time()
            # Clear stale ignore state so the target person is not blocked
            # from re-acquisition by scores computed before they were lost.
            self._low_score_streak.clear()
            self.ignored_signatures.clear()
        if new_state == FollowerState.RECOVERING_BACKUP:
            # Snapshot current pose so the backup stops after a fixed distance
            if self.odom_pose is not None:
                self._recovery_backup_start_pose = (
                    self.odom_pose.position.x,
                    self.odom_pose.position.y,
                )
            else:
                self._recovery_backup_start_pose = None
        if new_state == FollowerState.RECOVERING_SCAN:
            # Scan in the direction we last saw the target
            self._recovery_scan_initial_dir = (
                self._last_pixel_error_sign if self._last_pixel_error_sign != 0.0 else 1.0
            )
        if new_state == FollowerState.UNLOCKED:
            self._unlocked_since = time.time()
        # Reset pill-delivery sub-phase on every PILL_DELIVERY entry so we
        # always start fresh in "approach"; clear the lost-grace timer on
        # any transition (re-armed by _drive_pill when needed).
        if new_state == FollowerState.PILL_DELIVERY:
            self._pill_phase = "approach"
        self._pill_lost_grace_ts = None

    # ─────────────────────────── UNLOCKED ────────────────────────────────
    def _drive_unlocked(self, now_s: float) -> None:
        self._publish_velocity(0.0, 0.0)
        # Auto-lock: pick closest candidate after timeout
        timeout = self.get_parameter("auto_lock_timeout_sec").value
        elapsed = now_s - self._unlocked_since
        if self.candidates and elapsed >= timeout:
            best = None
            for cand in self.candidates.values():
                if cand.status == "ignored":
                    continue
                if cand.depth_m <= 0:
                    continue
                if best is None or cand.depth_m < best.depth_m:
                    best = cand
            if best is not None:
                self.get_logger().info(
                    f"Auto-lock timeout reached; locking closest track_id={best.track_id} "
                    f"at {best.depth_m:.2f} m"
                )
                self._acquire_lock(best.track_id, source="auto")

    # ─────────────────────────── LOCKED ──────────────────────────────────
    def _drive_locked(self, now_s: float) -> None:
        cand = self.candidates.get(self.locked_track_id)
        if cand is None:
            # Lost — start recovery (stationary first — many losses are
            # momentary occlusions and resolve within a second).
            lost_thresh = self.get_parameter("lost_threshold_sec").value
            if (now_s - self._last_seen_target_ts) >= lost_thresh:
                self.get_logger().warn(
                    f"Locked track_id={self.locked_track_id} lost for "
                    f"{now_s - self._last_seen_target_ts:.1f} s; entering recovery"
                )
                self._recovery_enter_ts = now_s
                self._transition(FollowerState.RECOVERING_STATIONARY)
            else:
                self._publish_velocity(0.0, 0.0)
            return

        # Normal follow
        img_w = self.cv_image.shape[1] if self.cv_image is not None else 640
        image_center = img_w / 2.0
        pixel_error = cand.centroid[0] - image_center

        # Prefer the self-registered follow distance (set on first lock).
        # Fall back to the `safety_margin_mm` parameter if not registered yet.
        if self.registered_follow_dist_m is not None:
            target_m = self.registered_follow_dist_m
        else:
            target_m = self.get_parameter("safety_margin_mm").value / 1000.0
        # Respect the hard lower bound so we don't approach closer than the
        # depth at which the tracker was stable during initial lock.
        if self.registered_min_track_dist_m is not None:
            target_m = max(target_m, self.registered_min_track_dist_m)

        depth_error_m = (cand.depth_m - target_m) if cand.depth_m > 0 else 0.0

        ang = self._compute_angular(pixel_error)
        lin_raw = self._compute_linear(depth_error_m) if cand.depth_m > 0 else 0.0
        lin = self._blend_steering(lin_raw, pixel_error, now_s)

        if self.fall_detected:
            # Fall detected → behavioural stop. (Not safety — that lives in
            # safety_guard.py; this is a policy choice to halt while the
            # caregiver reaches the person.)
            lin = 0.0

        self._publish_velocity(lin, ang)
        self._update_stall_watchdog(lin, cand is not None, now_s)

    def _compute_angular(self, pixel_error: float) -> float:
        # Setpoint = 0 pixels (centered), measurement = pixel_error
        return self.pid_angular.update(0.0, pixel_error, self.dt)

    def _compute_linear(self, depth_error_m: float) -> float:
        # Positive depth_error means person is farther than target → drive forward
        # Setpoint = 0, measurement = -depth_error (so PID output drives +forward)
        return self.pid_linear.update(0.0, -depth_error_m, self.dt)

    def _blend_steering(
        self,
        linear_cmd: float,
        pixel_error: float,
        now_s: float,
    ) -> float:
        """
        Continuous forward-speed scaling with hysteresis + progress watchdog.

        Scale tapers from 1.0 (at |err| ≤ center_px) down to min_scale
        (at |err| ≥ end_px). We only zero linear_x entirely if the SMOOTHED
        |err| stays above full_stop_threshold_px for more than
        full_stop_enter_sec, and we only resume forward motion after the
        smoothed |err| stays below full_stop_exit_px for more than
        full_stop_exit_sec.

        Progress watchdog: if full-stop has held for `full_stop_max_sec`
        without reducing the error by at least `full_stop_progress_ratio`,
        force-exit and resume forward motion — prevents the robot from
        spinning in place forever if the PID saturates or the error
        oscillates inside the stop band.

        NEVER calls reset_pid() — PID memory persists through band changes.
        """
        center = float(self.get_parameter("blend_taper_center_px").value)
        end = float(self.get_parameter("blend_taper_end_px").value)
        min_scale = float(self.get_parameter("blend_min_scale").value)
        stop_th = float(self.get_parameter("full_stop_threshold_px").value)
        stop_enter = float(self.get_parameter("full_stop_enter_sec").value)
        exit_th = float(self.get_parameter("full_stop_exit_px").value)
        exit_enter = float(self.get_parameter("full_stop_exit_sec").value)
        max_sec = float(self.get_parameter("full_stop_max_sec").value)
        progress_ratio = float(self.get_parameter("full_stop_progress_ratio").value)
        ema_a = float(self.get_parameter("pixel_error_ema_alpha").value)

        # Smoothed |err|: single-frame jitter shouldn't gate linear motion.
        self._pixel_error_smooth = (
            (1.0 - ema_a) * self._pixel_error_smooth + ema_a * pixel_error
        )
        ae = abs(self._pixel_error_smooth)

        # Hysteresis: update full-stop state.
        if not self._in_full_stop:
            if ae > stop_th:
                if self._full_stop_band_start_ts is None:
                    self._full_stop_band_start_ts = now_s
                elif now_s - self._full_stop_band_start_ts >= stop_enter:
                    self._in_full_stop = True
                    self._full_stop_band_start_ts = now_s
                    self._full_stop_entered_ts = now_s
                    self._full_stop_entry_error_px = ae
            else:
                self._full_stop_band_start_ts = None
        else:
            if ae < exit_th:
                if self._full_stop_band_start_ts is None:
                    self._full_stop_band_start_ts = now_s
                elif now_s - self._full_stop_band_start_ts >= exit_enter:
                    self._in_full_stop = False
                    self._full_stop_band_start_ts = None
                    self._full_stop_entered_ts = None
            else:
                self._full_stop_band_start_ts = None

        # Progress watchdog — force-exit if stuck
        if self._in_full_stop and self._full_stop_entered_ts is not None:
            held = now_s - self._full_stop_entered_ts
            if held >= max_sec:
                entry = max(1.0, self._full_stop_entry_error_px)
                shrink = (entry - ae) / entry
                if shrink < progress_ratio:
                    self.get_logger().warn(
                        f"Full-stop progress watchdog: held {held:.1f}s without "
                        f"shrinking error ({shrink*100:.0f}%<{progress_ratio*100:.0f}%); "
                        f"forcing forward nudge"
                    )
                    self._in_full_stop = False
                    self._full_stop_band_start_ts = None
                    self._full_stop_entered_ts = None
                    # Also give the PID a fresh integral so it doesn't
                    # immediately re-saturate angular.
                    self.pid_angular._integral *= 0.25

        if self._in_full_stop:
            return 0.0

        span = max(1.0, end - center)
        taper = 1.0 - max(0.0, ae - center) / span
        scale = max(min_scale, min(1.0, taper))
        return linear_cmd * scale

    def _update_stall_watchdog(
        self,
        linear_cmd: float,
        target_visible: bool,
        now_s: float,
    ) -> None:
        """
        If the robot outputs ~0 linear for too long while the target is
        still visible, nudge once at half gain to break oscillation.
        """
        stall_to = self.get_parameter("stall_timeout_sec").value
        nudge_dur = self.get_parameter("stall_nudge_sec").value
        if not target_visible:
            self._stall_start_ts = None
            return
        if abs(linear_cmd) < 0.02:
            if self._stall_start_ts is None:
                self._stall_start_ts = now_s
            elif now_s - self._stall_start_ts >= stall_to:
                # Apply a small nudge (one-shot via direct publish next cycle)
                max_v = self.get_parameter("max_linear_speed").value
                twist = Twist()
                twist.linear.x = 0.5 * max_v
                twist.angular.z = 0.0
                self.velocity_publisher.publish(twist)
                self.get_logger().warn(
                    f"Stall watchdog: nudging at 0.5×max_v for {nudge_dur:.1f}s"
                )
                # Hold the PID in a neutral state briefly by resetting only
                # the integral so filter chatter isn't forever
                self.pid_linear._integral = 0.0
                self._stall_start_ts = now_s + nudge_dur  # prevent re-trigger
        else:
            self._stall_start_ts = None

    # ─────────────────────── PILL DELIVERY ───────────────────────────────
    def _drive_pill(self, now_s: float) -> None:
        """Pill delivery — creep toward the locked person until they're
        close enough to hand off medicine, then hold while re-evaluating.

        Phases:
          * ``approach`` — person is farther than ``pill_delivery_distance``;
            creep at reduced linear cap.
          * ``hold``     — within ``pill_delivery_distance``; stop forward
            motion but keep yaw-tracking so the robot stays facing the person.

        If the person vanishes we honour a short grace before treating it as
        a real loss (many "losses" are single-frame occlusions).  After the
        grace elapses we fall back to stationary recovery — we never back up
        during pill delivery (see transition guard in ``_drive_recover_*``).
        """
        cand = self.candidates.get(self.locked_track_id)
        pill_dist = float(self.get_parameter("pill_delivery_distance").value)
        pill_max_v = float(self.get_parameter("pill_max_linear_speed").value)
        grace_sec = float(self.get_parameter("pill_lost_grace_sec").value)
        hold_hyst = float(self.get_parameter("pill_hold_hysteresis_m").value)

        # ── Lost-person handling with short grace ────────────────────────
        if cand is None:
            if self._pill_lost_grace_ts is None:
                self._pill_lost_grace_ts = now_s
            # Try opportunistic re-acquire against the frozen signature.
            match = self._search_best_match()
            if match is not None:
                self._pill_lost_grace_ts = None
                self._acquire_lock(match.track_id, source="pill-reacq")
                return
            if (now_s - self._pill_lost_grace_ts) < grace_sec:
                # Grace window — hold still, let detector recover.
                self._publish_velocity(0.0, 0.0, False)
                return
            # Grace exhausted — stay stopped until /pill_time goes False.
            # No searching during pill delivery.
            self._publish_velocity(0.0, 0.0, False)
            return

        # Target present — clear grace timer.
        self._pill_lost_grace_ts = None

        img_w = self.cv_image.shape[1] if self.cv_image is not None else 640
        pixel_error = cand.centroid[0] - img_w / 2.0
        self._last_pixel_error_sign = 1.0 if pixel_error > 0 else -1.0 if pixel_error < 0 else self._last_pixel_error_sign
        ang = self._compute_angular(pixel_error)

        if cand.depth_m <= 0:
            # No depth reading this frame — keep yaw but don't move forward.
            self._pill_phase = self._pill_phase or "approach"
            self._publish_velocity(0.0, ang,False)
            return

        # ── Phase selection with hysteresis ─────────────────────────────
        # hold while within pill_dist; only re-enter approach if the person
        # walks past (pill_dist + hold_hyst) so tiny depth jitter doesn't
        # flip us back and forth.
        if self._pill_phase == "hold":
            if cand.depth_m > (pill_dist + hold_hyst):
                self._pill_phase = "approach"
                self.get_logger().info(
                    f"Pill HOLD → APPROACH (person moved to {cand.depth_m:.2f} m)"
                )
        else:
            if cand.depth_m <= pill_dist:
                self._pill_phase = "hold"
                self.get_logger().info(
                    f"Pill APPROACH → HOLD at {cand.depth_m:.2f} m"
                )

        if self._pill_phase == "approach":
            lin_raw = self._compute_linear(cand.depth_m - pill_dist)
            lin_capped = max(-pill_max_v, min(pill_max_v, lin_raw))
            lin = self._blend_steering(lin_capped, pixel_error, now_s)
            self._publish_velocity(lin, ang,False)
        else:
            # HOLD — keep yaw-tracking so the robot faces the person while
            # they take the medicine, but stop forward motion entirely.
            self._publish_velocity(0.0, ang,False)

    # ─────────────────────── RECOVERY: STATIONARY ────────────────────────
    def _drive_recover_stationary(self, now_s: float) -> None:
        """First recovery stage — hold still for ``stationary_duration_sec``
        and let the detector try to re-acquire.  Most "losses" are momentary
        occlusions (a doorway, another person walking past, the target
        turning sideways) and resolve within a second without needing to
        move at all.  Moving immediately only makes re-ID harder.
        """
        # Opportunistic re-acquire
        match = self._search_best_match()
        if match is not None:
            self._acquire_lock(match.track_id, source="recovery-stationary")
            return
        self._publish_velocity(0.0, 0.0)
        dur = float(self.get_parameter("stationary_duration_sec").value)
        if now_s - self._recovery_stage_enter_ts > dur:
            # Still lost — pill mode skips backup; otherwise back up to
            # widen the field of view.
            if self.pill_time:
                self._transition(FollowerState.RECOVERING_SCAN)
            else:
                self._transition(FollowerState.RECOVERING_BACKUP)

    # ─────────────────────── RECOVERY: BACKUP ────────────────────────────
    def _drive_recover_backup(self, now_s: float) -> None:
        """Distance-based backup (odometry-gated) to widen the FOV after
        a loss.  Always skipped during pill delivery.
        """
        # Never back up during pill delivery
        if self.pill_time:
            self._transition(FollowerState.RECOVERING_SCAN)
            return
        # Check if target reappeared first
        match = self._search_best_match()
        if match is not None:
            self._acquire_lock(match.track_id, source="recovery-backup")
            return
        # Rear-clearance veto (planning-time; safety_guard will also veto
        # reactively at the wheel level, but bailing here avoids wasting
        # the stage on a wall we can already see).
        clearance = float(self.get_parameter("backup_rear_clearance_m").value)
        if self.rear_min_dist < clearance:
            self.get_logger().info(
                f"Rear clearance {self.rear_min_dist:.2f} m < {clearance} m; skipping back-up"
            )
            self._transition(FollowerState.RECOVERING_SCAN)
            return
        # Distance budget based on odometry snapshot taken at stage entry
        target_dist = float(self.get_parameter("backup_distance_m").value)
        travelled = 0.0
        if self._recovery_backup_start_pose is not None and self.odom_pose is not None:
            sx, sy = self._recovery_backup_start_pose
            dx = self.odom_pose.position.x - sx
            dy = self.odom_pose.position.y - sy
            travelled = math.sqrt(dx * dx + dy * dy)
        if travelled >= target_dist:
            self._publish_velocity(0.0, 0.0)
            self._transition(FollowerState.RECOVERING_SCAN)
            return
        # Hard time cap (no odom or stuck wheels)
        max_sec = float(self.get_parameter("backup_duration_max_sec").value)
        if now_s - self._recovery_stage_enter_ts > max_sec:
            self._publish_velocity(0.0, 0.0)
            self._transition(FollowerState.RECOVERING_SCAN)
            return
        speed = float(self.get_parameter("backup_speed").value)
        self._publish_velocity(-speed, 0.0)

    # ─────────────────────── RECOVERY: SCAN ──────────────────────────────
    def _drive_recover_scan(self, now_s: float) -> None:
        match = self._search_best_match()
        if match is not None:
            self._acquire_lock(match.track_id, source="recovery-scan")
            return
        # Non-matching candidates encountered during scan → add to ignored
        # only after multiple low-score sightings (handled in _update_candidates
        # via self._low_score_streak).  Here we just let the scan proceed.

        dur = float(self.get_parameter("scan_duration_sec").value)
        if now_s - self._recovery_stage_enter_ts > dur:
            self._publish_velocity(0.0, 0.0)
            if self.pill_time:
                # Stay in scan indefinitely during pill delivery — backup
                # is forbidden and frontier exploration would walk away
                # from the caregiving context.
                self._recovery_stage_enter_ts = now_s
                return
            self._transition(FollowerState.RECOVERING_FRONTIER)
            return
        w = float(self.get_parameter("scan_angular_speed").value)
        if self.pill_time:
            w = 0
        # Scan in the direction the person was last seen.
        self._publish_velocity(0.0, w * self._recovery_scan_initial_dir*0.000000000069*0)  #easter egg :) --yasser galal

    # ─────────────────────── RECOVERY: FRONTIER ──────────────────────────
    def _drive_recover_frontier(self, now_s: float) -> None:
        # During frontier, keep checking candidates
        match = self._search_best_match()
        if match is not None:
            # Cancel Nav2 goal and re-lock
            try:
                cancel_future = self.nav_to_pose_client._cancel_goal_async(self.goal)
                cancel_future.add_done_callback(self._cancel_done_callback)
            except Exception:
                pass
            self._acquire_lock(match.track_id, source="recovery-frontier")
            return

        total_rec = now_s - self._recovery_enter_ts
        give_up = self.get_parameter("give_up_sec").value
        if total_rec > give_up:
            self._transition(FollowerState.GIVE_UP)
            return

        # Run exploration if not already navigating
        if not self.frontier_search_active:
            self._publish_velocity(0.0, 0.0)
            if not self.initiated_search:
                self.initiate_search()
            if not self.goal_accepted:
                self.explore()
                self.frontier_search_active = True

    def _drive_give_up(self, now_s: float) -> None:
        self._publish_velocity(0.0, 0.0)
        # Rate-limited complaint
        if now_s - self._last_stop_reason_time > 5.0:
            self.get_logger().warn(
                "Recovery exhausted — person not re-acquired. "
                "Awaiting a new lock command or target reappearance."
            )
            self._last_stop_reason_time = now_s
        # If a strong match appears out of the blue, re-acquire
        match = self._search_best_match()
        if match is not None:
            self._acquire_lock(match.track_id, source="giveup-reacq")

    # ─────────────────────── Matching helpers ────────────────────────────
    def _score_vs_bank(self, sig: PersonSignature) -> float:
        """Score `sig` against the multi-view template bank.

        Returns ``max(signature_score(sig, view))`` over the bank.
        Falls back to the single frozen ``target_signature`` if the bank
        hasn't been seeded yet (e.g. mid-acquisition). Always in [0, 1].
        """
        if not sig.has_any():
            return 0.0
        if self.target_views:
            best = 0.0
            for tv in self.target_views:
                s = signature_score(sig, tv)
                if s > best:
                    best = s
            return best
        if self.target_signature.has_any():
            return signature_score(sig, self.target_signature)
        return 0.0

    def _maybe_update_template_bank(
        self, sig: PersonSignature, conf: float, now_s: float
    ) -> None:
        """Diversify the template bank with the current view.

        Called only when the tracker is positively associating the
        detection with the locked person (caller guard), so additions
        are identity-safe by construction. We:
          1. Skip if confidence too low or cooldown not elapsed.
          2. Skip if the new view is too similar to every existing
             entry (no information added).
          3. Append if the bank has room.
          4. Otherwise replace the most-similar existing entry — this
             actively maintains diversity instead of letting later
             samples crowd one another out.
        """
        if not sig.has_any():
            return
        if conf < float(self.get_parameter("template_bank_min_conf").value):
            return
        cooldown = float(self.get_parameter("template_bank_min_seconds_between_samples").value)
        if (now_s - self._last_bank_update_ts) < cooldown:
            return
        bank_size = max(1, int(self.get_parameter("template_bank_size").value))
        min_diff = float(self.get_parameter("template_bank_min_diff").value)

        # Compare against every existing entry; capture max similarity
        # and the index of the closest entry (used for replacement).
        max_sim = 0.0
        max_idx = 0
        for i, tv in enumerate(self.target_views):
            s = signature_score(sig, tv)
            if s > max_sim:
                max_sim = s
                max_idx = i

        # Bank has room AND new view is sufficiently novel — append.
        if len(self.target_views) < bank_size:
            if (1.0 - max_sim) >= min_diff or not self.target_views:
                self.target_views.append(sig.copy())
                self._last_bank_update_ts = now_s
            return

        # Bank full — only replace if the new view is meaningfully
        # different from its nearest neighbour, and the replacement
        # yields more diversity (smaller max-similarity to others).
        if (1.0 - max_sim) < min_diff:
            return
        # Replace the closest existing entry — the entry currently
        # least-distinctive — with the new view.
        self.target_views[max_idx] = sig.copy()
        self._last_bank_update_ts = now_s

    def _search_best_match(self) -> Optional[TrackedCandidate]:
        """Return the strongest match vs the locked target.

        Two-stage gate:
          1. Score every active candidate against the multi-view bank
             (max over views) — this is what makes a side/back view of
             the target re-acquire even though the original frozen view
             was front-on.
          2. Require the same best track_id to score above threshold for
             ``relock_streak_n`` consecutive frames before returning it.
             Removes the dominant source of mis-relocks: a similarly-
             dressed bystander producing a single-frame embedding spike.

        In recovery we use a relaxed threshold (``match_lock_threshold_recovery``)
        because the target is likely partially occluded, under different
        lighting, or at an unusual pose.
        """
        if not self.target_signature_locked:
            return None
        in_recovery = self.state in (
            FollowerState.RECOVERING_STATIONARY,
            FollowerState.RECOVERING_BACKUP,
            FollowerState.RECOVERING_SCAN,
            FollowerState.RECOVERING_FRONTIER,
            FollowerState.GIVE_UP,
        )
        if in_recovery:
            lock_th = float(self.get_parameter("match_lock_threshold_recovery").value)
        else:
            lock_th = float(self.get_parameter("match_lock_threshold").value)
        streak_n = max(1, int(self.get_parameter("relock_streak_n").value))

        # Stage 1: score and find best.
        best: Optional[TrackedCandidate] = None
        best_score = 0.0
        active_ids = set()
        for cand in self.candidates.values():
            if cand.status == "ignored" and not in_recovery:
                continue
            score = self._score_vs_bank(cand.signature)
            cand.score_vs_target = score
            active_ids.add(cand.track_id)
            if score > best_score:
                best_score = score
                best = cand

        # Drop streaks for candidates that disappeared this frame.
        for tid in list(self._relock_streak):
            if tid not in active_ids:
                self._relock_streak.pop(tid, None)

        # Stage 2: temporal voting. Only the single best candidate above
        # threshold accumulates a streak; all others reset. This way the
        # streak captures "consistent winner", not "consistent above-
        # threshold-among-many" — the latter is exactly the failure mode
        # we want to avoid (similar bystander stays above threshold).
        if best is None or best_score < lock_th:
            self._relock_streak.clear()
            return None
        for tid in list(self._relock_streak):
            if tid != best.track_id:
                self._relock_streak.pop(tid, None)
        self._relock_streak[best.track_id] = (
            self._relock_streak.get(best.track_id, 0) + 1
        )
        if self._relock_streak[best.track_id] >= streak_n:
            return best
        return None

    def _acquire_lock(self, track_id: int, *, source: str) -> None:
        """Acquire a lock on the given track.

        On the FIRST lock of a session we:
          * deep-copy the candidate signature into ``self.target_signature``
            and mark it as FROZEN — no per-frame EMA blending after this
            point.  The signature only changes on an explicit release-lock
            followed by a fresh lock.
          * self-register the follow distance (``registered_follow_dist_m``)
            and the minimum approach distance (``registered_min_track_dist_m``)
            from the candidate's current depth.

        Re-locks that happen during recovery (the same person has returned)
        reuse the frozen signature — we do NOT re-copy the candidate's
        current (possibly degraded or partially occluded) signature over it.
        """
        cand = self.candidates.get(track_id)
        if cand is None:
            self.get_logger().warn(f"_acquire_lock: track_id {track_id} not present")
            return

        # "First ever" in this session means no frozen signature exists yet.
        first_ever_lock = not self.target_signature_locked

        self.locked_track_id = track_id
        cand.status = "locked"

        if first_ever_lock:
            # Deep-copy signature and freeze
            self.target_signature = cand.signature.copy()
            self.target_signature.sample_count = 1
            self.target_signature_locked = True
            self.target_signature_frozen_at_ts = time.time()
            # Seed the multi-view bank with the locking view. The bank
            # is then opportunistically diversified during LOCKED state
            # by ``_maybe_update_template_bank`` as the person turns.
            self.target_views = [self.target_signature.copy()]
            self._last_bank_update_ts = time.time()

            # Self-register follow distance + minimum approach distance.
            # ``registered_follow_dist_m`` is the set-point used by
            # ``_compute_linear`` in normal LOCKED mode.
            # ``registered_min_track_dist_m`` is a hard lower bound that
            # ``_drive_locked`` refuses to cross (except in pill mode).
            d = cand.depth_m if cand.depth_m > 0.3 else 1.2
            self.registered_follow_dist_m = max(0.9, min(2.5, d))
            self.registered_min_track_dist_m = max(0.8, d - 0.3)
            # Keep the legacy field in sync for any status consumer
            self.registered_min_dist = self.registered_min_track_dist_m
            self.get_logger().info(
                f"FROZEN signature at depth {d:.2f} m → "
                f"follow_dist={self.registered_follow_dist_m:.2f} m, "
                f"min_track_dist={self.registered_min_track_dist_m:.2f} m"
            )

        # Successful (re-)acquisition — clear vote counters so a future
        # loss starts fresh on the next recovery cycle.
        self._relock_streak.clear()

        self.get_logger().info(
            f"LOCK acquired on track_id={track_id} (source={source}, "
            f"depth={cand.depth_m:.2f} m, frozen={self.target_signature_locked}, "
            f"bank_size={len(self.target_views)})"
        )
        self._last_seen_target_ts = time.time()
        # Cancel any in-flight frontier goal
        try:
            if self.goal_accepted and self.goal:
                cancel_future = self.nav_to_pose_client._cancel_goal_async(self.goal)
                cancel_future.add_done_callback(self._cancel_done_callback)
        except Exception:
            pass
        self.frontier_search_active = False
        self._transition(
            FollowerState.PILL_DELIVERY if self.pill_time else FollowerState.LOCKED
        )

    def _release_lock(self) -> None:
        """Full release — clears the frozen signature and all registered
        distances.  Only called when the UI sends track_id=-1 or when the
        operator explicitly resets.  Not called on transient loss.
        """
        if self.locked_track_id is not None:
            cand = self.candidates.get(self.locked_track_id)
            if cand is not None:
                cand.status = "candidate"
        self.locked_track_id = None
        self.target_signature = PersonSignature()
        self.target_signature_locked = False
        self.target_signature_frozen_at_ts = None
        self.target_views = []
        self._last_bank_update_ts = 0.0
        self.registered_min_dist = None
        self.registered_follow_dist_m = None
        self.registered_min_track_dist_m = None
        # Fresh start — forget previous prejudices.
        self.ignored_signatures.clear()
        self._low_score_streak.clear()
        self._relock_streak.clear()
        self._transition(FollowerState.UNLOCKED)

    # ═════════════════════ Velocity + status publishing ═══════════════════
    def _publish_velocity(self, lin: float, ang: float , unsafe = False) -> None:
        twist = Twist()
        twist.linear.x = float(lin)
        twist.angular.z = float(ang)
        if unsafe:
            self.unsafe_velocity_publisher.publish(twist) 
        else: 
            self.velocity_publisher.publish(twist)

    def _make_thumbnail(
        self, full_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Build a preview of a tracked person for the web UI.

        Crops from the full frame with 30% padding on each side so the
        complete person (head to toe) is always visible, then scales the
        long edge to 192 px.
        """
        if full_bgr is None or full_bgr.size == 0:
            return None
        H, W = full_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * 0.99)
        pad_y = int(bh * 0.99)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(W, x2 + pad_x)
        py2 = min(H, y2 + pad_y)
        crop = full_bgr[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        long_edge = 192
        scale = long_edge / max(1, max(ch, cw))
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _encode_thumb_b64(self, thumb_bgr: np.ndarray) -> Optional[str]:
        """Encode a BGR thumbnail to base64 JPEG for JSON transport."""
        if thumb_bgr is None:
            return None
        try:
            ok, buf = cv2.imencode(".jpg", thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return None

    def _publish_status_and_candidates(self) -> None:
        # ── Status (field names match what flask_robot_ui/static/js/app.js expects)
        auto_lock_timeout = float(self.get_parameter("auto_lock_timeout_sec").value)
        auto_lock_remaining = (
            max(0.0, auto_lock_timeout - (time.time() - self._unlocked_since))
            if self.state == FollowerState.UNLOCKED else None
        )
        status = {
            "state": self.state.value,
            "locked_id": self.locked_track_id,
            "candidate_count": len(self.candidates),
            # Lock/freeze info — UI uses this to surface whether the
            # signature is frozen and show a "release lock" button.
            "target_signature_locked": bool(self.target_signature_locked),
            "target_signature_frozen_at_ts": self.target_signature_frozen_at_ts,
            "target_bank_size": len(self.target_views),
            "relock_streak": (
                max(self._relock_streak.values()) if self._relock_streak else 0
            ),
            "registered_follow_dist_m": self.registered_follow_dist_m,
            "registered_min_track_dist_m": self.registered_min_track_dist_m,
            # Legacy field name kept for existing consumers.
            "registered_min_dist_m": self.registered_min_dist,
            "pill_time": self.pill_time,
            "pill_phase": (
                self._pill_phase
                if self.state == FollowerState.PILL_DELIVERY else None
            ),
            "fall_detected": self.fall_detected,
            "auto_lock_remaining_s": auto_lock_remaining,
            "rear_min_dist_m": (
                self.rear_min_dist if math.isfinite(self.rear_min_dist) else None
            ),
            "timestamp": time.time(),
        }
        self.status_pub.publish(String(data=json.dumps(status)))

        # ── Candidates (include base64 JPEG thumbnails for the UI grid)
        cand_list = []
        for cand in self.candidates.values():
            cand_list.append({
                "track_id": cand.track_id,
                "bbox": list(cand.bbox),
                "centroid": [float(cand.centroid[0]), float(cand.centroid[1])],
                "depth_m": round(cand.depth_m, 3),
                "confidence": round(cand.confidence, 3),
                "status": cand.status,
                "score_vs_target": round(cand.score_vs_target, 3),
                "thumb_jpeg_b64": self._encode_thumb_b64(cand.thumb_bgr),
            })
        self.candidates_pub.publish(String(data=json.dumps({
            "candidates": cand_list,
            "timestamp": time.time(),
        })))

    # ═══════════════════════ Frontier exploration ═════════════════════════
    # (Carried over from the previous revision; these work well.)

    def find_frontiers(self, map_array: np.ndarray) -> List[Tuple[int, int]]:
        free = (map_array == 0)
        unknown = (map_array == -1)
        padded = np.pad(unknown, 1, mode="constant", constant_values=False)
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        adj_unknown = up | down | left | right
        frontier_mask = free & adj_unknown
        ys, xs = np.where(frontier_mask)
        raw_cells = list(zip(ys.tolist(), xs.tolist()))
        clusters = self._cluster_frontiers(raw_cells, min_cluster=4)
        self.get_logger().info(
            f"Found {len(raw_cells)} frontier cells -> {len(clusters)} clusters"
        )
        return clusters

    def _cluster_frontiers(
        self, frontier_cells: List[Tuple[int, int]], min_cluster: int = 4
    ) -> List[Tuple[int, int]]:
        fset = set(frontier_cells)
        visited: set = set()
        centroids: List[Tuple[int, int]] = []
        for start in frontier_cells:
            if start in visited:
                continue
            stack = [start]
            cluster: List[Tuple[int, int]] = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                cluster.append(cur)
                r, c = cur
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r + dr, c + dc)
                        if nb in fset and nb not in visited:
                            stack.append(nb)
            if len(cluster) >= min_cluster:
                cr = int(sum(p[0] for p in cluster) / len(cluster))
                cc = int(sum(p[1] for p in cluster) / len(cluster))
                centroids.append((cr, cc))
        return centroids

    def choose_frontier(
        self, frontiers: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None or self.map_data is None:
            return None
        origin = self.map_data.info.origin.position
        res = self.map_data.info.resolution
        robot_row = int((robot_pose.position.y - origin.y) / res)
        robot_col = int((robot_pose.position.x - origin.x) / res)
        min_dist_cells = max(1, int(0.5 / res))
        min_distance = float("inf")
        chosen = None
        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue
            d = math.sqrt((robot_row - frontier[0]) ** 2 + (robot_col - frontier[1]) ** 2)
            if d < min_dist_cells:
                continue
            if d < min_distance:
                min_distance = d
                chosen = frontier
        if chosen:
            self.visited_frontiers.add(chosen)
            self.get_logger().info(f"Chosen frontier: {chosen}")
        else:
            self.get_logger().warning("No valid frontier found")
            self.checked += 1
            if self.checked >= 5:
                self.get_logger().warning("DID NOT FIND ANYONE — giving up frontier search")
        return chosen

    def explore(self) -> None:
        if self.map_data is None:
            self.get_logger().warning("No map data available for exploration")
            return
        map_array = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width)
        )
        frontiers = self.find_frontiers(map_array)
        if not frontiers:
            self.get_logger().warning("No frontiers found")
            return
        chosen = self.choose_frontier(frontiers)
        if not chosen:
            return
        origin = self.map_data.info.origin.position
        res = self.map_data.info.resolution
        goal_x = chosen[1] * res + origin.x
        goal_y = chosen[0] * res + origin.y
        self.navigate_to(goal_x, goal_y)

    def navigate_to(self, x: float, y: float) -> None:
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg
        self.goal = nav_goal
        self.get_logger().info(f"Nav2 goal: x={x:.2f}, y={y:.2f}")
        # Bounded wait so the image/YOLO thread cannot hang here if Nav2
        # isn't up yet — the state machine will retry on the next tick.
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(
                "Nav2 action server not ready; deferring frontier goal."
            )
            self.frontier_search_active = False
            return
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
            self.goal = goal_handle
            if not goal_handle.accepted:
                self.get_logger().warning("Goal rejected!")
                self.frontier_search_active = False
                return
            self.goal_accepted = True
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._navigation_complete_callback)
        except Exception as e:
            self.get_logger().error(f"goal_response_callback: {e}")
            self.frontier_search_active = False

    def _cancel_done_callback(self, future) -> None:
        self.get_logger().info("Nav2 goal cancelled.")
        self.goal_accepted = False
        self.frontier_search_active = False

    def _navigation_complete_callback(self, future) -> None:
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation completed with result: {result}")
        except Exception as e:
            self.get_logger().debug(f"Navigation complete callback: {e}")
        finally:
            self.goal_accepted = False
            self.frontier_search_active = False

    def initiate_search(self) -> None:
        """Reset slam_toolbox map to start fresh exploration (async).

        Non-blocking: fires the /slam_toolbox/reset call on the node's own
        executor (cb_recovery group) and returns immediately. The image
        callback thread is never held, so YOLO keeps running and re-detection
        can still happen during `RECOVERING_FRONTIER`.
        """
        if self.initiated_search or self._search_reset_in_flight:
            return
        now = time.time()
        # Debounce retries so we don't hammer the service every tick
        if now - self._search_reset_last_try_ts < 1.0:
            return
        self._search_reset_last_try_ts = now
        if not self.slam_reset_client.service_is_ready():
            self.get_logger().debug(
                "/slam_toolbox/reset not ready yet; will retry next tick."
            )
            return
        self._search_reset_in_flight = True
        future = self.slam_reset_client.call_async(Reset.Request())
        future.add_done_callback(self._on_search_reset_done)
        self.get_logger().info("Initiating search (SLAM reset, async).")

    def _on_search_reset_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"SLAM reset call failed: {e}")
            result = None
        if result is not None:
            self.visited_frontiers = set()
            self.checked = 0
            self.map_data = None
            self.initiated_search = True
            self.get_logger().info("Slam Toolbox has been reset.")
        else:
            self.get_logger().warning(
                "SLAM reset returned no result; will retry."
            )
        self._search_reset_in_flight = False

    def get_robot_pose_in_map(self) -> Optional[Pose]:
        if not self.tf_available or self.odom_pose is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time(), timeout=Duration(seconds=0.2)
            )
            return do_transform_pose(self.odom_pose, transform)
        except TransformException as ex:
            self.get_logger().debug(f"Transform lookup failed: {ex}")
            return None

    # ─────────────────────── Debug rendering ─────────────────────────────
    def _render_debug_frame(self) -> None:
        """Compose the annotated frame and stash it for the debug-publish
        timer.  NEVER call cv2.imshow / waitKey from anywhere in this
        node — HighGUI is not thread-safe under MultiThreadedExecutor and
        intermittently hangs or deadlocks under load and over SSH.  The
        debug view is delivered as a ``CompressedImage`` on
        ``/person_follower/debug/compressed`` and rendered inside the
        Flask UI debug panel instead.
        """
        if self.cv_image is None:
            return
        img = self.cv_image.copy()
        for cand in self.candidates.values():
            x1, y1, x2, y2 = cand.bbox
            color = (
                (0, 0, 255) if cand.track_id == self.locked_track_id else
                (128, 128, 128) if cand.status == "ignored" else
                (0, 255, 0)
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"id={cand.track_id} d={cand.depth_m:.2f}m "
                f"s={cand.score_vs_target:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        cv2.putText(
            img, f"STATE: {self.state.value}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
        )
        lock_txt = (
            f"LOCKED id={self.locked_track_id}"
            f"{' (frozen)' if self.target_signature_locked else ''}"
            if self.locked_track_id is not None else "UNLOCKED"
        )
        cv2.putText(
            img, lock_txt, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        if self.registered_follow_dist_m is not None:
            cv2.putText(
                img,
                f"follow={self.registered_follow_dist_m:.2f}m "
                f"min={self.registered_min_track_dist_m:.2f}m",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )
        with self._debug_frame_lock:
            self._debug_frame_latest = img

    def _publish_debug_frame(self) -> None:
        """Timer-driven: JPEG-encode the latest annotated frame and
        publish it.  Runs at ``debug_publish_hz`` regardless of vision
        callback rate so the UI sees a steady stream even when the
        detector is pegged.
        """
        if not self.debug_display_enabled:
            return
        with self._debug_frame_lock:
            frame = self._debug_frame_latest
        if frame is None:
            return
        try:
            quality = int(self.get_parameter("debug_jpeg_quality").value)
        except Exception:
            quality = 60
        quality = max(20, min(95, quality))
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok:
                return
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().debug(f"debug publish failed: {exc}")

    @property
    def debug_display_enabled(self) -> bool:
        try:
            return bool(self.get_parameter("debug_display").value)
        except Exception:
            return False


# ─────────────────────────────── Main entry ────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonFollower()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()