#!/usr/bin/env bash
# Fetch the perception model weights. Everything here except the fall-detection
# model is a stock pretrained checkpoint pulled from its original source.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/person_follower/person_follower/models}"
mkdir -p "$DEST"; cd "$DEST"
echo "==> $DEST"

get() { [ -f "$2" ] && { echo "    have $2"; return; }; echo "    get  $2"; curl -fL --retry 3 -o "$2" "$1"; }

# YOLO11 segmentation (Ultralytics) - person detection + masks
get https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt yolo11m-seg.pt

# MediaPipe selfie segmentation / DeepLabV3 - used by the first prototype
get https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite selfie_segmenter.tflite
get https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite deeplabv3.tflite

cat <<'MSG'

    OSNet ReID weights (osnet_x1_0_msmt17.pth) download automatically on first
    run via torchreid. To fetch manually see:
      https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

    fall_unified_yolo26m_v1.pt is OUR trained model - grab it from the
    GitHub Release attached to this repository and drop it in this folder.
MSG
