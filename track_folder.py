#!/usr/bin/env python3
# vim: expandtab:ts=4:sw=4
from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.iou_matching import iou


def natural_key(string_: str):
    import re
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", string_)]


def list_images(input_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort(key=natural_key)
    return files


class ColorHistogramEncoder:
    """Simple color-histogram feature encoder as a fallback when no deep ReID model is available.

    Produces a L2-normalized concatenated HSV histogram.
    """

    def __init__(self, h_bins: int = 16, s_bins: int = 16, v_bins: int = 16):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def __call__(self, bgr_image: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        features: List[np.ndarray] = []
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        for x, y, w, h in boxes_tlwh:
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = max(int(x + w), 0)
            y2 = max(int(y + h), 0)
            x2 = min(x2, hsv.shape[1] - 1)
            y2 = min(y2, hsv.shape[0] - 1)
            if x2 <= x1 or y2 <= y1:
                # Empty region; use zeros
                feat = np.zeros(self.h_bins + self.s_bins + self.v_bins, dtype=np.float32)
                features.append(feat)
                continue
            patch = hsv[y1:y2, x1:x2]
            h_hist = cv2.calcHist([patch], [0], None, [self.h_bins], [0, 180])
            s_hist = cv2.calcHist([patch], [1], None, [self.s_bins], [0, 256])
            v_hist = cv2.calcHist([patch], [2], None, [self.v_bins], [0, 256])
            feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
            # L2 normalize
            norm = np.linalg.norm(feat) + 1e-8
            feat = (feat / norm).astype(np.float32)
            features.append(feat)
        return np.asarray(features, dtype=np.float32)


def try_create_deep_encoder(model_path: Optional[str]):
    """Attempt to create a Deep ReID encoder from a TensorFlow .pb model.

    Falls back to ColorHistogramEncoder if TF is not available or model is missing.
    """
    if model_path is None:
        return ColorHistogramEncoder()
    try:
        from tools.generate_detections import create_box_encoder
    except Exception:
        # TensorFlow likely missing in environment
        return ColorHistogramEncoder()
    if not os.path.isfile(model_path):
        return ColorHistogramEncoder()
    try:
        return create_box_encoder(model_path, batch_size=32)
    except Exception:
        return ColorHistogramEncoder()


def load_yolo(model_name_or_path: str = "yolov8n.pt"):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from exc
    return YOLO(model_name_or_path)


def xyxy_to_tlwh(xyxy: np.ndarray) -> np.ndarray:
    tlwh = xyxy.copy()
    tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]
    tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]
    tlwh[:, 0] = tlwh[:, 0]
    tlwh[:, 1] = tlwh[:, 1]
    return tlwh


def draw_tracks(frame: np.ndarray, tracker: Tracker) -> np.ndarray:
    from application_util.visualization import create_unique_color_uchar
    # If a label map is provided via closure, use it
    id_to_label: Dict[int, str] = getattr(draw_tracks, "_id_to_label", {})  # type: ignore[attr-defined]
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        x, y, w, h = track.to_tlwh().astype(np.int64)
        color = create_unique_color_uchar(track.track_id)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = id_to_label.get(track.track_id, f"ID_{track.track_id}")
        cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def process_folder(
    input_dir: str,
    output_dir: str,
    yolo_model: str,
    reid_model: Optional[str],
    conf_thres: float,
    nms_max_overlap: float,
    max_cosine_distance: float,
    nn_budget: Optional[int],
    save_video: bool,
    video_filename: Optional[str],
    fps: int,
    imgsz: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    image_paths = list_images(input_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {input_dir}")

    # Initialize detector and encoders
    model = load_yolo(yolo_model)
    encoder = try_create_deep_encoder(reid_model)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    writer: Optional[cv2.VideoWriter] = None

    # Class labeling state across frames
    names: Dict[int, str] = getattr(model, "names", {})  # YOLO class id -> name
    per_class_counts: Dict[str, int] = {}
    friendly_label: Dict[int, str] = {}  # track_id -> friendly label like person_1

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        height, width = frame.shape[:2]

        if save_video and writer is None:
            # lazy init writer from first valid frame
            if not video_filename:
                video_filename = os.path.join(output_dir, "tracks.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # YOLO inference
        results = model(frame, imgsz=imgsz, verbose=False)
        det = results[0]
        # det.boxes.xyxy: (N,4), det.boxes.conf: (N,1)
        if det.boxes is None or len(det.boxes) == 0:
            # still advance tracker to age out tracks
            tracker.predict()
            tracker.update([])
            # no detections -> draw with existing labels
            annotated = draw_tracks(frame.copy(), tracker)
            out_path = os.path.join(images_out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            if writer is not None:
                writer.write(annotated)
            continue

        xyxy = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy().reshape(-1)
        cls_ids = det.boxes.cls.cpu().numpy().astype(int).reshape(-1)

        # filter by confidence
        mask = scores >= conf_thres
        xyxy = xyxy[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if xyxy.size == 0:
            tracker.predict()
            tracker.update([])
            annotated = draw_tracks(frame.copy(), tracker)
            out_path = os.path.join(images_out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)
            if writer is not None:
                writer.write(annotated)
            continue

        tlwh = xyxy_to_tlwh(xyxy)

        # Build features
        try:
            features = encoder(frame, tlwh.copy())  # works for both encoders
        except TypeError:
            # ColorHistogramEncoder signature
            features = encoder(frame, tlwh.copy())

        # Non-max suppression (DeepSORT expects tlwh)
        indices = preprocessing.non_max_suppression(tlwh, nms_max_overlap, scores)
        tlwh = tlwh[indices]
        scores = scores[indices]
        features = np.asarray(features)[indices]
        cls_ids = cls_ids[indices]

        detections: List[Detection] = []
        for i in range(len(tlwh)):
            detections.append(Detection(tlwh[i], scores[i], features[i]))

        # Tracker step
        tracker.predict()
        tracker.update(detections)

        # Resolve class per track using IoU with current frame detections
        track_id_to_class: Dict[int, str] = {}
        if len(tlwh) > 0:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                tb = track.to_tlwh()
                j = int(np.argmax(iou(tb, tlwh)))
                cname = names.get(int(cls_ids[j]), "obj") if isinstance(names, dict) else str(int(cls_ids[j]))
                track_id_to_class[track.track_id] = cname

        # Assign stable friendly labels per class (person_1, car_1, ...)
        for tid, cname in track_id_to_class.items():
            if tid not in friendly_label:
                count = per_class_counts.get(cname, 0) + 1
                per_class_counts[cname] = count
                friendly_label[tid] = f"{cname}_{count}"

        # Provide label map to drawing function via attribute
        label_map = {t.track_id: friendly_label.get(t.track_id, f"ID_{t.track_id}")
                     for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0}
        setattr(draw_tracks, "_id_to_label", label_map)  # type: ignore[attr-defined]

        # Visualization and save
        annotated = draw_tracks(frame.copy(), tracker)
        out_path = os.path.join(images_out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated)
        if writer is not None:
            writer.write(annotated)

    if writer is not None:
        writer.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in an image folder using YOLOv8 + DeepSORT")
    parser.add_argument("--input_dir", required=True, help="Path to input images folder")
    parser.add_argument("--output_dir", required=True, help="Path to output folder (images and video)")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="Ultralytics YOLO model path or name")
    parser.add_argument("--reid_model", default=None, help="Optional TensorFlow .pb ReID model (mars-small128.pb)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms_max_overlap", type=float, default=0.7, help="NMS max overlap for detections")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2, help="DeepSORT matching threshold")
    parser.add_argument("--nn_budget", type=int, default=100, help="DeepSORT appearance gallery size (None for unlimited)")
    parser.add_argument("--save_video", action="store_true", help="Save compiled tracking video to output_dir")
    parser.add_argument("--video_filename", default=None, help="Optional output video filename (mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    return parser.parse_args()


def main():
    args = parse_args()
    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        yolo_model=args.yolo_model,
        reid_model=args.reid_model,
        conf_thres=args.conf,
        nms_max_overlap=args.nms_max_overlap,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=None if args.nn_budget <= 0 else args.nn_budget,
        save_video=args.save_video,
        video_filename=args.video_filename,
        fps=args.fps,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()

