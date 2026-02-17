from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: (4,) in xyxy
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray          # xyxy float
    cls_id: int
    conf: float
    age: int = 0              # frames since created
    misses: int = 0           # consecutive frames without match
    hits: int = 1             # total matches

    def update(self, bbox: np.ndarray, cls_id: int, conf: float):
        self.bbox = bbox
        self.cls_id = cls_id
        self.conf = conf
        self.misses = 0
        self.hits += 1


class IOUTracker:
    """
    Simple multi-object tracker:
    - Greedy IoU matching
    - One instance per camera
    - Returns stable integer track IDs
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_misses: int = 20,
        min_conf: float = 0.25,
        class_aware: bool = True,
    ):
        self.iou_thresh = float(iou_thresh)
        self.max_misses = int(max_misses)
        self.min_conf = float(min_conf)
        self.class_aware = bool(class_aware)

        self._next_id = 1
        self.tracks: List[Track] = []

    def reset(self):
        self._next_id = 1
        self.tracks.clear()

    def update(
        self,
        det_xyxy: np.ndarray,
        det_conf: np.ndarray,
        det_cls: np.ndarray,
    ) -> List[Track]:
        """
        det_xyxy: (N,4) float xyxy
        det_conf: (N,) float
        det_cls:  (N,) int
        Returns: list of active Track (after update)
        """

        # Filter detections
        if det_xyxy is None or len(det_xyxy) == 0:
            det_xyxy = np.zeros((0, 4), dtype=np.float32)
            det_conf = np.zeros((0,), dtype=np.float32)
            det_cls = np.zeros((0,), dtype=np.int32)
        else:
            det_xyxy = det_xyxy.astype(np.float32)
            det_conf = det_conf.astype(np.float32)
            det_cls = det_cls.astype(np.int32)

            keep = det_conf >= self.min_conf
            det_xyxy = det_xyxy[keep]
            det_conf = det_conf[keep]
            det_cls = det_cls[keep]

        # Age existing tracks
        for t in self.tracks:
            t.age += 1

        # If no detections, just increment misses and prune
        if len(det_xyxy) == 0:
            for t in self.tracks:
                t.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
            return list(self.tracks)

        # Build IoU matrix (T x D)
        T = len(self.tracks)
        D = len(det_xyxy)

        iou_mat = np.zeros((T, D), dtype=np.float32)
        for ti, trk in enumerate(self.tracks):
            for di in range(D):
                if self.class_aware and trk.cls_id != int(det_cls[di]):
                    iou_mat[ti, di] = 0.0
                else:
                    iou_mat[ti, di] = iou_xyxy(trk.bbox, det_xyxy[di])

        # Greedy matching: repeatedly pick best IoU remaining
        matched_tracks = set()
        matched_dets = set()

        # Flatten indices by IoU descending
        candidates = np.dstack(np.unravel_index(np.argsort(-iou_mat.ravel()), (T, D)))[0]
        for ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            if iou_mat[ti, di] < self.iou_thresh:
                break
            matched_tracks.add(ti)
            matched_dets.add(di)
            self.tracks[ti].update(det_xyxy[di], int(det_cls[di]), float(det_conf[di]))

        # Unmatched tracks get a miss
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_tracks:
                trk.misses += 1

        # Create new tracks for unmatched detections
        for di in range(D):
            if di in matched_dets:
                continue
            new = Track(
                track_id=self._next_id,
                bbox=det_xyxy[di].copy(),
                cls_id=int(det_cls[di]),
                conf=float(det_conf[di]),
            )
            self._next_id += 1
            self.tracks.append(new)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return list(self.tracks)
