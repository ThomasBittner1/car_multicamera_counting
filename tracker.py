import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import List


def iou_batch(bboxes1, bboxes2):
    """ Vectorized IoU calculation for (N,4) and (M,4) """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.empty((len(bboxes1), len(bboxes2)))

    # bboxes: [x1, y1, x2, y2]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class KalmanBoxTracker:
    """
    Uses a Kalman Filter to track the state of a bounding box.
    State vector: [x, y, s, r, vx, vy, vs]
    (center x, center y, scale/area, aspect ratio, and their velocities)
    """

    def __init__(self, bbox):
        # Initialize Kalman Filter with 7 state variables and 4 observed variables
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
        ])
        # Setting uncertainties (R: measurement noise, P: covariance, Q: process noise)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        self.hits = 1  # <--- Add this line
        self.history = []

    def _bbox_to_z(self, bbox):
        """ Converts [x1,y1,x2,y2] to [cx, cy, s, r] """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array([bbox[0] + w / 2., bbox[1] + h / 2., w * h, w / float(h)]).reshape((4, 1))

    def _z_to_bbox(self, x):
        """ Converts [cx, cy, s, r] to [x1,y1,x2,y2] """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

    def predict(self):
        """ Advances the state vector and returns the predicted bounding box. """
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x)

    def update(self, bbox):
        """ Updates the state vector with observed bbox. """
        self.time_since_update = 0
        self.kf.update(self._bbox_to_z(bbox))


class AdvancedTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1

    def update(self, dets):
        """
        dets: np.array of [[x1,y1,x2,y2, score, class], ...]
        """
        self.frame_count += 1

        # 1. Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.delete(trks, to_del, axis=0)
        self.trackers = [t for i, t in enumerate(self.trackers) if i not in to_del]

        # 2. Hungarian Algorithm Matching (Linear Sum Assignment)
        iou_matrix = iou_batch(dets[:, :4], trks[:, :4])

        # We want to maximize IoU, so we minimize (-IoU)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T

        unmatched_detections = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]

        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        # 3. Update matched trackers
        for m in matches:
            # m[0, 0] is detection index, m[0, 1] is tracker index
            self.trackers[m[0, 1]].update(dets[m[0, 0], :4])
            # The .update() call now increments self.hits inside the class

        # 4. Create new trackers for unmatched detections
        for i in unmatched_detections:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.id = self.next_id  # <--- Ensure this is here
            self.next_id += 1
            self.trackers.append(trk)

        # 5. Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Change the return line in tracker.py to be more forgiving initially
        return [t for t in self.trackers if t.time_since_update <= 1 and t.hits >= 1]