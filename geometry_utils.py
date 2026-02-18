import numpy as np
import cv2
from collections import deque, defaultdict


def get_distributed_crops(crops, n=16):
    if len(crops) <= n:
        return crops

    indices = np.round(np.linspace(0, len(crops) - 1, n)).astype(int)
    distributed_crops = [crops[i] for i in indices]
    return distributed_crops


class TrajectoryManager:
    def __init__(self, max_points=30, max_lost_frames=5, on_delete_callback=None):
        self.max_points = max_points
        self.max_lost_frames = max_lost_frames
        self.tracks = {}
        self.on_delete_callback = on_delete_callback

    def update(self, obj_id, pos, cropped_box=None):
        if obj_id not in self.tracks:
            self.tracks[obj_id] = {
                "points": deque(maxlen=self.max_points),
                "crops": deque(maxlen=self.max_points),
                "lost_count": 0
            }

        track = self.tracks[obj_id]
        track["points"].append(pos)
        track["crops"].append(cropped_box)
        track["lost_count"] = 0

        num_points_for_distance = 3
        pts_count = len(track["points"])

        if pts_count >= 2:
            lookback = min(pts_count, num_points_for_distance)

            start_point = track["points"][-lookback]
            end_point = track["points"][-1]

            direction_vec = np.array(end_point, dtype='float64') - np.array(start_point, dtype='float64')
            # Normalize to unit vector
            mag = np.linalg.norm(direction_vec)
            track["mag"] = mag
            if mag > 0:
                track["direction"] = direction_vec / mag
            else:
                track["direction"] = np.array([0.0, 0.0])
        else:
            track["direction"] = np.array([0.0, 0.0])
            track["mag"] = 0.0


    def process_garbage_collection(self, active_ids, cid):
        # Use list() because we are deleting keys while iterating
        all_stored_ids = list(self.tracks.keys())

        for obj_id in all_stored_ids:
            track = self.tracks[obj_id]

            if obj_id not in active_ids:
                track["lost_count"] += 1
            else:
                track["lost_count"] = 0

            if track["lost_count"] > self.max_lost_frames:
                if self.on_delete_callback:
                    self.on_delete_callback(cid, obj_id, track["crops"])
                del self.tracks[obj_id]


    def draw(self, frame):
        for obj_id, data in self.tracks.items():
            points = data["points"]
            if len(points) < 2:
                continue
            dir = data['direction']
            cv2.putText(frame, f"dir: {dir[0]:0.3f}, {dir[1]:0.3f} (mag: {data['mag']:.3f})",
                        points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for i in range(1, len(points)):
                thickness = int(2 * (i / self.max_points) + 1)
                cv2.line(frame, points[i - 1], points[i], (0, 255, 255), thickness)




