import cv2
from collections import deque, defaultdict


class TrajectoryManager:
    def __init__(self, max_points=30):
        self.max_points = max_points
        self.tracks = defaultdict(lambda: deque(maxlen=self.max_points))

    def update(self, obj_id, pos):
        self.tracks[obj_id].append(pos)

    def draw(self, frame, active_ids):
        """
        Draws trajectories and removes IDs that are no longer active.
        """
        # 1. Identify IDs to remove (stored IDs not present in current frame)
        stored_ids = list(self.tracks.keys())
        for stored_id in stored_ids:
            if stored_id not in active_ids:
                del self.tracks[stored_id]

        # 2. Draw the remaining active tracks
        for obj_id, points in self.tracks.items():
            if len(points) < 2:
                continue


            # Draw lines between points
            for i in range(1, len(points)):
                # Visual logic: thickness grows as points get 'fresher'
                thickness = int(2 * (i / self.max_points) + 1)
                cv2.line(frame, points[i - 1], points[i], (0, 255, 255), thickness)