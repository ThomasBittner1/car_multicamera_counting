import cv2
from collections import deque, defaultdict


class TrajectoryManager:
    def __init__(self, max_points=30, max_lost_frames=5):
        self.max_points = max_points
        self.max_lost_frames = max_lost_frames
        self.tracks = defaultdict(lambda: deque(maxlen=self.max_points))
        # Tracks how many consecutive frames an ID has been missing
        self.lost_counters = defaultdict(int)

    def update(self, obj_id, pos):
        """Called when YOLO sees the object."""
        self.tracks[obj_id].append(pos)
        self.lost_counters[obj_id] = 0  # Reset counter because it's seen

    def process_garbage_collection(self, active_ids):
        """
        Increments 'lost' counters for missing IDs and deletes them if
        they exceed max_lost_frames.
        """
        all_stored_ids = list(self.tracks.keys())

        for obj_id in all_stored_ids:
            if obj_id not in active_ids:
                self.lost_counters[obj_id] += 1
            else:
                self.lost_counters[obj_id] = 0

            # Delete if gone for too long
            if self.lost_counters[obj_id] > self.max_lost_frames:
                del self.tracks[obj_id]
                del self.lost_counters[obj_id]
                car_got_lost()


    def draw(self, frame):
        """
        Strictly draws. No deletion logic here.
        """
        for obj_id, points in self.tracks.items():
            if len(points) < 2:
                continue

            # Draw lines between points
            for i in range(1, len(points)):
                # Visual logic: thickness grows as points get 'fresher'
                thickness = int(2 * (i / self.max_points) + 1)
                cv2.line(frame, points[i - 1], points[i], (0, 255, 255), thickness)