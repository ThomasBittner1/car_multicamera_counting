import cv2

import embedding_utils
import geometry_utils
import importlib
import time
import numpy as np
from ultralytics import YOLO
import tracker

importlib.reload(embedding_utils)
importlib.reload(geometry_utils)

sources = {"c041": "AICity22_Track1_MTMC_Tracking/test/S06/c041/vdo.avi",
           "c042": "AICity22_Track1_MTMC_Tracking/test/S06/c042/vdo.avi"}

caps = {cid: cv2.VideoCapture(src) for cid, src in sources.items()}


def calculate_embeddings(cid, id, crops):
    print (f'{cid}: car {id} deleted, {len(crops)} crops')
    crops = geometry_utils.get_distributed_crops(crops)
    vector = embedder.get_embeddings(crops)
    mean_vector = np.mean(vector, axis=0)


trajectories = {cid: geometry_utils.TrajectoryManager(on_delete_callback=calculate_embeddings) for cid in caps.keys()}
trackers = {cid: tracker.IOUTracker(iou_thresh=0.3, max_misses=15, min_conf=0.25, class_aware=True)
            for cid in caps.keys()}


embedder = embedding_utils.EmbeddingGenerator()
model = YOLO("yolo11m.pt")
class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

while True:
    any_ok = False
    frames = []
    camids = []
    for cid, cap in caps.items():
        ok, frame = cap.read()
        if not ok:
            continue
        any_ok = True
        frames.append(frame)
        camids.append(cid)

    if not any_ok:
        break

    results = model.predict(source=frames, classes=list(class_names.keys()), imgsz=1088, verbose=False)

    for c,cid in enumerate(camids):
        boxes_obj = results[c].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            boxes = boxes_obj.xyxy.cpu().numpy()
            class_ids = boxes_obj.cls.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()
            active_tracks = trackers[cid].update(boxes, class_ids, confs)
        else:
            active_tracks = trackers[cid].update(None, None, None)

        for trk in active_tracks:
            if trk.misses > 0:
                continue  # don't draw stale tracks

            x1, y1, x2, y2 = map(int, trk.bbox)
            label = f"#{trk.track_id}"

            cv2.rectangle(frames[c], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frames[c], label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop = frames[c][y1:y2, x1:x2]

            trajectories[cid].update(trk.track_id, (cx, cy), crop)

        trajectories[cid].process_garbage_collection(active_ids=[trk.track_id for trk in active_tracks], cid=cid)
        trajectories[cid].draw(frames[c])

        cv2.imshow(cid, frames[c])

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Esc key to exit
        break
    elif key == 32:  # Space bar to pause
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            # Press Space again to resume, or Esc to quit during pause
            if key2 == 32:
                break
            if key2 == 27:
                exit()  # Or 'break' depending on your loop nesting


for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
