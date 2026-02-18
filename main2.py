import threading
import cv2
import numpy as np
from ultralytics import YOLO
import geometry_utils
import embedding_utils


class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Global signals
do_break = False
is_paused = False  # Switch to a simple boolean for easier toggle


c042_exited_right = []

camera_trajectories = {}


def calculate_embeddings(cid, id, crops):
    print (f'{cid}: car {id} deleted, {len(crops)} crops')
    crops = geometry_utils.get_distributed_crops(crops)
    vector = embedder.get_embeddings(crops)
    mean_vector = np.mean(vector, axis=0)
    print ('mean_vector: ', mean_vector)

    data = camera_trajectories[cid].tracks[id]
    direction = data['direction']
    # if cid == 'c041':
    #     if direction
    print ('direction: ', data['direction'])
    # if data['direction'] > -40 and data['direction'] < -5:
    #     c042_exited_right.append(id)

# --- Launch Logic (Same as before) ---
sources = [
    # ("AICity22_Track1_MTMC_Tracking/test/S06/c041/vdo.avi", "c041"),
    ("AICity22_Track1_MTMC_Tracking/test/S06/c042/vdo.avi", "c042")
]


embedder = embedding_utils.EmbeddingGenerator()
sync_barrier = threading.Barrier(len(sources))

start_time_sec = 80.0

def run_tracker_in_thread(filename, model_path, cid):
    global do_break, is_paused
    cap = cv2.VideoCapture(filename)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    spf = 1.0 / fps
    model = YOLO(model_path)

    last_frame = None

    trajectories = camera_trajectories[cid]
    current_time = start_time_sec - spf
    while True:

        if do_break:
            break

        # --- THE SYNC POINT ---
        # Every thread waits here. Nobody moves until all threads reach this line.
        try:
            sync_barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            break


        if not is_paused:
            current_time += spf

            ret, frame = cap.read()
            if not ret:
                break

            # Heavy processing only happens when NOT paused
            results = model.track(frame, persist=True, verbose=False, classes=list(class_names.keys()), imgsz=640, conf=0.1)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trajectories.update(obj_id, (cx, cy), frame[y1:y2, x1:x2])

                trajectories.process_garbage_collection(ids.tolist(), cid)
                trajectories.draw(frame)

            last_frame = frame.copy()

        cv2.putText(last_frame, f"TIME: {round(current_time, 2)} seconds", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # Display either the new frame or the "frozen" last frame
        if last_frame is not None:
            display_frame = last_frame.copy()
            if is_paused:
                # Add a visual "PAUSED" indicator so you know it's working
                cv2.putText(display_frame, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            cv2.imshow(f"Camera {cid}", display_frame)

        # IMPORTANT: This must run every loop, even when paused!
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            do_break = True
            break
        elif key == 32:  # Spacebar
            is_paused = not is_paused
            state = "Paused" if is_paused else "Resumed"
            print(f"Global State: {state}")

    cap.release()


threads = []
for src, cid in sources:
    camera_trajectories[cid] = geometry_utils.TrajectoryManager(on_delete_callback=calculate_embeddings)
    thread = threading.Thread(target=run_tracker_in_thread, args=(src, "yolo26s.pt", cid), daemon=True)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
cv2.destroyAllWindows()