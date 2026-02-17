import threading
import cv2
from ultralytics import YOLO


class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

do_break = False
pause_event = threading.Event()
pause_event.set() # Start in 'play' mode


# 1. Define the function each thread will run
def run_tracker_in_thread(filename, model_path, camera_id):
    global do_break
    global do_pause
    cap = cv2.VideoCapture(filename)
    model = YOLO(model_path)  # Each thread gets its own model 'brain' instance

    while True:
        if do_break:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, classes=list(class_names.keys()), imgsz=1088, conf=0.1)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, obj_id, cls_id, conf in zip(boxes, ids, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"{obj_id} ({conf:.03f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            #     trajectories.update(obj_id, (cx, cy), frame[y1:y2, x1:x2])
            #
            # trajectories.process_garbage_collection(ids.tolist())
            # trajectories.draw(frame)

        # Visualize
        cv2.imshow(f"Camera {camera_id}", frame)

        # 3. COORDINATED KEYBOARD INPUT
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            do_break = True  # Signal all other threads to stop
            break

        if key == 32:  # Spacebar
            if pause_event.is_set():
                print(f"[{camera_id}] Pausing all streams...")
                pause_event.clear()
                # When paused, we need a separate waitKey loop to catch the "unpause"
                while not pause_event.is_set() and not do_break:
                    k = cv2.waitKey(100) & 0xFF
                    if k == 32:
                        pause_event.set()
                        print(f"[{camera_id}] Resuming all streams...")
                    if k == ord('q'):
                        do_break = True

    cap.release()


# 2. Setup your sources
sources = [
    ("AICity22_Track1_MTMC_Tracking/test/S06/c041/vdo.avi", "c041"),
    ("AICity22_Track1_MTMC_Tracking/test/S06/c042/vdo.avi", "c042")
]

# 3. Launch the threads
threads = []
for src, cid in sources:
    thread = threading.Thread(target=run_tracker_in_thread, args=(src, "yolo26m.pt", cid), daemon=True)
    threads.append(thread)
    thread.start()

# Wait for threads to finish
for thread in threads:
    thread.join()

cv2.destroyAllWindows()

