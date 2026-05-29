import cv2
import numpy as np
import time

from src.config import AppConfig
from src.tracking import create_trackers_by_camera, tracks_from_detections
from src.yolo import load_detection_model


def create_masks(captures, mask_points_by_camera):
    masks = []
    for camera_index, cap in enumerate(captures):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        points = np.array(mask_points_by_camera[camera_index], dtype=np.int32)
        cv2.fillPoly(mask, [points], (0, 0, 0))
        masks.append(mask)
    return masks


def draw_tracks(frame, tracks, color):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{track_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_detections(frame, detections, color=(0, 255, 255)):
    for detection in detections:
        x1, y1, x2, y2 = detection["bounds"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{detection['confidence']:.2f}",
            (x1, min(frame.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def draw_frame_count(frame, current_frame_index):
    cv2.putText(
        frame,
        f"Frame {current_frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def run(config=None):
    config = config or AppConfig()
    model = load_detection_model(config.model_path, confidence=0.02, iou=0.7, onnx_input_size=640)
    trackers = create_trackers_by_camera(config.model_path, len(config.video_paths))

    captures = [cv2.VideoCapture(video_path) for video_path in config.video_paths]
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame_index)

    masks = create_masks(captures, config.mask_points_by_camera)
    fps = captures[0].get(cv2.CAP_PROP_FPS) or 10.0
    interval_ms = 1 #max(1, int(round(1000.0 / fps)))
    paused = False
    step_next_frame = False
    current_frame_index = config.start_frame_index
    original_frames = [None for _ in captures]
    measured_fps = 0.0
    previous_frame_time = time.perf_counter()

    for window_name in config.window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        processed_frame = False
        if not paused or step_next_frame:
            step_next_frame = False
            ret_and_frame_by_camera = [cap.read() for cap in captures]
            if not all(ret for ret, _ in ret_and_frame_by_camera):
                break

            frame_by_camera = [frame for _, frame in ret_and_frame_by_camera]
            original_frames = [frame.copy() for frame in frame_by_camera]
            masked_frame_by_camera = [
                cv2.bitwise_and(frame, mask)
                for frame, mask in zip(frame_by_camera, masks)
            ]
            if hasattr(model, "predict_batch"):
                detections_by_camera = model.predict_batch(masked_frame_by_camera)
            else:
                detections_by_camera = [model.predict(frame) for frame in masked_frame_by_camera]
            tracks_by_camera = [
                tracks_from_detections(
                    detections_by_camera[camera_index],
                    trackers[camera_index],
                    original_frames[camera_index],
                    include_unconfirmed=False,
                )
                for camera_index in range(len(detections_by_camera))
            ]
            current_frame_time = time.perf_counter()
            elapsed_seconds = current_frame_time - previous_frame_time
            previous_frame_time = current_frame_time
            if elapsed_seconds > 0:
                measured_fps = 1.0 / elapsed_seconds

            for camera_index, tracks in enumerate(tracks_by_camera):
                draw_frame = original_frames[camera_index].copy()
                draw_detections(draw_frame, detections_by_camera[camera_index])
                draw_tracks(draw_frame, tracks, config.display_colors_by_camera[camera_index])
                draw_frame_count(draw_frame, current_frame_index)
                draw_fps(draw_frame, measured_fps)
                cv2.imshow(config.window_names[camera_index], draw_frame)
            processed_frame = True

        key = cv2.waitKeyEx(interval_ms)
        key_code = key & 0xFF
        if key_code == ord("q"):
            break
        if key_code == ord(" "):
            paused = not paused
        elif paused and key in (83, 63235, 65363, 2555904):
            step_next_frame = True

        if processed_frame:
            current_frame_index += 1

    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
