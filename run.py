import cv2
import numpy as np
import time

from src import embedding_utils, geometry_utils
from src.config import AppConfig
from src.cross_camera_matcher import CrossCameraMatcher
from src.tracking import create_trackers_by_camera, predict_and_track
from src.visualization import Visualizer
from src.yolo import load_detection_model


def _create_masks(captures, mask_points_by_camera):
    masks = []
    for camera_index, cap in enumerate(captures):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        pts = np.array(mask_points_by_camera[camera_index], dtype=np.int32)
        cv2.fillPoly(mask, [pts], (0, 0, 0))
        masks.append(mask)
    return masks



def _register_mouse_callbacks(config, pending_click_by_camera):
    for camera_index, window_name in enumerate(config.window_names):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param, callback_camera_index=camera_index):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_click_by_camera[callback_camera_index] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)


def _handle_pending_clicks(pending_click_by_camera, isolated_track_id_by_camera, frame_draw_data_by_camera):
    camera_count = min(len(pending_click_by_camera), len(frame_draw_data_by_camera))
    for camera_index in range(camera_count):
        pending_click = pending_click_by_camera[camera_index]
        if pending_click is None:
            continue
        if frame_draw_data_by_camera[camera_index] is None:
            pending_click_by_camera[camera_index] = None
            continue

        clicked_box = None
        for box in reversed(frame_draw_data_by_camera[camera_index]["boxes"]):
            if geometry_utils.point_inside_box(pending_click, box["coords"]):
                clicked_box = box
                break

        if clicked_box is not None:
            isolated_track_id_by_camera[camera_index] = clicked_box["track_id"]
        elif isolated_track_id_by_camera[camera_index] is not None:
            isolated_track_id_by_camera[camera_index] = None

        pending_click_by_camera[camera_index] = None


def _point_side_of_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def _track_center(track):
    x1, y1, x2, y2 = map(int, track[:4])
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _crossed_line(previous_center, current_center, crossing_line, directional=False):
    if previous_center is None:
        return False

    crossed = geometry_utils.segments_intersect(
        previous_center,
        current_center,
        crossing_line[0],
        crossing_line[1],
    )
    if not crossed or not directional:
        return crossed

    return (
        _point_side_of_line(previous_center, crossing_line) < 0
        and _point_side_of_line(current_center, crossing_line) > 0
    )


def run(config=None):
    config = config or AppConfig(model_path="cars.engine")
    source_camera_index = 0
    query_camera_index = 1
    camera_count = len(config.video_paths)
    if camera_count <= query_camera_index:
        raise ValueError("The current source/query matching algorithm requires at least two video paths.")
    embedder = embedding_utils.EmbeddingGenerator()
    cross_camera_matcher = CrossCameraMatcher(embedder, config.not_from_other_camera_masks_query_camera)
    visualizer = Visualizer(config)
    model = load_detection_model(config.model_path, confidence=0.02, iou=0.7, onnx_input_size=640)

    previous_centers_by_camera = [dict() for _ in config.video_paths]
    discarded_source_track_ids = set()
    source_track_last_seen_frame = {}
    registered_source_track_ids = set()
    source_exit_seconds = {}
    crossed_seconds_query = {}
    pending_click_by_camera = [None for _ in config.window_names]
    isolated_track_id_by_camera = [None for _ in config.window_names]
    frame_draw_data_by_camera = [None for _ in range(camera_count)]

    _register_mouse_callbacks(config, pending_click_by_camera)

    captures = [cv2.VideoCapture(video_path) for video_path in config.video_paths]
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame_index)

    fps = captures[0].get(cv2.CAP_PROP_FPS) or 10.0
    frame_interval_seconds = 1.0 / fps
    visualizer.set_recording_fps(fps)
    masks = _create_masks(captures, config.mask_points_by_camera)
    trackers = create_trackers_by_camera(config.model_path, camera_count)

    paused = False
    step_next_frame = False
    current_frame_index = config.start_frame_index
    debug_paused_at_target_frame = False
    original_frames = [None for _ in range(camera_count)]
    measured_fps = 0.0
    last_processed_frame_time = None

    while True:
        processed_frame = False

        if not paused or step_next_frame:
            step_next_frame = False
            ret_and_frame_by_camera = [cap.read() for cap in captures]
            ret_by_camera = [ret for ret, _ in ret_and_frame_by_camera]
            if not all(ret_by_camera):
                break

            frame_by_camera = [frame for _, frame in ret_and_frame_by_camera]
            original_frames = [frame.copy() for frame in frame_by_camera]
            masked_frame_by_camera = [
                cv2.bitwise_and(frame, mask)
                for frame, mask in zip(frame_by_camera, masks)
            ]

            tracks_by_camera = predict_and_track(model, masked_frame_by_camera, trackers, original_frames, include_unconfirmed=False)

            cross_camera_matcher.process_query_embeddings(
                tracks_by_camera[query_camera_index],
                original_frames[query_camera_index],
            )
            current_source_track_ids = {int(track[4]) for track in tracks_by_camera[source_camera_index]}

            source_draw_data = {"boxes": [],
                                "others": [],
                                "line": None,
                                "discard_lines": config.source_discard_lines,
                                "frame_text": f"Frame {current_frame_index}",
                                "fps_text": f"FPS {measured_fps:.1f}"}
            for track in tracks_by_camera[source_camera_index]:
                track_id = int(track[4])
                if track_id in discarded_source_track_ids:
                    continue
                x1, y1, x2, y2 = map(int, track[:4])
                previous_center = previous_centers_by_camera[source_camera_index].get(track_id)
                current_center = _track_center(track)
                source_track_last_seen_frame[track_id] = current_frame_index
                for discard_line in config.source_discard_lines:
                    if _crossed_line(previous_center, current_center, discard_line, directional=True):
                        discarded_source_track_ids.add(track_id)
                        cross_camera_matcher.discard_source_camera_track(track_id)
                        source_track_last_seen_frame.pop(track_id, None)
                        registered_source_track_ids.discard(track_id)
                        source_exit_seconds.pop(track_id, None)
                        previous_centers_by_camera[source_camera_index].pop(track_id, None)
                        break
                if track_id in discarded_source_track_ids:
                    continue

                previous_centers_by_camera[source_camera_index][track_id] = current_center

                label = f"{track_id}"
                if previous_center is not None:
                    velocity = np.array((current_center[0] - previous_center[0], current_center[1] - previous_center[1]), dtype='float32')
                    vel_magnitude = np.linalg.norm(velocity)
                    if vel_magnitude > 5.0:
                        min_side_length = min(abs(x2 - x1), abs(y2 - y1))
                        if min_side_length > 40:
                            direction = velocity / vel_magnitude
                            angle = geometry_utils.get_angle_degrees(direction) % 360
                            # label += f" ({angle:.1f} deg) | {vel_magnitude:.2f}"

                            shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2 = geometry_utils.get_shrunk_box(original_frames[source_camera_index], x1, y1, x2, y2, scale=0.8)
                            crop = original_frames[source_camera_index][shrunk_y1:shrunk_y2, shrunk_x1:shrunk_x2]
                            shrunk_track = (shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2, track_id)
                            is_overlapping = geometry_utils.is_box_overlapping(
                                shrunk_track,
                                tracks_by_camera[source_camera_index],
                                min_iou=0.1,
                                box_id=track_id,
                            )
                            is_strong_crop = (
                                not is_overlapping
                                and (shrunk_x2 - shrunk_x1) > 90
                                and 330 <= angle <= 350
                            )
                            cross_camera_matcher.append_source_camera_crop(track_id, crop, is_strong_crop)
                            label = f"{label} {'strong' if is_strong_crop else 'weak'} ({angle:.2f} deg)"

                        source_draw_data["boxes"].append({"track_id": track_id,
                                                          "coords": (x1, y1, x2, y2),
                                                          "label": label,
                                                          "label_color": config.display_colors_by_camera[source_camera_index],
                                                          "box_color": config.display_colors_by_camera[source_camera_index]})

            frame_draw_data_by_camera[source_camera_index] = source_draw_data

            current_frame_seconds = current_frame_index * frame_interval_seconds
            expired_source_track_ids = cross_camera_matcher.discard_expired_source_records(
                current_frame_seconds,
                source_exit_seconds,
                config.source_record_ttl_seconds,
            )
            registered_source_track_ids.difference_update(expired_source_track_ids)

            query_draw_data = {"boxes": [],
                               "others": [],
                               "line": config.entry_line_query,
                               "discard_lines": [],
                               "frame_text": f"Frame {current_frame_index}",
                               "fps_text": f"FPS {measured_fps:.1f}"}
            for track in tracks_by_camera[query_camera_index]:
                track_id = int(track[4])
                if not cross_camera_matcher.query_camera_track_is_relevant(track_id):
                    continue

                x1, y1, x2, y2 = map(int, track[:4])
                previous_center = previous_centers_by_camera[query_camera_index].get(track_id)
                current_center = _track_center(track)
                label = f"{track_id}"

                if _crossed_line(previous_center, current_center, config.entry_line_query):
                    if track_id not in crossed_seconds_query:
                        crossed_seconds_query[track_id] = current_frame_index * frame_interval_seconds

                previous_centers_by_camera[query_camera_index][track_id] = current_center

                if track_id in crossed_seconds_query:
                    label = f"{label} crossed"

                if track_id in crossed_seconds_query:
                    cross_camera_matcher.check_matches(track_id, crossed_seconds_query[track_id], source_exit_seconds)

                query_draw_data["boxes"].append({"track_id": track_id,
                                                 "coords": (x1, y1, x2, y2),
                                                 "label": label,
                                                 "has_crossed_entry_line": track_id in crossed_seconds_query,
                                                 "label_color": config.display_colors_by_camera[query_camera_index],
                                                 "box_color": config.display_colors_by_camera[query_camera_index]})
            frame_draw_data_by_camera[query_camera_index] = query_draw_data

            source_gallery_changed = False
            for track_id, last_seen_frame in source_track_last_seen_frame.items():
                if track_id in current_source_track_ids or track_id in registered_source_track_ids:
                    continue
                if track_id in discarded_source_track_ids:
                    continue

                if not cross_camera_matcher.record_embeddings(track_id):
                    continue

                registered_source_track_ids.add(track_id)
                source_exit_seconds[track_id] = last_seen_frame * frame_interval_seconds
                source_gallery_changed = True

            if source_gallery_changed:
                cross_camera_matcher.refresh_source_camera_gallery()

            processed_frame = True
            if config.debug_pause_at_frame_index == current_frame_index and not debug_paused_at_target_frame:
                paused = True
                debug_paused_at_target_frame = True

        _handle_pending_clicks(pending_click_by_camera, isolated_track_id_by_camera, frame_draw_data_by_camera)

        key = cv2.waitKeyEx(1)
        key_code = key & 0xFF
        if key_code == ord("q"):
            break
        if key_code == ord(" "):
            paused = not paused
        elif paused and key in (83, 63235, 65363, 2555904):
            step_next_frame = True
        else:
            visualizer.handle_key(key_code)

        if processed_frame:
            now = time.perf_counter()
            if last_processed_frame_time is not None:
                frame_seconds = now - last_processed_frame_time
                if frame_seconds > 0:
                    current_fps = 1.0 / frame_seconds
                    if measured_fps == 0.0:
                        measured_fps = current_fps
                    else:
                        measured_fps = measured_fps * 0.9 + current_fps * 0.1
            last_processed_frame_time = now
            current_frame_index += 1

        query_best_matches = (
            cross_camera_matcher.get_best_matches()
            if visualizer.debug_mode
            else cross_camera_matcher.get_chosen_matches()
        )

        visualizer.draw(
            original_frames,
            frame_draw_data_by_camera,
            isolated_track_id_by_camera,
            query_best_matches,
            cross_camera_matcher,
            source_exit_seconds,
        )

    for cap in captures:
        cap.release()
    visualizer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
