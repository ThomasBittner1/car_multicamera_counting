from dataclasses import dataclass


@dataclass
class AppConfig:
    start_frame_index: int = 0
    debug_mode: bool = False
    show_frame_info: bool = True
    source_record_ttl_seconds: float = 60.0
    record_to_file: str | None = None
    debug_pause_at_frame_index: int | None = None
    model_path: str = "models/cars_1.engine"
    window_names: tuple = ("source c042", "query c041")
    video_paths: tuple = (
        r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
        r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
    )
    entry_line_query: tuple = ((227, 283), (731, 956))

    source_discard_lines: tuple = (
        ((411, 131), (5, 464)),
        ((298, 953), (963, 307)),
    )
    mask_points_by_camera: tuple = (
        (
            (984, 346), (961, 256), (1101, 163), (1027, 101), (881, 126), (684, 165), (472, 124), (328, 141), (236, 182),
            (285, 193), (-2, 334), (0, 2), (1277, 1), (1273, 953), (1059, 957), (1218, 806), (831, 517)
        ),
        (
            (1, 293), (146, 205), (105, 124), (247, 86), (424, 135), (534, 116),
            (728, 168), (1011, 148), (1133, 145), (1199, 197), (1087, 273),
            (1138, 361), (1278, 412), (1276, 3), (5, 3),
        ),
    )
    not_from_other_camera_masks_query_camera: tuple = (
        ((657, 948), (1083, 286), (1278, 419), (1277, 956)),
        ((2, 370), (536, 188), (888, 162), (1277, 199), (1275, 5), (2, 4)),
    )

    display_colors_by_camera: tuple = ((255, 0, 0), (0, 255, 0))
    display_inference_ignore_area_color: tuple = (255, 0, 0)
    display_inference_ignore_area_alpha: float = 0.5
    display_not_from_other_camera_area_color: tuple = (0, 0, 255)
    display_not_from_other_camera_area_alpha: float = 0.5
