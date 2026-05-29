from pathlib import Path
import ast
import json

import cv2
import numpy as np
import torch


class UltralyticsDetectionModel:
    def __init__(self, model_path, confidence, iou):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("Ultralytics backend requested, but the 'ultralytics' package is not installed.") from error
        self.model = YOLO(str(model_path), task="detect")

    def predict(self, frame):
        results = self.model.predict(frame, verbose=False, conf=self.confidence, iou=self.iou)
        return [] if not results else ultralytics_result_to_detections(results[0], frame.shape)

    def describe(self):
        return f"Detection backend: ultralytics ({self.model_path.name})"


class TensorRtDetectionModel:
    def __init__(self, model_path, confidence, iou, input_size):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.input_size = input_size
        self.names = {}
        try:
            import tensorrt as trt
        except ImportError as error:
            raise ImportError("TensorRT backend requested, but the 'tensorrt' package is not installed.") from error

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires a CUDA-capable PyTorch install.")

        logger = trt.Logger(trt.Logger.INFO)
        with self.model_path.open("rb") as engine_file, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(engine_file.read(4), byteorder="little")
                metadata = json.loads(engine_file.read(meta_len).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                engine_file.seek(0)
                metadata = {}

            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.model_path}")

        self.names = parse_model_names(metadata.get("names"))
        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.tensor_names = []
        self.binding_addresses = {}
        self.output_names = []
        self.input_dtype = None
        self.batch_size = 1
        self.output_tensors = {}

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.tensor_names.append(name)
            tensor_mode = self.engine.get_tensor_mode(name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            tensor_shape = tuple(self.engine.get_tensor_shape(name))

            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_dtype = tensor_dtype
                if len(tensor_shape) != 4:
                    raise RuntimeError(f"Expected BCHW TensorRT input, got {name} shape {tensor_shape}")
                self.batch_size = int(tensor_shape[0])
                self.input_size = int(tensor_shape[2])
            else:
                output_tensor = torch.empty(tensor_shape, device="cuda", dtype=numpy_dtype_to_torch_dtype(tensor_dtype))
                self.output_names.append(name)
                self.output_tensors[name] = output_tensor
                self.binding_addresses[name] = int(output_tensor.data_ptr())

        if self.input_name is None or not self.output_names:
            raise RuntimeError(f"TensorRT engine has invalid inputs/outputs: {self.model_path}")

    def predict(self, frame):
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames):
        if not frames:
            return []

        results = []
        for start_index in range(0, len(frames), self.batch_size):
            batch_frames = frames[start_index:start_index + self.batch_size]
            results.extend(self._predict_batch(batch_frames))
        return results

    def _predict_batch(self, frames):
        input_tensor, preprocess_data = self.preprocess_batch(frames)
        self.binding_addresses[self.input_name] = int(input_tensor.data_ptr())
        self.context.execute_v2([self.binding_addresses[name] for name in self.tensor_names])
        torch.cuda.synchronize()
        outputs = [self.output_tensors[name].detach().cpu().numpy() for name in self.output_names]
        detections = []
        for frame_index, (frame, scale, pad_x, pad_y) in enumerate(preprocess_data):
            frame_outputs = [output[frame_index:frame_index + 1] for output in outputs]
            detections.append(
                model_output_to_detections(
                    frame_outputs,
                    frame.shape,
                    scale,
                    pad_x,
                    pad_y,
                    self.confidence,
                    self.iou,
                    self.names,
                )
            )
        return detections

    def preprocess(self, frame):
        image, scale, pad_x, pad_y = self.preprocess_frame(frame)
        return torch.from_numpy(image[None]).cuda(), scale, pad_x, pad_y

    def preprocess_batch(self, frames):
        images = []
        preprocess_data = []
        for frame in frames:
            image, scale, pad_x, pad_y = self.preprocess_frame(frame)
            images.append(image)
            preprocess_data.append((frame, scale, pad_x, pad_y))

        while len(images) < self.batch_size:
            images.append(images[-1].copy())

        return torch.from_numpy(np.stack(images, axis=0)).cuda(), preprocess_data

    def preprocess_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.input_size / frame_width, self.input_size / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

        image = canvas[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image).astype(self.input_dtype) / 255.0
        return image, scale, pad_x, pad_y

    def describe(self):
        return f"Detection backend: tensorrt-direct ({self.model_path.name})"


class OnnxDetectionModel:
    def __init__(self, model_path, confidence, iou, input_size):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.backend = None
        self.execution_provider = None
        self.input_name = None
        self.input_size = input_size
        self.names = {}
        self.net = None
        self.session = None
        try:
            import onnxruntime as ort

            available_providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available_providers:
                self._preload_gpu_dlls(ort)
            preferred_providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available_providers]
            if not preferred_providers:
                raise RuntimeError(f"No supported ONNX Runtime execution providers available: {sorted(available_providers)}")
            self.session = ort.InferenceSession(str(model_path), providers=preferred_providers)
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int) and input_shape[2] == input_shape[3]:
                self.input_size = input_shape[2]
            self.names = parse_model_names(self.session.get_modelmeta().custom_metadata_map.get("names"))
            self.execution_provider = self.session.get_providers()[0]
            self.backend = "onnxruntime"
            return
        except ImportError:
            pass
        except Exception as error:
            raise RuntimeError(f"Failed to initialize ONNX Runtime for {model_path}: {error}") from error

        try:
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.backend = "opencv-dnn"
        except cv2.error as error:
            raise RuntimeError(
                f"Failed to load ONNX model {model_path}. Install 'onnxruntime' for Ultralytics ONNX exports, or provide a .pt model instead."
            ) from error

    @staticmethod
    def _preload_gpu_dlls(ort):
        try:
            import torch  # Importing torch preloads CUDA/cuDNN DLLs from the PyTorch install.
        except ImportError:
            pass
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

    def preprocess(self, frame):
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.input_size / frame_width, self.input_size / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size), swapRB=True, crop=False)
        return blob, scale, pad_x, pad_y

    def predict(self, frame):
        blob, scale, pad_x, pad_y = self.preprocess(frame)
        if self.backend == "onnxruntime":
            outputs = self.session.run(None, {self.input_name: blob})
        else:
            self.net.setInput(blob)
            outputs = self.net.forward()
        return model_output_to_detections(outputs, frame.shape, scale, pad_x, pad_y, self.confidence, self.iou, self.names)

    def describe(self):
        if self.backend == "onnxruntime":
            return f"Detection backend: onnxruntime ({self.execution_provider}, {self.model_path.name})"
        return f"Detection backend: opencv-dnn ({self.model_path.name})"


def load_detection_model(model_path, confidence, iou, onnx_input_size):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")

    model_suffix = model_path.suffix.lower()
    if model_suffix == ".onnx":
        return OnnxDetectionModel(model_path, confidence, iou, onnx_input_size)
    if model_suffix == ".engine":
        return TensorRtDetectionModel(model_path, confidence, iou, onnx_input_size)
    if model_suffix in {".pt", ".pth"}:
        return UltralyticsDetectionModel(model_path, confidence, iou)
    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def ultralytics_result_to_detections(yolo_result, frame_shape):
    boxes = getattr(yolo_result, "boxes", None)
    names = getattr(yolo_result, "names", {})
    if boxes is None or len(boxes) == 0:
        return []

    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
    classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [None] * len(boxes)
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(boxes)
    detections = []
    for xyxy, track_id, class_id, confidence in zip(boxes.xyxy.cpu().tolist(), track_ids, classes, confidences):
        bounds = clamp_xyxy_to_frame(xyxy, frame_shape)
        if bounds is None:
            continue
        label_name = names.get(class_id, str(class_id)) if class_id is not None else "box"
        detections.append({"bounds": bounds, "confidence": float(confidence), "class_id": class_id, "track_id": track_id, "label": label_name})
    return detections


def clamp_xyxy_to_frame(xyxy, frame_shape):
    x1, y1, x2, y2 = xyxy
    frame_height, frame_width = frame_shape[:2]
    left = max(0, min(frame_width - 1, int(np.floor(x1))))
    top = max(0, min(frame_height - 1, int(np.floor(y1))))
    right = max(left + 1, min(frame_width, int(np.ceil(x2))))
    bottom = max(top + 1, min(frame_height, int(np.ceil(y2))))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def numpy_dtype_to_torch_dtype(dtype):
    if dtype == np.float16:
        return torch.float16
    if dtype == np.float32:
        return torch.float32
    if dtype == np.int32:
        return torch.int32
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")


def parse_model_names(raw_names):
    if isinstance(raw_names, str):
        try:
            raw_names = ast.literal_eval(raw_names)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def model_output_to_detections(outputs, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold, names):
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            return []
        output = outputs[0]
    else:
        output = outputs

    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim == 2 and predictions.shape[1] == 6:
        return nms_xyxy_output_to_detections(
            predictions,
            frame_shape,
            scale,
            pad_x,
            pad_y,
            confidence_threshold,
            iou_threshold,
            names,
        )

    return raw_yolo_output_to_detections(outputs, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold)


def nms_xyxy_output_to_detections(predictions, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold, names):
    boxes = []
    confidences = []
    class_ids = []
    for row in predictions:
        confidence = float(row[4])
        if confidence < confidence_threshold:
            continue

        x1 = (float(row[0]) - pad_x) / scale
        y1 = (float(row[1]) - pad_y) / scale
        x2 = (float(row[2]) - pad_x) / scale
        y2 = (float(row[3]) - pad_y) / scale
        bounds = clamp_xyxy_to_frame((x1, y1, x2, y2), frame_shape)
        if bounds is None:
            continue

        left, top, right, bottom = bounds
        boxes.append([left, top, right - left, bottom - top])
        confidences.append(confidence)
        class_ids.append(int(row[5]))

    if not boxes:
        return []

    kept_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold)
    if kept_indices is None or len(kept_indices) == 0:
        return []

    detections = []
    for kept_index in np.array(kept_indices).reshape(-1):
        index = int(kept_index)
        left, top, width, height = boxes[index]
        class_id = class_ids[index]
        detections.append(
            {
                "bounds": (left, top, left + width, top + height),
                "confidence": float(confidences[index]),
                "class_id": class_id,
                "track_id": None,
                "label": names.get(class_id, str(class_id)),
            }
        )
    return detections


def raw_yolo_output_to_detections(outputs, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold):
    return onnx_output_to_detections(outputs, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold)


def onnx_output_to_detections(outputs, frame_shape, scale, pad_x, pad_y, confidence_threshold, iou_threshold):
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            return []
        outputs = outputs[0]
    predictions = np.asarray(outputs)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] >= 5 and (predictions.shape[1] < 5 or predictions.shape[0] < predictions.shape[1]):
        predictions = predictions.T
    if predictions.shape[1] < 5:
        return []

    boxes = []
    confidences = []
    class_ids = []
    for row in predictions:
        class_scores = row[4:]
        if class_scores.size == 0:
            continue
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue
        cx, cy, width, height = row[:4]
        x1 = (cx - width * 0.5 - pad_x) / scale
        y1 = (cy - height * 0.5 - pad_y) / scale
        x2 = (cx + width * 0.5 - pad_x) / scale
        y2 = (cy + height * 0.5 - pad_y) / scale
        bounds = clamp_xyxy_to_frame((x1, y1, x2, y2), frame_shape)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        boxes.append([left, top, right - left, bottom - top])
        confidences.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    kept_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold)
    if kept_indices is None or len(kept_indices) == 0:
        return []

    detections = []
    for kept_index in np.array(kept_indices).reshape(-1):
        left, top, width, height = boxes[int(kept_index)]
        detections.append(
            {
                "bounds": (left, top, left + width, top + height),
                "confidence": float(confidences[int(kept_index)]),
                "class_id": int(class_ids[int(kept_index)]),
                "track_id": None,
                "label": "box",
            }
        )
    return detections


def draw_yolo_overlay(frame, detections, combined_bounds=None):
    detection_count = len(detections)
    if detection_count == 0:
        if combined_bounds is not None:
            left, top, right, bottom = combined_bounds
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, "combined ROI", (left, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return detection_count

    for detection in detections:
        x1, y1, x2, y2 = detection["bounds"]
        track_id = detection["track_id"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        label_name = detection["label"] if class_id is not None else "box"
        label = f"{label_name} {confidence:.2f}" if track_id is None else f"{label_name} #{track_id} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

    if combined_bounds is not None:
        left, top, right, bottom = combined_bounds
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(frame, "combined ROI", (left, min(frame.shape[0] - 10, bottom + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return detection_count


def select_best_yolo_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda detection: detection["confidence"])
