from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2

def l2_normalize(x, axis=1, eps=1e-10):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + eps)


def cosine_similarity(query, gallery):
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)

    if query.ndim == 1:
        query = query[None, :]
    if gallery.ndim == 1:
        gallery = gallery[None, :]

    query = l2_normalize(query)
    gallery = l2_normalize(gallery)

    return query @ gallery.T


def cosine_distance(query, gallery):
    return 1.0 - cosine_similarity(query, gallery)


def find_closest_embeddings(query, gallery, max_count=5):
    # gallery = np.asarray(gallery, dtype=np.float32)
    # if gallery.ndim == 1:
    #     gallery = gallery[None, :]
    if len(gallery) == 0:
        return None, None

    scores = cosine_similarity(query, gallery)
    scores = scores[0]

    max_count = max(1, min(int(max_count), len(scores)))
    best_indices = np.argsort(scores)[::-1][:max_count]
    best_scores = scores[best_indices]
    return best_indices.tolist(), best_scores.astype(float).tolist()


class IBN(nn.Module):
    def __init__(self, planes, ratio=0.5):
        super().__init__()
        half1 = int(planes * ratio)
        half2 = planes - half1
        self.half = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, [self.half, x.size(1) - self.half], dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetIBNBackbone(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), last_stride=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1, ibn=True)
        self.layer2 = self._make_layer(128, layers[1], stride=2, ibn=True)
        self.layer3 = self._make_layer(256, layers[2], stride=2, ibn=True)
        self.layer4 = self._make_layer(512, layers[3], stride=last_stride, ibn=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, blocks, stride, ibn):
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BottleneckIBN.expansion, stride),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )

        layers = [BottleneckIBN(self.inplanes, planes, ibn=ibn, stride=stride, downsample=downsample)]
        self.inplanes = planes * BottleneckIBN.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckIBN(self.inplanes, planes, ibn=ibn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class VehicleReIDResNetIBN(nn.Module):
    def __init__(self, last_stride=2, embedding_dim=2048, dropout_p=0.5, num_classes=34071):
        super().__init__()
        self.model = ResNetIBNBackbone(last_stride=last_stride)
        self.classifier = nn.Module()
        self.classifier.add_block = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_p),
        )
        self.classifier.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier.add_block(x)
        return x


class EmbeddingGenerator:
    DEFAULT_VEHICLE_REID_CKPT = Path("models/net_19.pth")
    DEFAULT_VEHICLE_REID_OPTS = Path("models/opts.yaml")

    def __init__(self, model_path=None, opts_path=None, allow_generic_fallback=False):
        if allow_generic_fallback:
            raise ValueError("Generic fallback is disabled for vehicle embeddings in this project.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EmbeddingGenerator is using: {self.device}")

        self.model_path = Path(model_path) if model_path else self.DEFAULT_VEHICLE_REID_CKPT
        self.opts_path = Path(opts_path) if opts_path else self.DEFAULT_VEHICLE_REID_OPTS
        self.model_kind = "vehicle_reid_resnet50_ibn"
        self.uses_vehicle_model = True

        self.model = self._load_vehicle_model(self.model_path, self.opts_path)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.embedding_dim = self._infer_embedding_dim()

    def _load_vehicle_model(self, model_path, opts_path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vehicle ReID checkpoint was not found at {model_path}."
            )
        if not opts_path.exists():
            raise FileNotFoundError(
                f"Vehicle ReID opts file was not found at {opts_path}."
            )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint type in {model_path}: {type(checkpoint).__name__}")

        model = VehicleReIDResNetIBN()
        model_state = model.state_dict()

        missing_keys = []
        for key in model_state:
            if key not in checkpoint:
                missing_keys.append(key)
                continue
            model_state[key] = checkpoint[key]

        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys for the vehicle ReID model. "
                f"Examples: {missing_keys[:10]}"
            )

        model.load_state_dict(model_state, strict=True)
        return model

    def _infer_embedding_dim(self):
        sample = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            output = self.model(sample)
        return int(output.shape[1])

    @torch.no_grad()
    def get_embeddings(self, crops):
        if len(crops) == 0:
            return None

        batch_tensors = []
        for crop in crops:
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            batch_tensors.append(self.preprocess(pil_img))

        if not batch_tensors:
            return None

        input_batch = torch.stack(batch_tensors).to(self.device)
        embeddings = self.model(input_batch)
        return embeddings.cpu().numpy()

    @staticmethod
    def normalize_embeddings(embeddings):
        return l2_normalize(embeddings)

    @staticmethod
    def compare_embeddings(query, gallery):
        return cosine_similarity(query, gallery)

    @staticmethod
    def compare_embedding_distance(query, gallery):
        return cosine_distance(query, gallery)

    @staticmethod
    def find_closest(query, gallery, max_count=5):
        return find_closest_embeddings(query, gallery, max_count=max_count)
