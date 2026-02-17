import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np


class EmbeddingGenerator:
    def __init__(self):
        # 1. Setup Device (Gaming Laptop GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EmbeddingGenerator is using: {self.device}")

        # 2. Load and Prepare Model
        # Using ResNet18 is often better for real-time tracking on laptops than ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.model.to(self.device)
        self.model.eval()

        # 3. Preprocessing
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embeddings(self, frame, bboxes):
        """
        frame: Full BGR image
        bboxes: List or array of [x1, y1, x2, y2]
        """
        if len(bboxes) == 0:
            return None

        batch_tensors = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Convert to RGB -> PIL -> Tensor
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            batch_tensors.append(self.preprocess(pil_img))

        if not batch_tensors:
            return None

        # 4. Create a single batch and move it to GPU
        # Shape becomes: [N, 3, 224, 224] where N is number of cars
        input_batch = torch.stack(batch_tensors).to(self.device)

        # 5. Run inference on all crops simultaneously
        embeddings = self.model(input_batch)

        # Flatten and return as a numpy array
        # Shape: [N, 2048]
        return embeddings.view(embeddings.size(0), -1).cpu().numpy()