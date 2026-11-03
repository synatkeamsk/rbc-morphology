"""
src/infer.py
Inference pipeline and ONNX export for SmearScope.
"""

import os
import argparse
from typing import Dict, List, Union
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from src.data import get_transforms
from src.model import get_model


class SmearPredictor:
    def __init__(
        self,
        model_path: str,
        classes: List[str],
        img_size: int = 224,
        backend: str = "torch",
        device: str = "cpu"
    ):
        self.classes = classes
        self.img_size = img_size
        self.backend = backend.lower()
        self.device = device
        self.transform = get_transforms(img_size=img_size, is_train=False)

        if self.backend == "torch":
            self.model = get_model(num_classes=len(classes), pretrained=False, device=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        elif self.backend == "onnx":
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError("Backend must be 'torch' or 'onnx'")

    def preprocess(self, image_path_or_bgr: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_path_or_bgr, str):
            image = cv2.imread(image_path_or_bgr)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_path_or_bgr}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image_path_or_bgr, cv2.COLOR_BGR2RGB)

        return self.transform(image=image)["image"]

    def predict(self, image_path_or_bgr: Union[str, np.ndarray]) -> Dict[str, Union[str, float, Dict[str, float]]]:
        tensor = self.preprocess(image_path_or_bgr)

        if self.backend == "torch":
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            input_tensor = tensor.unsqueeze(0).numpy().astype(np.float32)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))

        pred_idx = int(np.argmax(probs))
        return {
            "prediction": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(self.classes, probs)}
        }


def export_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    num_classes: int,
    backbone_name: str = "convnext_nano",
    img_size: int = 224
):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_model(num_classes=num_classes, backbone_name=backbone_name, pretrained=False, device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=False)
    os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)

    print(f"Exporting ONNX: {output_onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[*] ONNX model successfully verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmearScope Inference & ONNX Export")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_smearscope.pt")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/best_smearscope.onnx")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--image", type=str, help="Path to single RBC image")
    args = parser.parse_args()

    if args.export:
        export_to_onnx(args.checkpoint, args.onnx_path, num_classes=args.num_classes)
