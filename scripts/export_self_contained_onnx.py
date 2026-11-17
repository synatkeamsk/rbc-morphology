import sys
from pathlib import Path
import torch
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import cfg
from src.model import get_model

CKPT_PATH = ROOT / "model_training/v05_leakage_free/best_smearscope.pt"
OUT_ONNX = ROOT / "model_training/v05_leakage_free/smearscope_deploy.onnx"

model = get_model(
    num_classes=cfg.num_classes,
    backbone_name=cfg.backbone,
    pretrained=False,
    device="cpu",
)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()

dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size)
tmp_onnx = OUT_ONNX.with_suffix(".tmp.onnx")

torch.onnx.export(
    model,
    dummy,
    str(tmp_onnx),
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

onnx_model = onnx.load(str(tmp_onnx), load_external_data=True)
onnx.save_model(onnx_model, str(OUT_ONNX), save_as_external_data=False)

for leftover in [tmp_onnx, Path(str(tmp_onnx) + ".data")]:
    if leftover.exists():
        leftover.unlink()

sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
print(f"[*] Saved self-contained ONNX ({OUT_ONNX.stat().st_size / 1e6:.1f} MB): {OUT_ONNX}")
print("    Inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("    Outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
