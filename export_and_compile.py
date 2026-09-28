"""Export the trained model to ONNX, then compile + profile it on a Snapdragon X device with Qualcomm AI Hub.
One-time setup:  qai-hub configure --api_token <your token from aihub.qualcomm.com>
Check option names against the current AI Hub docs if a job is rejected.
"""
import json, os
import torch
import qai_hub as hub
from torchvision import models

labels = json.load(open("labels.json"))
model = models.mobilenet_v3_large()
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(labels))
model.load_state_dict(torch.load("ecosort.pt", map_location="cpu"))
model.eval()

x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, x, "ecosort.onnx", input_names=["image"], output_names=["logits"], opset_version=17)
print("wrote ecosort.onnx")

devices = [d for d in hub.get_devices() if "Snapdragon X" in d.name]
if not devices:
    raise SystemExit("No Snapdragon X device found in AI Hub device list")
device = devices[0]
print("target device:", device.name)

traced = torch.jit.trace(model, x)
compile_job = hub.submit_compile_job(
    model=traced,
    device=device,
    input_specs={"image": ((1, 3, 224, 224), "float32")},
    options="--target_runtime onnx",
)
compile_job.wait()
compile_job.get_target_model().download("ecosort_snapdragon.onnx")

profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
profile_job.wait()
os.makedirs("benchmarks", exist_ok=True)
json.dump(profile_job.download_profile(), open("benchmarks/aihub_profile.json", "w"), indent=2, default=str)
print("compile job:", compile_job.url)
print("profile job:", profile_job.url)
