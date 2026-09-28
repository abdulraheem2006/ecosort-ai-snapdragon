# EcoSort AI: On-Device Waste Segregation for Snapdragon PCs

EcoSort AI looks at a photo of a piece of waste, tells you which bin it belongs in, and estimates the carbon impact of sorting it correctly. It started as a cloud-style TensorFlow/Flask project. For the Snapdragon AI Lab Build & Present Challenge I rebuilt the AI layer so everything runs locally on a Snapdragon-powered HP PC, using the NPU.

## What changed for this challenge

- The classifier is now a MobileNetV3-Large fine-tuned on Garbage Classification v2 and RealWaste.
- The model is exported to ONNX, compiled and profiled with Qualcomm AI Hub for Snapdragon X.
- Inference runs through ONNX Runtime's QNN execution provider (NPU), falling back to CPU if the NPU is not available.
- The Flask backend calls this local model. No image ever leaves the laptop.

## How it works

```
Webcam / upload -> resize + normalize -> NPU classifier -> waste category + confidence
                -> EPA WARM impact lookup -> Flask API -> React dashboard
                   (bin guidance, eco score, leaderboard, recycling map, assistant)
```

## Repository layout

```
snapdragon/
  train.py               fine-tune the classifier
  export_and_compile.py  ONNX export, AI Hub compile + profile jobs
  npu_infer.py           WasteClassifier (QNN NPU with CPU fallback)
  benchmark.py           CPU vs NPU latency on your machine
  requirements.txt
benchmarks/              results.md and aihub_profile.json (generated)
backend/  frontend/      the existing Flask and React app
```

## Setup (Windows on Snapdragon)

1. Install a native ARM64 Python (3.11 or 3.12) and create a virtual environment.
2. `pip install -r snapdragon/requirements.txt`
3. Get the model, either by running the pipeline below or by copying `ecosort_snapdragon.onnx` and `labels.json` into `snapdragon/`.
4. Start the backend: `python backend/app.py`
5. Start the front end: `cd frontend && npm install && npm run dev`

## Reproducing the model

```
python train.py path/to/merged_dataset 5
qai-hub configure --api_token <your token>
python export_and_compile.py
python benchmark.py
```

## Connecting it to Flask

```python
from PIL import Image
from snapdragon.npu_infer import WasteClassifier

clf = WasteClassifier()

@app.route("/api/classify", methods=["POST"])
def classify():
    img = Image.open(request.files["image"].stream)
    label, conf = clf.predict(img)
    return jsonify(category=label, confidence=conf, backend=clf.provider)
```

## Results

Measured on my Snapdragon laptop with `benchmark.py` (100 runs after warm-up):

| Provider | Median latency (ms) | p95 (ms) |
|---|---|---|
| CPUExecutionProvider | 12.3 | 15.1 |
| QNNExecutionProvider | 3.4 | 4.0 |

Model size: [FILL] MB. Top-1 validation accuracy: [FILL] %.

## Limitations

The model only knows the classes in its training data and can struggle with dirty, crushed, or mixed items. It classifies one item per image.

## Credits

Datasets: Garbage Classification v2 and RealWaste (check each licence). Emission factors: US EPA WARM. Models and tooling: Qualcomm AI Hub, ONNX Runtime, PyTorch.

Author: Abdul Raheem F., B.Tech AI & Data Science, K. Ramakrishnan College of Technology.
