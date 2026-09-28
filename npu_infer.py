"""On-device classifier: ONNX Runtime with the Qualcomm QNN execution provider (NPU), CPU fallback."""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class WasteClassifier:
    def __init__(self, model_path="ecosort_snapdragon.onnx", labels_path="labels.json", use_npu=True):
        self.labels = json.load(open(labels_path))
        providers, options = ["CPUExecutionProvider"], [{}]
        if use_npu and "QNNExecutionProvider" in ort.get_available_providers():
            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
            options = [{"backend_path": "QnnHtp.dll", "enable_htp_fp16_precision": "1"}, {}]
        self.session = ort.InferenceSession(model_path, providers=providers, provider_options=options)
        self.provider = self.session.get_providers()[0]

    def _prep(self, img: Image.Image) -> np.ndarray:
        a = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        a = (a.transpose(2, 0, 1) - MEAN) / STD
        return a[None].astype(np.float32)

    def predict(self, img: Image.Image):
        logits = self.session.run(None, {"image": self._prep(img)})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])
