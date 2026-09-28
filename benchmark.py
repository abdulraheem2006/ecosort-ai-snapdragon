"""Measure CPU vs NPU latency on this machine and write benchmarks/results.md."""
import os, time
import numpy as np
from PIL import Image
from npu_infer import WasteClassifier

MODEL = "ecosort_snapdragon.onnx"
img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))


def bench(use_npu, n=100):
    c = WasteClassifier(MODEL, use_npu=use_npu)
    for _ in range(10):
        c.predict(img)
    t = []
    for _ in range(n):
        s = time.perf_counter()
        c.predict(img)
        t.append((time.perf_counter() - s) * 1000)
    return c.provider, float(np.median(t)), float(np.percentile(t, 95))


os.makedirs("benchmarks", exist_ok=True)
rows = [bench(False), bench(True)]
size = os.path.getsize(MODEL) / 1e6
with open("benchmarks/results.md", "w") as f:
    f.write("| Provider | Median latency (ms) | p95 (ms) |\n|---|---|---|\n")
    for p, m, q in rows:
        f.write(f"| {p} | {m:.1f} | {q:.1f} |\n")
    f.write(f"\nModel file size: {size:.1f} MB\n")
print(open("benchmarks/results.md").read())
