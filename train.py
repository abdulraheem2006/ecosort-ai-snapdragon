"""Fine-tune MobileNetV3-Large on a folder-per-class waste dataset.
Usage: python train.py path/to/dataset [epochs]
Merge Garbage Classification v2 and RealWaste into one folder first (one sub-folder per class).
"""
import json, sys
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

data_dir = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder(data_dir, transform=tf)
n_val = int(0.2 * len(ds))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(ds.classes))
model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for ep in range(epochs):
    model.train()
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_dl:
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    print(f"epoch {ep + 1}/{epochs}  val accuracy: {correct / n_val:.4f}")

torch.save(model.cpu().state_dict(), "ecosort.pt")
json.dump(ds.classes, open("labels.json", "w"))
print("saved ecosort.pt and labels.json")
