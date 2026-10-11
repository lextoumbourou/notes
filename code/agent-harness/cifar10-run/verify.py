"""Verify the saved CIFAR-10 checkpoint on the official test split.

Install torch and torchvision, then run: python verify.py [run_directory]
The official dataset downloads into data/ if absent.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


work = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
checkpoint = torch.load(work / "cifar10_model.pt", map_location="cpu", weights_only=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
])
test = datasets.CIFAR10(work / "data", train=False, download=True, transform=transform)
archive = work / "data/cifar-10-python.tar.gz"
if archive.exists():
    with archive.open("rb") as file:
        checksum = hashlib.file_digest(file, "md5").hexdigest()
    if checksum != "c58f30108f718f92721af3b95e74349a":
        raise ValueError(f"unexpected CIFAR-10 archive checksum: {checksum}")

spec = importlib.util.spec_from_file_location("cifar_agent_train", work / "train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.SmallCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = total = 0
with torch.inference_mode():
    for images, labels in DataLoader(test, batch_size=256, shuffle=False):
        correct += (model(images).argmax(1) == labels).sum().item()
        total += len(labels)

reported = json.loads((work / "results.json").read_text())
if (correct, total) != (reported["test_correct"], reported["test_images"]):
    raise ValueError("saved checkpoint does not reproduce the reported score")
print(f"Verified: {correct}/{total} = {correct / total:.2%}")
