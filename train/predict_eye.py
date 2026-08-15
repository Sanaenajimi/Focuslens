"""Teste TON modèle entraîné sur une image d'œil.

    python train/predict_eye.py --model train/models/eye_cnn.pt --image oeil.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.eye_cnn import EyeCNN, IMG_SIZE


def load_model(path: str) -> EyeCNN:
    ckpt = torch.load(path, map_location="cpu")
    model = EyeCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def preprocess(image_path: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tf(Image.open(image_path)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="train/models/eye_cnn.pt")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    model = load_model(args.model)
    x = preprocess(args.image)
    p_closed = model.predict_closed_proba(x).item()
    verdict = "FERMÉ 😴" if p_closed > 0.5 else "OUVERT 👁️"
    print(f"Probabilité 'fermé' : {p_closed:.3f}  →  {verdict}")


if __name__ == "__main__":
    main()
