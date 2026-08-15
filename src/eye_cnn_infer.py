"""Pont entre TON modèle entraîné et FocusLens.

Ce module charge le CNN (train/models/eye_cnn.pt) et expose une fonction simple
qui prend les patchs d'yeux découpés dans la frame et rend une probabilité 'fermé'.

FocusLens l'utilise à la place de l'EAR quand cfg.use_cnn_eye = True.
La logique temporelle (durées, hystérésis, calibration) reste identique : on remplace
juste la SOURCE du signal "œil fermé" — heuristique géométrique → réseau appris.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torchvision import transforms
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

_MODEL_PATH = Path(__file__).resolve().parents[1] / "train" / "models" / "eye_cnn.pt"


class EyeStateCNN:
    """Charge le modèle une fois, prédit la proba 'fermé' pour un patch d'œil."""

    def __init__(self, model_path: Path | None = None):
        if not _TORCH_OK:
            raise ImportError("PyTorch requis pour le mode CNN : pip install torch torchvision")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from models.eye_cnn import EyeCNN, IMG_SIZE

        path = model_path or _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {path}\n"
                "Entraîne-le d'abord : python train/train_eye_cnn.py --data train/data")
        ckpt = torch.load(path, map_location="cpu")
        self.model = EyeCNN()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.img_size = ckpt.get("img_size", IMG_SIZE)
        # ImageFolder trie les classes alphabétiquement (closed=0, open=1).
        # On récupère l'index RÉEL de "closed" pour renvoyer la bonne proba.
        c2i = ckpt.get("class_to_idx", {"closed": 0, "open": 1})
        self.closed_idx = c2i.get("closed", 0)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def eye_crop(self, frame_bgr: np.ndarray, eye_pts: np.ndarray, pad: float = 0.6) -> np.ndarray | None:
        """Découpe la région de l'œil autour de ses landmarks (avec marge)."""
        xs, ys = eye_pts[:, 0], eye_pts[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(w, h) * (1 + pad) / 2
        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = int(cx + half), int(cy + half)
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return frame_bgr[y1:y2, x1:x2]

    @torch.no_grad() if _TORCH_OK else (lambda f: f)
    def proba_closed(self, crops: list[np.ndarray]) -> float:
        """Moyenne des probas 'fermé' des deux yeux (robuste si un œil occulté)."""
        valid = [c for c in crops if c is not None and c.size > 0]
        if not valid:
            return 0.0
        batch = torch.stack([self.tf(c) for c in valid])
        logits = self.model(batch)
        p = torch.softmax(logits, dim=1)[:, self.closed_idx]  # proba de la classe "closed"
        return float(p.mean().item())
