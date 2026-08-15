"""Entraînement du CNN de détection d'œil ouvert/fermé.

    python train/train_eye_cnn.py --data train/data --epochs 12

Ce script montre la MÉCANIQUE COMPLÈTE de l'apprentissage supervisé :
  1. Charger et préparer les données (+ data augmentation)
  2. Séparer train / validation / test (pour une évaluation honnête)
  3. La boucle d'entraînement : forward → loss → backward → step
  4. Suivre les métriques, early stopping
  5. Évaluer sur le test, sauvegarder le modèle et les rapports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.eye_cnn import EyeCNN, IMG_SIZE


# ─────────────────────────── DONNÉES ───────────────────────────
def build_datasets(data_dir: Path, open_dir: str, closed_dir: str):
    """Charge les images depuis data_dir/{open,closed}/ et crée train/val/test.

    Data augmentation (UNIQUEMENT sur le train) : on "invente" des variantes des
    images d'entraînement (petites rotations, translations, changements de
    luminosité) pour que le modèle GÉNÉRALISE au lieu d'apprendre par cœur.
    On N'augmente PAS val/test : on veut les évaluer sur des données "propres".
    """
    # Normalise les noms de dossiers si besoin (Open_Eyes → open, etc.)
    _normalize_folders(data_dir, open_dir, closed_dir)

    train_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(12),                       # œil légèrement penché
        transforms.RandomAffine(0, translate=(0.08, 0.08)),  # œil pas parfaitement centré
        transforms.ColorJitter(brightness=0.3, contrast=0.3),# éclairage variable (jour/nuit)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),                  # recentre les pixels autour de 0
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # On charge tout une fois pour connaître la taille, puis on split les INDICES.
    full = ImageFolder(str(data_dir), transform=eval_tf)
    if full.classes != ["closed", "open"] and full.classes != ["open", "closed"]:
        print(f"⚠ Classes détectées : {full.classes} (attendu open + closed)")

    n = len(full)
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test], generator=g)

    # Le train utilise l'augmentation : on lui donne une vue avec train_tf.
    train_set.dataset = _with_transform(str(data_dir), train_tf)
    return train_set, val_set, test_set, full.class_to_idx


def _normalize_folders(data_dir: Path, open_dir: str, closed_dir: str) -> None:
    """Renomme d'éventuels dossiers Open_Eyes/Closed_Eyes en open/closed."""
    mapping = {open_dir: "open", closed_dir: "closed"}
    for src, dst in mapping.items():
        p_src, p_dst = data_dir / src, data_dir / dst
        if p_src.exists() and not p_dst.exists() and src != dst:
            p_src.rename(p_dst)


def _with_transform(root: str, tf):
    ds = ImageFolder(root, transform=tf)
    return ds


# ─────────────────────────── ENTRAÎNEMENT ───────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Une passe sur tout le loader. train=True → on met à jour les poids."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 1) FORWARD : le modèle prédit
        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            # 2) BACKWARD : calcule le gradient de l'erreur p/r à chaque poids
            optimizer.zero_grad()
            loss.backward()
            # 3) STEP : ajuste les poids dans le sens qui réduit l'erreur
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate_test(model, loader, device):
    """Métriques honnêtes sur le test : accuracy, AUROC, précision, rappel, confusion."""
    from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
    model.eval()
    all_proba, all_pred, all_true = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            proba = torch.softmax(logits, 1)[:, 1].cpu().numpy()  # proba "fermé"
            all_proba += proba.tolist()
            all_pred += (proba > 0.5).astype(int).tolist()
            all_true += labels.tolist()
    # Attention : ImageFolder trie alphabétiquement → closed=0, open=1.
    # On veut "fermé" comme positif, donc on inverse si nécessaire au niveau interprétation.
    return {
        "accuracy": float(np.mean(np.array(all_pred) == np.array(all_true))),
        "auroc": float(roc_auc_score(all_true, all_proba)),
        "precision": float(precision_score(all_true, all_pred, zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, zero_division=0)),
        "confusion": confusion_matrix(all_true, all_pred).tolist(),
    }


def plot_curves(history, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["train_loss"], label="train"); a1.plot(history["val_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(history["train_acc"], label="train"); a2.plot(history["val_acc"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4, help="early stopping")
    ap.add_argument("--open-dir", default="open")
    ap.add_argument("--closed-dir", default="closed")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    data_dir = Path(args.data)
    train_set, val_set, test_set, class_to_idx = build_datasets(
        data_dir, args.open_dir, args.closed_dir)
    print(f"Classes : {class_to_idx}")
    print(f"Train {len(train_set)} · Val {len(val_set)} · Test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch, num_workers=0)

    model = EyeCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params:,} paramètres")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    out_dir = Path("train/models"); out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, patience_left, best_state = 1e9, args.patience, None

    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(vl)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta); history["val_acc"].append(va)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} "
              f"| val loss {vl:.3f} acc {va:.3f}")

        # Early stopping : on garde le meilleur modèle sur la validation.
        if vl < best_val - 1e-4:
            best_val, patience_left = vl, args.patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping à l'epoch {epoch} (val ne s'améliore plus).")
                break

    if best_state:
        model.load_state_dict(best_state)

    # --- ÉVALUATION FINALE sur le test (jamais vu pendant l'entraînement) ---
    metrics = evaluate_test(model, test_loader, device)
    metrics["class_to_idx"] = class_to_idx
    metrics["n_params"] = n_params
    print("\n===== TEST =====")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"AUROC     : {metrics['auroc']:.4f}")
    print(f"Précision : {metrics['precision']:.4f}  Rappel : {metrics['recall']:.4f}")
    print(f"Confusion : {metrics['confusion']}")

    # --- SAUVEGARDES ---
    torch.save({"state_dict": model.state_dict(), "class_to_idx": class_to_idx,
                "img_size": IMG_SIZE}, out_dir / "eye_cnn.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_curves(history, out_dir / "training_curves.png")
    _plot_confusion(metrics["confusion"], out_dir / "confusion.png")
    print(f"\n✅ Modèle et rapports sauvegardés dans {out_dir}/")
    print("   Active-le dans FocusLens : src/config.py → use_cnn_eye = True")


def _plot_confusion(cm, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["closed", "open"]); ax.set_yticklabels(["closed", "open"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Matrice de confusion")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
