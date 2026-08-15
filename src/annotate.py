"""Annotation de la vidéo de sortie : jauges, alertes, landmarks.

Sobriété volontaire : un overlay lisible en un coup d'œil, pas un sapin de Noël.
Couleurs : cyan = OK, orange = attention, rouge = alerte.
"""
from __future__ import annotations

import cv2
import numpy as np

CYAN = (195, 214, 61)      # BGR
ORANGE = (75, 184, 242)
RED = (74, 107, 255)
DARK = (32, 18, 12)

# Palette claire du panneau composé (BGR)
PANEL_BG = (250, 248, 246)
PANEL_CARD = (255, 255, 255)
PANEL_EDGE = (226, 216, 208)
INK = (66, 52, 42)
GREEN = (126, 165, 47)


def _ascii(s: str) -> str:
    """OpenCV putText ne rend pas les accents : translittération ASCII."""
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def compose_dashboard(annotated: np.ndarray, result, cfg) -> np.ndarray:
    """Assemble la frame annotée + un panneau latéral (score, état, métriques,
    justification) en UNE image : c'est CETTE vue complète qui est enregistrée,
    pas seulement le flux caméra. L'utilisateur revoit exactement son tableau
    de bord tel qu'il était à chaque instant."""
    h, w = annotated.shape[:2]
    pw = 400  # largeur du panneau
    canvas = np.full((h, w + pw, 3), PANEL_BG, dtype=np.uint8)
    canvas[:, :w] = annotated
    x0 = w  # origine du panneau

    # -- cadran score --
    cx, cy, r = x0 + pw // 2, 108, 78
    col = _color_for(result.score)
    cv2.circle(canvas, (cx, cy), r, (235, 228, 222), 12)
    if result.score is not None:
        ang = int(360 * min(100, max(0, result.score)) / 100)
        cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, ang, col, 12)
        txt = f"{result.score:.0f}"
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.9, 4)[0]
        cv2.putText(canvas, txt, (cx - size[0] // 2, cy + size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, INK, 4)
    cv2.putText(canvas, "VIGILANCE", (cx - 52, cy + r + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 128, 118), 1)

    # -- état --
    if result.phase == "calibration":
        state_txt, state_col = "CALIBRATION...", ORANGE
    elif result.microsleep:
        state_txt, state_col = "!! SOMNOLENCE !!", RED
    elif not result.face_found:
        state_txt, state_col = "VISAGE ABSENT", ORANGE
    elif result.yaw is not None and abs(result.yaw) > cfg.yaw_away_deg:
        state_txt, state_col = "DISTRAIT", ORANGE
    elif result.score is not None and result.score > 65:
        state_txt, state_col = "VIGILANT", GREEN
    else:
        state_txt, state_col = "ATTENTION MOYENNE", ORANGE
    cv2.rectangle(canvas, (x0 + 22, 224), (x0 + pw - 22, 268), PANEL_CARD, -1)
    cv2.rectangle(canvas, (x0 + 22, 224), (x0 + pw - 22, 268), PANEL_EDGE, 1)
    ts = cv2.getTextSize(state_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    cv2.putText(canvas, state_txt, (cx - ts[0] // 2, 253),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_col, 2)

    # -- 4 métriques --
    metrics = [
        ("EAR", f"{result.ear:.2f}" if result.ear is not None else "--"),
        ("BLK/MIN", f"{result.blinks_per_min:.0f}"),
        ("PERCLOS", f"{result.perclos * 100:.0f}%"),
        ("YAW", f"{result.yaw:+.0f}" if result.yaw is not None else "--"),
    ]
    bw, gap = (pw - 44 - 3 * 10) // 4, 10
    for k, (lab, val) in enumerate(metrics):
        bx = x0 + 22 + k * (bw + gap)
        cv2.rectangle(canvas, (bx, 286), (bx + bw, 356), PANEL_CARD, -1)
        cv2.rectangle(canvas, (bx, 286), (bx + bw, 356), PANEL_EDGE, 1)
        ls = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(canvas, lab, (bx + (bw - ls[0]) // 2, 308),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 138, 128), 1)
        vs = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 2)[0]
        cv2.putText(canvas, val, (bx + (bw - vs[0]) // 2, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, INK, 2)

    # -- justification (le "pourquoi"), sur 3 lignes max --
    reason = result.reason(cfg) if cfg is not None else None
    text = _ascii(reason) if reason else "Aucun signal notable - vigilance normale."
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 <= 44:
            cur = (cur + " " + wd).strip()
        else:
            lines.append(cur); cur = wd
        if len(lines) == 3:
            break
    if cur and len(lines) < 3:
        lines.append(cur)
    y_r = 386
    cv2.rectangle(canvas, (x0 + 22, y_r - 20), (x0 + pw - 22, y_r + 22 * len(lines) + 4),
                  PANEL_CARD, -1)
    cv2.rectangle(canvas, (x0 + 22, y_r - 20), (x0 + pw - 22, y_r + 22 * len(lines) + 4),
                  PANEL_EDGE, 1)
    for k, ln in enumerate(lines):
        cv2.putText(canvas, ln, (x0 + 32, y_r + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, INK, 1)
    return canvas


def _color_for(score: float | None) -> tuple:
    if score is None:
        return ORANGE
    return CYAN if score > 65 else (ORANGE if score > 40 else RED)


def draw_overlay(frame: np.ndarray, result, landmarks: np.ndarray | None, cfg=None) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame

    # landmarks (sous-échantillonnés : 1/6 des 468 points suffit visuellement)
    if landmarks is not None:
        for p in landmarks[::6]:
            cv2.circle(out, (int(p[0]), int(p[1])), 1, CYAN, -1)

    # bandeau
    cv2.rectangle(out, (0, 0), (w, 58), DARK, -1)
    col = _color_for(result.score)

    if result.phase == "calibration":
        cv2.putText(out, "CALIBRATION... regarde la camera normalement",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
        return out

    score_txt = f"{result.score:.0f}" if result.score is not None else "--"
    cv2.putText(out, f"ATTENTION {score_txt}/100", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    # jauge
    cv2.rectangle(out, (12, 34), (212, 48), (60, 60, 60), 1)
    if result.score is not None:
        cv2.rectangle(out, (12, 34), (12 + int(2 * result.score), 48), col, -1)

    info = []
    if result.ear is not None:
        info.append(f"EAR {result.ear:.2f}")
    info.append(f"blk/min {result.blinks_per_min:.0f}")
    info.append(f"PERCLOS {result.perclos * 100:.0f}%")
    if result.yaw is not None:
        info.append(f"yaw {result.yaw:+.0f}")
    cv2.putText(out, "  ".join(info), (226, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 240), 1)

    # alertes
    if result.microsleep:
        cv2.putText(out, "!! MICRO-SOMMEIL !!", (12, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)
    elif not result.face_found:
        cv2.putText(out, "visage absent", (12, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
    elif result.yaw is not None and abs(result.yaw) > 25:
        cv2.putText(out, "regard hors camera", (12, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)

    # -- explicabilité : le "pourquoi" incrusté directement sur l'image --
    # Bandeau bas semi-transparent + phrase justifiant l'état courant avec les
    # valeurs mesurées réelles (pas juste un label, la preuve numérique derrière).
    if cfg is not None:
        reason = result.reason(cfg)
        if reason:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, h - 30), (w, h), DARK, -1)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
            cv2.putText(out, _ascii(reason)[:95], (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 220, 235), 1)
    return out