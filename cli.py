"""FocusLens CLI — analyse un fichier vidéo OU la webcam.

    python cli.py ma_video.mp4                 # → outputs/ma_video/
    python cli.py ma_video.mp4 --no-video      # rapport seul (plus rapide)
    python cli.py --webcam                     # temps réel, q pour quitter
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.analyzer import FocusLens
from src.annotate import draw_overlay
from src.config import FocusLensConfig
from src.report import save_report


def open_video(path: str, cfg: FocusLensConfig) -> tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:   # métadonnées absentes ou absurdes
        fps = cfg.fallback_fps
        print(f"⚠ fps non fiable dans les métadonnées → fallback {fps}")
    return cap, fps


def analyze_file(path: Path, cfg: FocusLensConfig, write_video: bool = True) -> None:
    cap, fps = open_video(str(path), cfg)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_dir = Path("outputs") / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = FocusLens(cfg)
    results = []
    writer = None
    i, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        res = fl.process(frame, t)
        results.append(res)

        if write_video:
            annotated = draw_overlay(fl.preprocess(frame), res, fl.last_landmarks)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)

        i += 1
        if total and i % 150 == 0:
            speed = i / max(time.time() - t0, 1e-6)
            print(f"  {i}/{total} frames ({100*i/total:.0f}%) · {speed:.0f} fps de traitement")

    duration = i / fps
    fl.finalize(duration)
    cap.release()
    if writer:
        writer.release()

    summary = save_report(results, fl.temporal.events, duration, cfg, out_dir)
    print("\n===== RÉSUMÉ =====")
    print(f"Durée {summary['duration_s']}s · visage présent {summary['face_presence_pct']}%")
    print(f"Attention moyenne : {summary['attention']['mean']}/100 "
          f"(concentré {summary['attention']['pct_time_focused']}% du temps)")
    ev = summary["events"]
    print(f"Clignements {ev['blinks']} ({ev['blinks_per_min']}/min) · "
          f"micro-sommeils {ev['microsleeps']} · bâillements {ev['yawns']} · "
          f"regards ailleurs {ev['look_away_episodes']}")
    print(f"VERDICT : {summary['verdict']}")
    print(f"→ {out_dir}/  (annotated.mp4, timeline.csv, report.json, timeline.png)")


def analyze_webcam(cfg: FocusLensConfig) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Webcam introuvable.")
    fl = FocusLens(cfg)
    t0 = time.time()
    results = []
    print("Webcam active — appuie sur 'q' pour quitter et générer le rapport.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        res = fl.process(frame, t)
        results.append(res)
        cv2.imshow("FocusLens (q pour quitter)",
                   draw_overlay(fl.preprocess(frame), res, fl.last_landmarks))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    duration = time.time() - t0
    fl.finalize(duration)
    cap.release()
    cv2.destroyAllWindows()
    summary = save_report(results, fl.temporal.events, duration, cfg,
                          Path("outputs") / "webcam_session")
    print(f"Session {summary['duration_s']}s · verdict : {summary['verdict']}")
    print("→ outputs/webcam_session/")


def main() -> None:
    ap = argparse.ArgumentParser(description="FocusLens — analyse d'attention vidéo")
    ap.add_argument("video", nargs="?", help="chemin du fichier vidéo (mp4/avi/mov/webm)")
    ap.add_argument("--webcam", action="store_true", help="utiliser la webcam")
    ap.add_argument("--no-video", action="store_true", help="ne pas écrire la vidéo annotée")
    args = ap.parse_args()

    cfg = FocusLensConfig()
    if args.webcam:
        analyze_webcam(cfg)
    elif args.video:
        analyze_file(Path(args.video), cfg, write_video=not args.no_video)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
