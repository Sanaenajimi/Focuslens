"""Rapport final : agrégats JSON, timeline CSV, graphiques PNG.

Le livrable d'un data scientist n'est pas une vidéo qui clignote : c'est un
rapport exploitable. Tout est horodaté et re-traçable frame par frame.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(results: list, path: Path) -> None:
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].to_row().keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())


def build_summary(results: list, events: list, duration: float, cfg) -> dict:
    analysis = [r for r in results if r.phase == "analysis"]
    scores = [r.score for r in analysis if r.score is not None]
    faces = [r for r in analysis if r.face_found]
    ev = lambda k: [e for e in events if e.kind == k]

    return {
        "duration_s": round(duration, 1),
        "frames_analyzed": len(analysis),
        "face_presence_pct": round(100 * len(faces) / max(1, len(analysis)), 1),
        "attention": {
            "mean": round(sum(scores) / max(1, len(scores)), 1) if scores else None,
            "min": round(min(scores), 1) if scores else None,
            "pct_time_focused": round(100 * sum(s > 65 for s in scores) / max(1, len(scores)), 1) if scores else None,
            "pct_time_low": round(100 * sum(s < 40 for s in scores) / max(1, len(scores)), 1) if scores else None,
        },
        "events": {
            "blinks": len(ev("blink")),
            "blinks_per_min": round(len(ev("blink")) / max(duration / 60, 1e-6), 1),
            "microsleeps": len(ev("microsleep")),
            "microsleep_total_s": round(sum(e.duration for e in ev("microsleep")), 1),
            "yawns": len(ev("yawn")),
            "look_away_episodes": len(ev("look_away")),
            "look_away_total_s": round(sum(e.duration for e in ev("look_away")), 1),
            "absent_total_s": round(sum(e.duration for e in ev("absent")), 1),
        },
        "verdict": _verdict(scores, ev),
        "config": cfg.to_dict(),
    }


def _verdict(scores: list, ev) -> str:
    if not scores:
        return "aucun visage exploitable"
    if len(ev("microsleep")) > 0:
        return "FATIGUE CRITIQUE : micro-sommeil(s) detecte(s)"
    mean = sum(scores) / len(scores)
    if len(ev("yawn")) >= 3 or mean < 45:
        return "signes de fatigue / attention degradee"
    if mean > 70:
        return "attention soutenue"
    return "attention moyenne, episodes de distraction"


def plot_timeline(results: list, events: list, path: Path) -> None:
    analysis = [r for r in results if r.phase == "analysis" and r.score is not None]
    if not analysis:
        return
    t = [r.t for r in analysis]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, [r.score for r in analysis], color="#147A6C", lw=1.4)
    axes[0].axhspan(0, 40, color="#E4572E", alpha=0.08)
    axes[0].axhspan(65, 100, color="#147A6C", alpha=0.08)
    axes[0].set_ylabel("Attention /100")
    axes[0].set_ylim(0, 100)

    axes[1].plot(t, [r.ear if r.ear is not None else float("nan") for r in analysis],
                 color="#0E2B27", lw=1)
    axes[1].set_ylabel("EAR")

    axes[2].plot(t, [r.yaw if r.yaw is not None else float("nan") for r in analysis],
                 color="#F2B84B", lw=1, label="yaw")
    axes[2].axhline(25, ls="--", c="grey", lw=0.7)
    axes[2].axhline(-25, ls="--", c="grey", lw=0.7)
    axes[2].set_ylabel("Yaw (°)")
    axes[2].set_xlabel("Temps (s)")

    colors = {"blink": "#8FD6C8", "microsleep": "#E4572E",
              "yawn": "#F2B84B", "look_away": "#B77", "absent": "#999"}
    for e in events:
        if e.kind == "blink":
            continue  # trop nombreux pour être lisibles
        for ax in axes:
            ax.axvspan(e.t_start, e.t_end, color=colors.get(e.kind, "#ccc"), alpha=0.30)

    fig.suptitle("FocusLens — timeline (zones colorées = événements)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_report(results, events, duration, cfg, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, out_dir / "timeline.csv")
    summary = build_summary(results, events, duration, cfg)
    (out_dir / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_timeline(results, events, out_dir / "timeline.png")
    return summary
