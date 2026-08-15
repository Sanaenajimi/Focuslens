"""FocusLens ◈ — Campagne de sensibilisation & outil de vigilance conducteur.

Le front est pensé comme une expérience narrative : route animée, chiffres qui
frappent, simulation visuelle de la seconde fatale, avant l'accès à l'outil.
Palette blanc / bleu profond, storytelling assumé.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyzer import FocusLens
from src.annotate import draw_overlay, compose_dashboard
from src.config import FocusLensConfig
from src.report import build_summary

st.set_page_config(page_title="FocusLens ◈ Drive Safe", page_icon="◈", layout="wide")

# ╔══════════════════════════ DESIGN SYSTEM ══════════════════════════╗
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@700;800;900&family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

.stApp{background:#FFFFFF}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(700px 550px at 92% 5%, rgba(59,130,246,.12) 0%, transparent 55%),
    radial-gradient(650px 500px at -5% 40%, rgba(30,64,175,.09) 0%, transparent 55%);
  animation:breathe 14s ease-in-out infinite}
@keyframes breathe{0%,100%{opacity:.85}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.stApp::before,.reveal,.eye-lid,.eye-iris,.pulse-ring,
  .highway,.car-hero,.tick,.wave,.hero-shine,.count-up,.sim-car,.sim-obs{animation:none !important}}

html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;color:#0F172A}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.6rem;max-width:1200px;position:relative;z-index:1}

@keyframes fadeUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:none}}
.reveal{animation:fadeUp .85s cubic-bezier(.2,.7,.2,1) both}
.d1{animation-delay:.08s}.d2{animation-delay:.22s}.d3{animation-delay:.38s}
.d4{animation-delay:.56s}.d5{animation-delay:.76s}.d6{animation-delay:.96s}

/* ═══════ SCÈNE UNIQUE : PARE-BRISE CONDUCTEUR (boucle 14s) ═══════
   Palette claire premium : brume pêche à l'horizon, ville bleu pâle,
   route gris-bleu, voitures blanches, alertes corail. */
.dash-scene{position:relative;height:600px;border-radius:26px;overflow:hidden;
  background:linear-gradient(180deg,#EBD9CC 0%,#E4E0DC 15%,#D6DEE8 33%,#C9D4E1 50%,#B9C7D7 100%);
  box-shadow:0 30px 70px -24px rgba(70,90,115,.45), inset 0 0 0 1px rgba(255,255,255,.6);
  animation:ds-shake 14s linear infinite}
@keyframes ds-shake{
  0%,70.4%{transform:translate(0,0)}
  70.8%{transform:translate(-9px,5px)}71.4%{transform:translate(10px,-6px)}
  72%{transform:translate(-7px,4px)}72.6%{transform:translate(5px,-3px)}
  73.2%{transform:translate(-2px,1px)}73.8%,100%{transform:translate(0,0)}}
.ds-haze{position:absolute;left:0;right:0;top:26%;height:20%;
  background:linear-gradient(180deg,rgba(240,220,200,0),rgba(244,228,210,.9) 45%,rgba(230,224,218,0));
  filter:blur(7px);z-index:1}
.ds-city{position:absolute;top:20%;height:28%;width:47%;opacity:.8;z-index:0;
  background:linear-gradient(180deg,#A9BACE,#C6D1DF);
  -webkit-mask:linear-gradient(180deg,#000 55%,transparent);mask:linear-gradient(180deg,#000 55%,transparent);
  clip-path:polygon(0 34%,6% 34%,6% 14%,14% 14%,14% 40%,23% 40%,23% 6%,31% 6%,31% 30%,40% 30%,40% 18%,49% 18%,49% 48%,60% 48%,60% 26%,72% 26%,72% 42%,100% 42%,100% 100%,0 100%)}
.ds-city.l{left:0}
.ds-city.r{right:0;transform:scaleX(-1)}
.ds-drift{position:absolute;inset:0;transform-origin:50% 47%;z-index:2;
  animation:ds-drift 14s ease-in-out infinite}
@keyframes ds-drift{0%,56%{transform:rotate(0)}63%{transform:rotate(2.5deg)}70.4%,100%{transform:rotate(5deg)}}
.ds-road{position:absolute;left:0;right:0;top:47%;bottom:0;
  background:linear-gradient(180deg,#B7C3D1 0%,#9DADC0 45%,#8093A9 100%);
  clip-path:polygon(44% 0,56% 0,98% 100%,2% 100%)}
.ds-lane{position:absolute;top:47%;bottom:0;width:4px;opacity:.9;transform-origin:top;
  background:repeating-linear-gradient(180deg,#FFFFFF 0 30px,transparent 30px 78px);
  background-size:100% 108px;animation:ds-lane 14s linear infinite}
.ds-lane.l{left:47.2%;transform:rotate(3.5deg)}
.ds-lane.r{left:52.8%;transform:rotate(-3.5deg)}
@keyframes ds-lane{0%{background-position:0 0}56%{background-position:0 2100px}70.4%,100%{background-position:0 2600px}}
/* voitures : rear-view blanches, la nôtre (voie centrale) se rapproche pendant le sommeil */
.ds-car{position:absolute;z-index:3;filter:drop-shadow(0 10px 16px rgba(60,80,105,.35))}
.ds-car.c1{left:50%;bottom:30%;width:120px;margin-left:-60px;transform-origin:50% 100%;
  animation:ds-approach 14s ease-in infinite}
@keyframes ds-approach{
  0%,52%{transform:scale(.62) translateY(0)}
  60%{transform:scale(.85) translateY(26px)}
  66%{transform:scale(1.35) translateY(72px)}
  70.4%,100%{transform:scale(2.05) translateY(130px)}}
.ds-car.c1 .glow{opacity:0;animation:ds-brake 14s linear infinite}
@keyframes ds-brake{0%,49%{opacity:0}53%,73%{opacity:1}74%,100%{opacity:0}}
.ds-car.c2{left:23%;bottom:34%;width:76px;animation:ds-pass2 14s linear infinite}
@keyframes ds-pass2{0%{transform:scale(.5) translateY(0)}56%{transform:scale(.72) translateY(90px)}
  70.4%,100%{transform:scale(.8) translateY(120px)}}
.ds-car.c3{right:21%;bottom:36%;width:64px;animation:ds-pass3 14s linear infinite}
@keyframes ds-pass3{0%{transform:scale(.46) translateY(-14px)}56%{transform:scale(.6) translateY(50px)}
  70.4%,100%{transform:scale(.66) translateY(72px)}}
/* HUD vitesse */
.ds-hud{position:absolute;top:22px;left:50%;transform:translateX(-50%);z-index:6;text-align:center}
.ds-hud .n{font-family:'Archivo';font-weight:900;font-size:3.2rem;line-height:1;color:#2A3442;
  text-shadow:0 2px 8px rgba(255,255,255,.6)}
.ds-hud .u{font-family:'IBM Plex Mono';font-size:.62rem;letter-spacing:.3em;color:#5B6B7E}
.ds-limit{position:absolute;top:26px;right:26px;z-index:6;width:52px;height:52px;border-radius:50%;
  background:#FFFFFF;border:5px solid #E8574A;display:flex;align-items:center;justify-content:center;
  font-family:'Archivo';font-weight:900;color:#2A3442;font-size:1rem;
  box-shadow:0 8px 20px rgba(70,90,115,.3)}
.ds-brand{position:absolute;top:26px;left:26px;z-index:6;font-family:'IBM Plex Mono';
  font-size:.68rem;letter-spacing:.28em;color:#3A4656;background:rgba(255,255,255,.75);
  padding:8px 16px;border-radius:999px;border:1px solid rgba(255,255,255,.9)}
/* chips d'état, superposées, timées sur la boucle */
.ds-chip{position:absolute;left:50%;transform:translateX(-50%);bottom:36%;z-index:6;
  font-family:'IBM Plex Mono';font-size:.7rem;letter-spacing:.14em;padding:9px 18px;
  border-radius:999px;color:#FFFFFF;opacity:0;white-space:nowrap;
  box-shadow:0 10px 26px rgba(60,80,105,.35)}
.ds-chip.ok{background:#2FA57E;animation:chip-ok 14s linear infinite}
.ds-chip.warn{background:#E79A2E;animation:chip-warn 14s linear infinite}
.ds-chip.alert{background:#E8574A;animation:chip-alert 14s linear infinite}
@keyframes chip-ok{0%,36%{opacity:1}38%,100%{opacity:0}}
@keyframes chip-warn{0%,37%{opacity:0}39%,55%{opacity:1}57%,100%{opacity:0}}
@keyframes chip-alert{0%,56%{opacity:0}58%,72%{opacity:1}74%,100%{opacity:0}}
/* panneau yeux réalistes */
.ds-eyes{position:absolute;left:50%;transform:translateX(-50%);bottom:17%;z-index:7;
  background:rgba(255,255,255,.78);backdrop-filter:blur(9px);
  border:1px solid rgba(255,255,255,.95);border-radius:22px;padding:8px 20px 2px;
  box-shadow:0 16px 38px rgba(70,90,115,.3)}
.eyelid{transform-origin:center 34px;transform:scaleY(0);animation:lids14 14s linear infinite}
@keyframes lids14{
  0%,6%{transform:scaleY(0)}7%,8%{transform:scaleY(1)}9%,18%{transform:scaleY(0)}
  19%,20%{transform:scaleY(1)}21%,34%{transform:scaleY(0)}
  38%,44%{transform:scaleY(.55)}46%,48%{transform:scaleY(.82)}49%,52%{transform:scaleY(.5)}
  54%,56%{transform:scaleY(.88)}58%{transform:scaleY(.6)}
  61%,100%{transform:scaleY(1)}}
.closedline{opacity:0;animation:closed14 14s linear infinite}
@keyframes closed14{0%,60%{opacity:0}62%,95%{opacity:1}97%,100%{opacity:0}}
.iris-g{animation:iris-scan 14s ease-in-out infinite}
@keyframes iris-scan{0%,10%{transform:translate(0,0)}16%{transform:translate(5px,-1px)}
  26%{transform:translate(-5px,1px)}34%{transform:translate(2px,0)}
  44%{transform:translate(0,3px)}56%,100%{transform:translate(0,5px)}}
/* volant */
.ds-wheel{position:absolute;left:50%;bottom:-15%;transform:translateX(-50%);width:640px;z-index:5;
  filter:drop-shadow(0 -8px 30px rgba(50,65,85,.35));animation:wheel-turn 14s ease-in-out infinite;
  transform-origin:50% 85%}
@keyframes wheel-turn{0%,56%{transform:translateX(-50%) rotate(0)}
  63%{transform:translateX(-50%) rotate(-4deg)}70.4%,100%{transform:translateX(-50%) rotate(-9deg)}}
/* crash */
.ds-flash{position:absolute;inset:0;z-index:8;opacity:0;pointer-events:none;
  background:radial-gradient(circle at 50% 45%,#FFFFFF 0%,#F3B7AE 38%,rgba(232,87,74,.55) 60%,transparent 80%);
  animation:ds-flash 14s linear infinite}
@keyframes ds-flash{0%,70%{opacity:0}70.6%{opacity:.97}72%{opacity:.55}74.5%{opacity:0}100%{opacity:0}}
.ds-cracks{position:absolute;inset:0;z-index:8;opacity:0;pointer-events:none;
  animation:ds-cracks 14s linear infinite}
@keyframes ds-cracks{0%,70.4%{opacity:0}71%,95%{opacity:1}98%,100%{opacity:0}}
.ds-msg{position:absolute;inset:0;z-index:9;display:flex;flex-direction:column;align-items:center;
  justify-content:center;opacity:0;pointer-events:none;background:rgba(24,32,44,.85);
  animation:ds-msg 14s linear infinite}
@keyframes ds-msg{0%,74%{opacity:0}78%,95%{opacity:1}98.5%,100%{opacity:0}}
.ds-msg .m1{font-family:'Archivo';font-weight:900;font-size:clamp(1.5rem,3.6vw,2.4rem);
  color:#F4F7FA;letter-spacing:-.01em;text-align:center}
.ds-msg .m2{font-family:'IBM Plex Mono';font-size:.85rem;color:#F3B7AE;margin-top:12px}
.tag{font-family:'IBM Plex Mono';font-size:.72rem;letter-spacing:.32em;
  text-transform:uppercase;color:#1E40AF;text-align:center;margin:2px 0 8px;font-weight:600}

/* ═══════ STUDIO : STEPPER + DIAL + FRISE (inchangés/adaptés) ═══════ */
.steps{display:flex;gap:0;justify-content:center;margin:8px 0 22px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:9px;font-family:'IBM Plex Mono';font-size:.7rem;
  color:#94A3B8;padding:9px 16px;position:relative}
.step .n{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:#FFFFFF;border:1.5px solid #CBD5E1;font-weight:700;flex:none;color:#475569;transition:all .3s}
.step.on{color:#0F172A}
.step.on .n{background:linear-gradient(135deg,#3B82F6,#1E40AF);color:#fff;border-color:transparent;
  box-shadow:0 6px 18px rgba(59,130,246,.5);transform:scale(1.08)}
.step.done .n{background:#10B981;color:#fff;border-color:transparent}
.step:not(:last-child)::after{content:'';width:38px;height:1.5px;background:#CBD5E1;margin-left:14px}
.dial{width:220px;height:220px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  margin:6px auto;position:relative;
  background:conic-gradient(var(--rc) calc(var(--pct)*1%), #EFF6FF 0);
  box-shadow:0 14px 40px rgba(30,58,138,.2), inset 0 0 0 2px rgba(255,255,255,.7)}
.dial::before{content:'';position:absolute;inset:18px;border-radius:50%;background:#FFFFFF;
  box-shadow:inset 0 4px 12px rgba(30,58,138,.08)}
.dial .in{position:relative;text-align:center;z-index:1}
.dial .n{font-family:'Archivo';font-weight:900;font-size:3.3rem;line-height:1;color:var(--rc)}
.dial .l{font-family:'IBM Plex Mono';font-size:.6rem;color:#64748B;text-transform:uppercase;letter-spacing:.22em;margin-top:4px}
.state{font-family:'Archivo';font-weight:800;font-size:1.08rem;text-align:center;
  padding:13px 0;border-radius:16px;margin-top:14px;background:#FFFFFF;border:1px solid #E2E8F0;color:#0F172A}
.alarm{background:linear-gradient(135deg,#FEE2E2,#FECACA) !important;color:#B91C1C !important;
  border-color:#EF4444 !important;animation:siren .7s infinite}
@keyframes siren{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.55)}50%{box-shadow:0 0 0 18px rgba(239,68,68,0)}}
.mini{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;padding:13px 8px;
  text-align:center;box-shadow:0 3px 10px rgba(30,58,138,.06)}
.mini .t{font-family:'IBM Plex Mono';font-size:.58rem;color:#64748B;text-transform:uppercase;letter-spacing:.18em}
.mini .v{font-family:'Archivo';font-weight:800;font-size:1.5rem;margin-top:3px;color:#0F172A}
.strip{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;padding:12px;
  background:#F8FAFC;border-radius:14px;border:1px solid #E2E8F0}
.sec{width:17px;height:32px;border-radius:5px;flex:none;position:relative;
  box-shadow:0 2px 6px rgba(30,58,138,.15), inset 0 1px 0 rgba(255,255,255,.4);transition:transform .12s}
.sec:hover{transform:translateY(-4px) scale(1.3);z-index:3}
.sec:hover::after{content:attr(data-tip);position:absolute;bottom:40px;left:50%;
  transform:translateX(-50%);background:#0F172A;color:#F1F5F9;font-family:'IBM Plex Mono';
  font-size:.62rem;padding:5px 9px;border-radius:7px;white-space:nowrap;z-index:9}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:'IBM Plex Mono';font-size:.66rem;color:#64748B;margin-top:12px}
.legend .dot{display:inline-block;width:11px;height:11px;border-radius:4px;margin-right:5px;vertical-align:-1px}
.story{position:relative;padding-left:36px;margin:14px 0 0}
.story::before{content:'';position:absolute;left:12px;top:6px;bottom:6px;width:2px;
  background:linear-gradient(180deg,#3B82F6,#1E40AF)}
.chap{position:relative;margin:0 0 14px;background:#FFFFFF;border:1px solid #E2E8F0;
  border-radius:16px;padding:14px 18px;box-shadow:0 3px 10px rgba(30,58,138,.06)}
.chap::before{content:'';position:absolute;left:-29px;top:18px;width:12px;height:12px;
  border-radius:50%;background:var(--cc,#3B82F6);border:2.5px solid #FFFFFF;box-shadow:0 0 0 2px var(--cc,#3B82F6)}
.chap .tm{font-family:'IBM Plex Mono';font-size:.66rem;color:#64748B;letter-spacing:.14em}
.chap .tx{font-size:.93rem;margin-top:3px;color:#1E293B;line-height:1.5}
.verdict{border-radius:22px;padding:24px 28px;position:relative;overflow:hidden;
  background:#FFFFFF;border:1px solid #E2E8F0;box-shadow:0 10px 32px rgba(30,58,138,.1)}
.verdict::before{content:'';position:absolute;left:0;top:0;bottom:0;width:7px;background:var(--vc)}
.verdict .big{font-family:'Archivo';font-weight:900;font-size:1.32rem;color:var(--vc)}
.mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:8px 0 20px}
@media(max-width:820px){.mode-grid{grid-template-columns:1fr}}
.mode{background:#FFFFFF;border:2px solid #E2E8F0;border-radius:22px;padding:26px 24px;
  transition:all .3s;position:relative;overflow:hidden}
.mode:hover{border-color:#3B82F6;transform:translateY(-4px);box-shadow:0 20px 40px -10px rgba(59,130,246,.3)}
.mode .big-ic{font-size:2.4rem;display:inline-flex;align-items:center;justify-content:center;
  width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,#DBEAFE,#93C5FD);
  box-shadow:0 8px 20px rgba(59,130,246,.3)}
.mode .mtt{font-family:'Archivo';font-weight:900;font-size:1.35rem;margin:14px 0 6px;color:#0F172A}
.mode .mtx{color:#475569;font-size:.9rem;line-height:1.55}
.mode .hint{margin-top:12px;font-family:'IBM Plex Mono';font-size:.68rem;color:#3B82F6;letter-spacing:.14em}

/* widgets */
.stButton>button{font-family:'IBM Plex Mono';font-weight:600;border-radius:999px;
  border:1.5px solid #3B82F6;color:#1E40AF;padding:.8rem 2rem;background:#FFFFFF;
  box-shadow:0 4px 14px rgba(59,130,246,.15);transition:all .25s;font-size:.9rem;letter-spacing:.05em}
.stButton>button:hover{background:linear-gradient(135deg,#3B82F6,#1E40AF);color:#fff;
  border-color:transparent;transform:translateY(-2px);box-shadow:0 14px 32px rgba(59,130,246,.4)}
[data-testid="stFileUploader"]{background:#F8FAFC;border:2px dashed #93C5FD;border-radius:20px;padding:16px}
[data-testid="stFileUploader"] button{background:#FFFFFF !important;color:#1E40AF !important;
  border:1.5px solid #3B82F6 !important}
div[role="radiogroup"]{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:999px;padding:6px 10px;
  display:inline-flex !important;gap:4px;box-shadow:0 3px 10px rgba(30,58,138,.06)}
.stProgress > div > div{background:linear-gradient(90deg,#3B82F6,#1E40AF)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF,#EFF6FF);border-right:1px solid #E2E8F0}
[data-testid="stSidebar"] h3{color:#1E3A8A}
</style>
""", unsafe_allow_html=True)

# ╔════════════════════ CONSTANTES ════════════════════╗
STATE_META = {
    "somnole":  {"col": "#EF4444", "lab": "🚨 SOMNOLENCE"},
    "distrait": {"col": "#F59E0B", "lab": "⚠ Distraction"},
    "attentif": {"col": "#10B981", "lab": "✅ Vigilant"},
    "moyen":    {"col": "#93C5FD", "lab": "Attention baisse"},
    "absent":   {"col": "#94A3B8", "lab": "Hors champ"},
    "calib":    {"col": "#DBEAFE", "lab": "Calibration"},
}
EVENT_STORY = {
    "microsleep": ("🚨", "#EF4444", "MICRO-SOMMEIL au volant ({d:.1f} s) — à 90 km/h, tu viens de parcourir "
                                    "{km:.0f} m les yeux fermés. C'est LE signal d'arrêt immédiat."),
    "yawn":       ("🥱", "#F59E0B", "Bâillement ({d:.1f} s). Ton corps demande à s'arrêter — écoute-le."),
    "look_away":  ("👀", "#F59E0B", "Yeux hors de la route pendant {d:.1f} s — 3 s à 90 km/h = 75 m à l'aveugle."),
    "absent":     ("🚶", "#94A3B8", "Visage hors champ pendant {d:.1f} s."),
    "long_blink": ("😑", "#93C5FD", "Clignement long ({d:.1f} s) — signe précoce d'hypovigilance."),
}


def fmt_t(s: float) -> str:
    return f"{int(s)//60:02d}:{int(s)%60:02d}"


def second_status(frames: list) -> str:
    """Somnolence = micro-sommeil réel OU >70% du TEMPS (pas des frames) yeux fermés.

    Bug corrigé : l'ancienne version comptait des frames (sum(...)/n), ce qui est
    biaisé à FPS webcam faible ou variable — un clignement normal de 250ms peut
    représenter 3 frames sur 4 captées si le FPS réel chute à ~4-8 (charge CPU
    MediaPipe + rendu Streamlit), déclenchant un faux positif. La version
    corrigée pondère par la durée RÉELLE entre frames (f.t), invariante au FPS."""
    if not frames:
        return "absent"
    if all(f.phase == "calibration" for f in frames):
        return "calib"
    n = len(frames)
    if sum(not f.face_found for f in frames) > n * 0.5:
        return "absent"
    if any(f.microsleep for f in frames):
        return "somnole"
    closed_time, total_time = 0.0, 0.0
    for i in range(1, n):
        dt = frames[i].t - frames[i - 1].t
        if dt > 0:
            total_time += dt
            if frames[i - 1].eye_state == "closed":
                closed_time += dt
    closed_ratio = (closed_time / total_time) if total_time > 0 else 0.0
    if closed_ratio > 0.70:
        return "somnole"
    yaws = [abs(f.yaw) for f in frames if f.yaw is not None]
    if yaws and sorted(yaws)[len(yaws) // 2] > 25:
        return "distrait"
    scores = [f.score for f in frames if f.score is not None]
    mean = sum(scores) / len(scores) if scores else 0
    return "attentif" if mean > 65 else "moyen"


def render_strip(statuses: list[str]) -> str:
    cells = "".join(
        f'<div class="sec" style="background:{STATE_META[s]["col"]}" '
        f'data-tip="{fmt_t(i)} · {STATE_META[s]["lab"]}"></div>' for i, s in enumerate(statuses))
    legend = "".join(
        f'<span><span class="dot" style="background:{m["col"]}"></span>{m["lab"]}</span>'
        for k, m in STATE_META.items() if k != "calib")
    return f'<div class="strip">{cells}</div><div class="legend">{legend}</div>'


def render_dial(score, state: str) -> str:
    col = STATE_META.get(state, STATE_META["attentif"])["col"]
    pct = 0 if score is None else max(0, min(100, score))
    n = "--" if score is None else f"{score:.0f}"
    return (f'<div class="dial" style="--pct:{pct};--rc:{col}">'
            f'<div class="in"><div class="n">{n}</div><div class="l">vigilance</div></div></div>')


def render_spark(scores: list[float], w=560, h=64) -> str:
    if len(scores) < 2:
        return ""
    pts = scores[-240:]
    step = w / (len(pts) - 1)
    path = " ".join(f"{'M' if i == 0 else 'L'}{i*step:.1f},{h - 4 - v/100*(h-8):.1f}"
                    for i, v in enumerate(pts))
    return (f'<div class="card tight"><svg width="100%" height="{h}" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="none"><defs><linearGradient id="gr" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="#3B82F6"/><stop offset="1" stop-color="#1E40AF"/></linearGradient></defs>'
            f'<path d="{path}" fill="none" stroke="url(#gr)" stroke-width="2.8" stroke-linejoin="round"/>'
            f'<line x1="0" y1="{h-4-65/100*(h-8):.1f}" x2="{w}" y2="{h-4-65/100*(h-8):.1f}" '
            f'stroke="#CBD5E1" stroke-dasharray="4 6"/></svg></div>')


def mini(label: str, value: str, col: str = "#0F172A") -> str:
    return f'<div class="mini"><div class="t">{label}</div><div class="v" style="color:{col}">{value}</div></div>'


def render_stepper(active: int, done_upto: int = -1) -> str:
    names = ["Source", "Calibration", "Route", "Bilan"]
    out = '<div class="steps">'
    for i, nm in enumerate(names):
        cls = "step" + (" on" if i == active else "") + (" done" if i <= done_upto else "")
        mark = "✓" if i <= done_upto else str(i + 1)
        out += f'<div class="{cls}"><span class="n">{mark}</span>{nm}</div>'
    return out + "</div>"


def build_story(events, sec_statuses, summary, duration) -> list:
    chaps = [("00:00", "🚗", "#3B82F6",
              "Démarrage. Calibration 3 s — FocusLens apprend TON visage détendu, au repos.")]
    big = [e for e in events if e.kind != "blink"]
    if not big:
        chaps.append(("—", "🌟", "#10B981",
                      "Trajet sans incident : vigilance constante, aucun signal d'alarme. Conduite exemplaire."))
    for e in sorted(big, key=lambda x: x.t_start)[:14]:
        emoji, col, tpl = EVENT_STORY.get(e.kind, ("•", "#3B82F6", "Événement de {d:.1f} s."))
        km = e.duration * 25
        # Sévérité affichée UNIQUEMENT pour microsleep/long_blink, avec le(s)
        # signal(aux) qui l'ont déclenchée : tendance PERCLOS (le TEMPS) et/ou
        # vélocité de fermeture/ouverture lente (le MOUVEMENT) — deux preuves
        # indépendantes d'une fatigue installée plutôt qu'un accident isolé.
        tag = ""
        if e.kind in ("microsleep", "long_blink"):
            slow = (e.closing_velocity is not None and abs(e.closing_velocity) < 3.0 and
                    e.opening_velocity is not None and abs(e.opening_velocity) < 3.0)
            perclos_high = e.perclos_before is not None and e.perclos_before >= 0.15
            if e.severity == "critique":
                signals = []
                if slow:
                    signals.append(f"paupières lentes ({abs(e.closing_velocity):.1f}/"
                                   f"{abs(e.opening_velocity):.1f} baseline/s)")
                if perclos_high:
                    signals.append(f"{e.perclos_before*100:.0f}% du temps fermé sur 30s")
                tag = " · ⚠ " + " + ".join(signals) if signals else " · ⚠ signal de fatigue confirmé"
                col = "#EF4444"
            elif e.severity == "modere":
                tag = " · événement isolé et rapide, pas de tendance de fond"
        chaps.append((fmt_t(e.t_start), emoji, col, tpl.format(d=e.duration, km=km) + tag))
    cnt = Counter(sec_statuses)
    chaps.append((fmt_t(duration), "🏁", "#3B82F6",
                  f"Fin de trajet ({fmt_t(duration)}). Vigilance moyenne {summary['attention']['mean']}/100. "
                  f"Vigilant {cnt.get('attentif', 0)} s · somnolence {cnt.get('somnole', 0)} s · "
                  f"distraction {cnt.get('distrait', 0)} s · {summary['events']['blinks']} clignements."))
    return chaps


# ╔══════════════════ STREAMING ══════════════════╗
def run_stream(cap, fps, cfg, show_lm, target_dt, ph, total_frames, max_seconds, stepper_ph,
               record_path: str | None = None):
    """Boucle d'analyse live. Si record_path est fourni, CHAQUE frame annotée est
    écrite en continu sur disque via cv2.VideoWriter (streaming, pas d'accumulation
    en mémoire) — la vidéo annotée reste disponible après la session, y compris
    pour de longues sessions caméra."""
    fl = FocusLens(cfg)
    results, sec_statuses, sec_buf, score_series = [], [], [], []
    cur_sec, i, t0 = 0, 0, time.time()
    stage = 1
    writer = None
    while True:
        t_loop = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        t = (i / fps) if total_frames else (time.time() - t0)
        if max_seconds and t >= max_seconds:
            break
        res = fl.process(frame, t)
        results.append(res)
        sec_buf.append(res)
        if res.score is not None:
            score_series.append(res.score)
        if stage == 1 and res.phase == "analysis":
            stage = 2
            stepper_ph.markdown(render_stepper(2, 1), unsafe_allow_html=True)
            baseline = fl.temporal.baseline_ear
            thr_c = fl.temporal.thr_closed
            thr_o = fl.temporal.thr_open
            dbg = st.session_state.get("debug_ph")
            if dbg and baseline:
                dbg.markdown(
                    f"Baseline EAR : **{baseline:.3f}**  \n"
                    f"Seuil fermé : **{thr_c:.3f}**  \n"
                    f"Seuil ouvert : **{thr_o:.3f}**  \n"
                    f"*Si baseline ≈ EAR affiché yeux ouverts → OK*"
                )

        # -- overlay annoté : EAR/PERCLOS/score + justification incrustée --
        annotated = draw_overlay(fl.preprocess(frame), res, fl.last_landmarks if show_lm else None, cfg)

        # -- enregistrement de la VUE COMPLÈTE (frame + panneau score/état/métriques/raison),
        #    frame par frame, streamé sur disque --
        if record_path:
            composed = compose_dashboard(annotated, res, cfg)
            if writer is None:
                hh, ww = composed.shape[:2]
                # avc1 (H.264) = lisible dans le navigateur pour la relecture en page ;
                # repli mp4v si le codec n'est pas dispo sur la machine.
                writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"avc1"),
                                         max(1.0, fps), (ww, hh))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, fps), (ww, hh))
            writer.write(composed)

        if i % 2 == 0:
            ph["frame"].image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
        if i % 3 == 0:
            s_now = "calib" if res.phase == "calibration" else second_status(sec_buf)
            meta = STATE_META[s_now]
            ph["dial"].markdown(render_dial(res.score, s_now), unsafe_allow_html=True)
            alarm = " alarm" if s_now == "somnole" else ""
            ph["state"].markdown(f'<div class="state{alarm}" style="color:{meta["col"]}">{meta["lab"]}</div>',
                                 unsafe_allow_html=True)
            # ◈ explicabilité live : la phrase qui justifie la conclusion affichée,
            # avec les valeurs mesurées réelles — pas juste un label de couleur.
            reason = res.reason(cfg)
            ph["reason"].markdown(
                f'<div class="card tight" style="font-size:.82rem;color:#475569">'
                f'{"◈ " + reason if reason else "Aucun signal notable — vigilance normale."}</div>',
                unsafe_allow_html=True)
            c1, c2, c3, c4 = ph["minis"].columns(4)
            c1.markdown(mini("EAR", f"{res.ear:.2f}" if res.ear else "--"), unsafe_allow_html=True)
            c2.markdown(mini("Clign./min", f"{res.blinks_per_min:.0f}"), unsafe_allow_html=True)
            c3.markdown(mini("PERCLOS", f"{res.perclos*100:.0f}%",
                             "#EF4444" if res.perclos > 0.3 else "#0F172A"), unsafe_allow_html=True)
            c4.markdown(mini("Yaw", f"{res.yaw:+.0f}°" if res.yaw is not None else "--"), unsafe_allow_html=True)
            ph["spark"].markdown(render_spark(score_series), unsafe_allow_html=True)
        if int(t) > cur_sec:
            sec_statuses.append(second_status(sec_buf))
            ph["strip"].markdown(render_strip(sec_statuses), unsafe_allow_html=True)
            sec_buf, cur_sec = [], int(t)
        if total_frames:
            ph["prog"].progress(min(1.0, i / total_frames), text=f"⏵ {fmt_t(t)} / {fmt_t(total_frames/fps)}")
        elif max_seconds:
            rec_tag = " ● REC" if record_path else ""
            ph["prog"].progress(min(1.0, t / max_seconds), text=f"● LIVE{rec_tag} {fmt_t(t)} / {fmt_t(max_seconds)}")
        i += 1
        remain = target_dt - (time.time() - t_loop)
        if remain > 0:
            time.sleep(remain)
    duration = (i / fps) if total_frames else (time.time() - t0)
    if sec_buf:
        sec_statuses.append(second_status(sec_buf))
        ph["strip"].markdown(render_strip(sec_statuses), unsafe_allow_html=True)
    fl.finalize(duration)
    cap.release()
    if writer is not None:
        writer.release()
    ph["prog"].progress(1.0, text="✅ Trajet analysé")
    return results, fl, duration, sec_statuses


def make_placeholders():
    col_v, col_l = st.columns([1.25, 1], gap="large")
    with col_v:
        p_frame = st.empty(); p_prog = st.empty()
    with col_l:
        p_dial = st.empty(); p_state = st.empty()
        p_reason = st.empty()   # ◈ explicabilité live : le "pourquoi" en texte
        p_minis = st.empty(); p_spark = st.empty()
    st.markdown('<div class="tag" style="text-align:left;margin-top:14px">◈ chaque seconde de route</div>',
                unsafe_allow_html=True)
    p_strip = st.empty()
    return {"frame": p_frame, "prog": p_prog, "dial": p_dial, "state": p_state,
            "reason": p_reason, "minis": p_minis, "spark": p_spark, "strip": p_strip}


def explain_verdict(summary: dict, events: list) -> str:
    """Synthèse explicable du verdict final : cite l'événement le plus grave et
    UNIQUEMENT les signaux qui ont réellement déclenché sa sévérité — pas de
    justification générique, la preuve numérique précise à chaque fois."""
    critiques = [e for e in events if getattr(e, "severity", "info") == "critique"]
    if critiques:
        worst = max(critiques, key=lambda e: e.duration)
        signals = []
        slow = (worst.closing_velocity is not None and abs(worst.closing_velocity) < 3.0 and
                worst.opening_velocity is not None and abs(worst.opening_velocity) < 3.0)
        perclos_high = worst.perclos_before is not None and worst.perclos_before >= 0.15
        if slow:
            signals.append(f"vitesse de fermeture/réouverture lente "
                           f"({abs(worst.closing_velocity):.1f} baseline/s en fermeture, "
                           f"{abs(worst.opening_velocity):.1f} en réouverture — seuil 3,0)")
        if perclos_high:
            signals.append(f"tendance de fond déjà installée : {worst.perclos_before*100:.0f}% "
                           f"du temps yeux fermés sur les 30 s précédentes (seuil 15%)")
        if not signals:
            signals.append(f"durée de fermeture de {worst.duration:.1f}s dépassant le seuil critique")
        return (f"Le verdict '{summary['verdict']}' s'appuie sur l'événement le plus sévère : "
               f"une fermeture de {worst.duration:.1f}s détectée à {fmt_t(worst.t_start)}, "
               f"classée critique car {' ET '.join(signals)}.")
    modere = [e for e in events if getattr(e, "severity", "info") == "modere"]
    if modere:
        return (f"{len(modere)} fermeture(s) prolongée(s) détectée(s) mais classées 'modérées' : "
               f"ni le PERCLOS de fond (< 15%) ni la vitesse de fermeture/ouverture "
               f"(> 3,0 baseline/s, mouvement rapide) ne dépassaient les seuils d'alerte — "
               f"probablement des clignements volontaires isolés.")
    return ("Aucune fermeture prolongée détectée sur l'ensemble du trajet : le verdict "
           "repose sur le score de vigilance moyen, sans événement individuel à expliquer.")


def show_report(results, fl, duration, sec_statuses, cfg, stepper_ph, record_path: str | None = None):
    stepper_ph.markdown(render_stepper(3, 2), unsafe_allow_html=True)
    summary = build_summary(results, fl.temporal.events, duration, cfg)
    st.write("")
    v = summary["verdict"]
    vc = "#EF4444" if "CRITIQUE" in v else ("#F59E0B" if ("fatigue" in v or "degradee" in v) else "#10B981")
    st.markdown(f'<div class="verdict reveal" style="--vc:{vc}"><span class="big">{v}</span></div>',
                unsafe_allow_html=True)
  
    st.write("")
    cnt = Counter(sec_statuses)
    m = st.columns(5)
    vals = [("Vigilance moy.", f"{summary['attention']['mean']}/100", "#0F172A"),
            ("Sec. somnolence", str(cnt.get("somnole", 0)), "#EF4444" if cnt.get("somnole") else "#0F172A"),
            ("Sec. distraction", str(cnt.get("distrait", 0)), "#0F172A"),
            ("Micro-sommeils", str(summary["events"]["microsleeps"]),
             "#EF4444" if summary["events"]["microsleeps"] else "#0F172A"),
            ("Bâillements", str(summary["events"]["yawns"]), "#0F172A")]
    for k, (col, (l, x, c)) in enumerate(zip(m, vals)):
        col.markdown(f'<div class="reveal d{k+1}">{mini(l, x, c)}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="tag" style="text-align:left">◈ Carnet de bord</div>', unsafe_allow_html=True)
    chaps = build_story(fl.temporal.events, sec_statuses, summary, duration)
    html = '<div class="story">'
    for k, (tm, emoji, col, tx) in enumerate(chaps):
        html += (f'<div class="chap reveal d{min(k+1,6)}" style="--cc:{col}">'
                 f'<div class="tm">{tm}</div><div class="tx">{emoji} {tx}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)
    st.write("")
    # ◈ relecture en page : la session complète (caméra + panneau) revisionnable ici même
    if record_path and Path(record_path).exists() and Path(record_path).stat().st_size > 0:
        st.markdown('<div class="tag" style="text-align:left">◈ Revoir la session</div>',
                    unsafe_allow_html=True)
        with open(record_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.caption("Si la lecture ne démarre pas dans le navigateur, utilise le bouton "
                   "de téléchargement ci-dessous — la vidéo est identique.")
    else:
        video_bytes = None
    st.write("")
    df = pd.DataFrame([r.to_row() for r in results if r.phase == "analysis"])
    cols = st.columns(4 if video_bytes else 3)
    cols[0].download_button("⬇ timeline.csv", df.to_csv(index=False).encode(), "timeline.csv",
                            use_container_width=True)
    cols[1].download_button("⬇ report.json", json.dumps(summary, indent=2, ensure_ascii=False).encode(),
                            "report.json", use_container_width=True)
    if video_bytes:
        cols[2].download_button("⬇ vidéo de session", video_bytes, "session_focuslens.mp4",
                                mime="video/mp4", use_container_width=True)
    if cols[-1].button("↺ Nouveau trajet", use_container_width=True):
        st.session_state.page = "intro"
        st.session_state.mode = None
        st.rerun()


# ╔══════════════════════════ PAGES ══════════════════════════╗
if "page" not in st.session_state:
    st.session_state.page = "intro"
if "mode" not in st.session_state:
    st.session_state.mode = None

# ────────────────── INTRO / CAMPAGNE ──────────────────
if st.session_state.page == "intro":

    st.markdown("""
    <div class="dash-scene reveal">
      <div class="ds-city l"></div>
      <div class="ds-city r"></div>
      <div class="ds-haze"></div>
      <div class="ds-drift">
        <div class="ds-road"></div>
        <div class="ds-lane l"></div>
        <div class="ds-lane r"></div>
        <div class="ds-car c2">
          <svg viewBox="0 0 120 96" xmlns="http://www.w3.org/2000/svg">
            <rect x="14" y="18" width="92" height="66" rx="20" fill="#F6F8FA" stroke="#C9D2DC" stroke-width="1.5"/>
            <rect x="24" y="24" width="72" height="26" rx="10" fill="#9DAEC0"/>
            <rect x="18" y="66" width="22" height="10" rx="5" fill="#D9DFE6"/>
            <rect x="80" y="66" width="22" height="10" rx="5" fill="#D9DFE6"/>
            <rect x="12" y="80" width="96" height="8" rx="4" fill="#2E3844" opacity=".85"/>
          </svg>
        </div>
        <div class="ds-car c3">
          <svg viewBox="0 0 120 96" xmlns="http://www.w3.org/2000/svg">
            <rect x="14" y="18" width="92" height="66" rx="20" fill="#F1F4F7" stroke="#C9D2DC" stroke-width="1.5"/>
            <rect x="24" y="24" width="72" height="26" rx="10" fill="#A7B7C8"/>
            <rect x="12" y="80" width="96" height="8" rx="4" fill="#2E3844" opacity=".85"/>
          </svg>
        </div>
        <div class="ds-car c1">
          <svg viewBox="0 0 140 110" xmlns="http://www.w3.org/2000/svg">
            <g class="glow">
              <ellipse cx="34" cy="84" rx="16" ry="9" fill="#FF5A47" opacity=".55" filter="blur(6px)"/>
              <ellipse cx="106" cy="84" rx="16" ry="9" fill="#FF5A47" opacity=".55"/>
            </g>
            <rect x="12" y="16" width="116" height="78" rx="24" fill="#F8FAFC" stroke="#C6CFDA" stroke-width="2"/>
            <rect x="24" y="24" width="92" height="32" rx="12" fill="#94A6BA"/>
            <rect x="24" y="24" width="92" height="12" rx="8" fill="#FFFFFF" opacity=".25"/>
            <rect x="20" y="76" width="30" height="12" rx="6" fill="#E8574A"/>
            <rect x="90" y="76" width="30" height="12" rx="6" fill="#E8574A"/>
            <rect x="54" y="80" width="32" height="9" rx="4" fill="#DDE3EA"/>
            <rect x="10" y="92" width="120" height="10" rx="5" fill="#2A3440" opacity=".9"/>
          </svg>
        </div>
      </div>
      <div class="ds-brand">◈ FOCUSLENS</div>
      <div class="ds-hud"><div class="n">90</div><div class="u">KM / H</div></div>
      <div class="ds-limit">90</div>
      <div class="ds-chip ok">● VIGILANCE NOMINALE · REGARD SUR LA ROUTE</div>
      <div class="ds-chip warn">◐ VIGILANCE EN BAISSE · PAUPIÈRES LOURDES</div>
      <div class="ds-chip alert">✕ ALERTE · YEUX FERMÉS — LA VOITURE DÉRIVE</div>
      <div class="ds-eyes">
        <svg width="300" height="104" viewBox="0 0 300 104" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="irisG" cx="45%" cy="40%" r="65%">
              <stop offset="0%" stop-color="#B9C2CE"/>
              <stop offset="45%" stop-color="#7C8BA0"/>
              <stop offset="100%" stop-color="#454F5E"/>
            </radialGradient>
          </defs>
          <g>
            <path d="M18,26 C46,8 96,8 124,24" stroke="#262B33" stroke-width="5.5" fill="none" stroke-linecap="round"/>
            <path d="M12,58 C38,30 102,28 132,52 L129,58 C101,40 42,42 16,63 Z" fill="#1E232B"/>
            <path d="M128,50 L138,44 M124,46 L133,38 M119,43 L126,35" stroke="#1E232B" stroke-width="3" stroke-linecap="round"/>
            <path d="M16,62 C42,38 102,38 130,58 C105,80 45,82 16,62 Z" fill="#FFFFFF" stroke="#CBD3DC" stroke-width="1.2"/>
            <g class="iris-g">
              <circle cx="73" cy="59" r="20" fill="url(#irisG)"/>
              <circle cx="73" cy="59" r="20" fill="none" stroke="#39424E" stroke-width="2"/>
              <circle cx="73" cy="59" r="8" fill="#14181F"/>
              <circle cx="66" cy="52" r="4.6" fill="#FFFFFF" opacity=".95"/>
              <circle cx="80" cy="66" r="2.4" fill="#FFFFFF" opacity=".6"/>
            </g>
            <path d="M22,66 C48,84 100,84 126,62" stroke="#2A303A" stroke-width="2.6" fill="none" opacity=".8"/>
            <path d="M10,58 C40,28 104,28 134,54 L134,66 C106,88 42,90 10,66 Z" fill="#FBFBFC" class="eyelid"/>
            <g class="closedline">
              <path d="M18,60 C46,74 100,74 128,58" stroke="#1E232B" stroke-width="5" fill="none" stroke-linecap="round"/>
              <path d="M30,68 L24,76 M52,73 L48,82 M96,73 L100,82 M118,68 L124,76" stroke="#1E232B" stroke-width="2.6" stroke-linecap="round"/>
            </g>
          </g>
          <g transform="translate(300,0) scale(-1,1)">
            <path d="M18,26 C46,8 96,8 124,24" stroke="#262B33" stroke-width="5.5" fill="none" stroke-linecap="round"/>
            <path d="M12,58 C38,30 102,28 132,52 L129,58 C101,40 42,42 16,63 Z" fill="#1E232B"/>
            <path d="M128,50 L138,44 M124,46 L133,38 M119,43 L126,35" stroke="#1E232B" stroke-width="3" stroke-linecap="round"/>
            <path d="M16,62 C42,38 102,38 130,58 C105,80 45,82 16,62 Z" fill="#FFFFFF" stroke="#CBD3DC" stroke-width="1.2"/>
            <g class="iris-g">
              <circle cx="73" cy="59" r="20" fill="url(#irisG)"/>
              <circle cx="73" cy="59" r="20" fill="none" stroke="#39424E" stroke-width="2"/>
              <circle cx="73" cy="59" r="8" fill="#14181F"/>
              <circle cx="66" cy="52" r="4.6" fill="#FFFFFF" opacity=".95"/>
              <circle cx="80" cy="66" r="2.4" fill="#FFFFFF" opacity=".6"/>
            </g>
            <path d="M22,66 C48,84 100,84 126,62" stroke="#2A303A" stroke-width="2.6" fill="none" opacity=".8"/>
            <path d="M10,58 C40,28 104,28 134,54 L134,66 C106,88 42,90 10,66 Z" fill="#FBFBFC" class="eyelid"/>
            <g class="closedline">
              <path d="M18,60 C46,74 100,74 128,58" stroke="#1E232B" stroke-width="5" fill="none" stroke-linecap="round"/>
              <path d="M30,68 L24,76 M52,73 L48,82 M96,73 L100,82 M118,68 L124,76" stroke="#1E232B" stroke-width="2.6" stroke-linecap="round"/>
            </g>
          </g>
        </svg>
      </div>
      <div class="ds-wheel">
        <svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg">
          <path d="M40,320 A280,280 0 0 1 600,320 L556,320 A236,236 0 0 0 84,320 Z" fill="#2E3540"/>
          <path d="M40,320 A280,280 0 0 1 600,320 L588,320 A268,268 0 0 0 52,320 Z" fill="#3B4450" opacity=".7"/>
          <path d="M110,300 L268,252 L268,286 L140,320 Z" fill="#39424E"/>
          <path d="M530,300 L372,252 L372,286 L500,320 Z" fill="#39424E"/>
          <rect x="268" y="248" width="104" height="72" rx="18" fill="#262C36"/>
          <circle cx="320" cy="284" r="10" fill="#3B82F6"/>
        </svg>
      </div>
      <div class="ds-flash"></div>
      <div class="ds-cracks">
        <svg viewBox="0 0 800 600" width="100%" height="100%" preserveAspectRatio="none">
          <g stroke="#3A4656" stroke-width="3.4" fill="none" opacity=".9">
            <path d="M400 290 L268 150 M400 290 L540 130 M400 290 L590 330 M400 290 L300 470 M400 290 L480 500 M400 290 L195 300 M400 290 L640 240"/>
          </g>
          <g stroke="#FFFFFF" stroke-width="1.7" fill="none" opacity=".95">
            <path d="M400 290 L268 150 M400 290 L540 130 M400 290 L590 330 M400 290 L300 470 M400 290 L480 500 M400 290 L195 300 M400 290 L640 240"/>
            <path d="M336 214 L368 240 M472 190 L450 236 M486 356 L444 322 M330 380 L366 340" stroke-width="1.2"/>
            <circle cx="400" cy="290" r="28" stroke-width="2.6"/>
            <circle cx="400" cy="290" r="58" stroke-width="1.4" opacity=".6"/>
          </g>
        </svg>
      </div>
      <div class="ds-msg">
        <div class="m1">Il ne s'est jamais senti fatigué.</div>
        <div class="m2">◈ 4 secondes de micro-sommeil · 100 mètres à l'aveugle</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        if st.button("🎬 Analyser une vidéo", use_container_width=True, key="go_video"):
            st.session_state.page = "studio"
            st.session_state.mode = "video"
            st.rerun()
    with b2:
        if st.button("📷 Caméra live", use_container_width=True, key="go_cam"):
            st.session_state.page = "studio"
            st.session_state.mode = "cam"
            st.rerun()

else:
    st.markdown('<div class="tag" style="text-align:left">focuslens ◈ tableau de bord</div>',
                unsafe_allow_html=True)
    stepper_ph = st.empty()
    stepper_ph.markdown(render_stepper(0), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Réglages fins")
        cfg = FocusLensConfig()
        cfg.microsleep_s = st.slider("Alarme somnolence après (s)", 0.8, 3.0, 1.5, 0.1,
                                     help="Durée minimale yeux fermés pour déclencher l'alarme.")
        cfg.yaw_away_deg = st.slider("Yeux hors route dès (°)", 10, 45, 25)
        show_lm = st.toggle("Landmarks visage", value=True)
        glasses = st.toggle("🕶 Mode lunettes", value=False,
                            help="Réduit encore les seuils si tu portes des lunettes épaisses "
                                 "qui compriment l'EAR. Active si la somnolence n'est pas détectée.")
        if glasses:
            cfg.ear_ratio_closed = 0.50
            cfg.ear_ratio_open = 0.65
            cfg.ear_absolute_floor = 0.04
        st.divider()
        st.markdown("**Debug calibration**")
        debug_ph = st.empty()
        st.session_state["debug_ph"] = debug_ph
        if st.button("← Accueil", use_container_width=True):
            st.session_state.page = "intro"
            st.session_state.mode = None
            st.rerun()

    if st.session_state.mode is None:
        st.markdown('<div class="tag" style="text-align:center;margin-top:8px">◈ comment veux-tu tester ?</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="mode-grid">
          <div class="mode reveal d1"><div class="big-ic">🎬</div>
            <div class="mtt">Analyser une vidéo</div>
            <div class="mtx">Tu as déjà une vidéo de conduite (webcam, dashcam intérieure) ?
            Dépose-la, on la relit et on l'analyse seconde par seconde.</div>
            <div class="hint">→ Idéal pour tester rapidement</div></div>
          <div class="mode reveal d2"><div class="big-ic">📷</div>
            <div class="mtt">Activer ma caméra</div>
            <div class="mtx">FocusLens allume ta webcam et te surveille en direct.
            Simule ta conduite : regarde la route, puis ferme les yeux 2 s. Bilan à la fin.</div>
            <div class="hint">→ Le vrai test grandeur nature</div></div>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            if st.button("🎬 Choisir une vidéo", use_container_width=True, key="pick_video"):
                st.session_state.mode = "video"
                st.rerun()
        with col_b:
            if st.button("📷 Activer ma caméra", use_container_width=True, key="pick_cam"):
                st.session_state.mode = "cam"
                st.rerun()
        st.stop()

    if st.session_state.mode == "video":
        c1, c2 = st.columns([2.6, 1])
        with c2:
            speed = st.select_slider("Vitesse", ["×1", "×2", "×4", "Max"], value="×2")
        with c1:
            uploaded = st.file_uploader("Dépose ton fichier ici", type=["mp4", "avi", "mov", "webm"])
        if not uploaded:
            st.markdown('<div class="card reveal" style="text-align:center;color:#64748B;margin-top:14px">'
                        '🎬 <b>Formats acceptés :</b> mp4, avi, mov, webm.<br>'
                        '<small>Astuce démo : filme-toi 30 s au volant simulé — regarde la route, '
                        'puis le téléphone, puis ferme les yeux 2 s pleines.</small></div>',
                        unsafe_allow_html=True)
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = cfg.fallback_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stepper_ph.markdown(render_stepper(1, 0), unsafe_allow_html=True)
        ph = make_placeholders()
        dt = 1.0 / fps / {"×1": 1, "×2": 2, "×4": 4, "Max": 1e9}[speed]
        record_path = str(Path(tempfile.gettempdir()) / f"focuslens_rec_{int(time.time())}.mp4")
        results, fl, duration, secs = run_stream(cap, fps, cfg, show_lm, dt, ph, total, None,
                                                 stepper_ph, record_path=record_path)
        show_report(results, fl, duration, secs, cfg, stepper_ph, record_path=record_path)

    elif st.session_state.mode == "cam":
        c1, c2, c3 = st.columns([1.2, 1, 1.4])
        with c1:
            dur = st.select_slider("Durée de session", [15, 30, 60, 120, 300], value=30,
                                   format_func=lambda s: f"{s}s" if s < 60 else f"{s//60}min")
        with c2:
            cam_id = st.number_input("Caméra n°", 0, 4, 0)
        with c3:
            st.write("")
            go = st.button("● Démarrer la surveillance", use_container_width=True, key="cam_go")
        st.markdown('<div class="card tight reveal" style="margin-top:12px;color:#475569">'
                    '📋 <b>Comment ça marche :</b> positionne-toi comme au volant, reste 3 s face '
                    'caméra (calibration), puis simule ta conduite. Le bilan tombe à la fin, avec '
                    'la vidéo annotée de la session téléchargeable.</div>',
                    unsafe_allow_html=True)
        if go:
            cap = cv2.VideoCapture(int(cam_id))
            if not cap.isOpened():
                st.error("Caméra introuvable — vérifie le n° et les permissions Windows "
                         "(Paramètres → Confidentialité → Caméra).")
                st.stop()
            fps_guess = cap.get(cv2.CAP_PROP_FPS)
            fps = fps_guess if fps_guess and 1 < fps_guess <= 240 else 30.0
            stepper_ph.markdown(render_stepper(1, 0), unsafe_allow_html=True)
            ph = make_placeholders()
            record_path = str(Path(tempfile.gettempdir()) / f"focuslens_rec_{int(time.time())}.mp4")
            results, fl, duration, secs = run_stream(cap, fps, cfg, show_lm, 0.0, ph,
                                                     None, float(dur), stepper_ph,
                                                     record_path=record_path)
            show_report(results, fl, duration, secs, cfg, stepper_ph, record_path=record_path)