"""
Audio Visualizer Pro — Headless/GUI Hybrid
Fix: gracefully handle environments without Tkinter by providing a CLI (headless) mode.

New in this build:
- Video **progress bar** overlay (shows track progress), position + size configurable.
- **Clock overlays**: digital and analog. Modes: system time or track elapsed time. Multiple styles and positions.
- **NEW** (per feedback): optional **reverse countdown** near the progress bar; digital clock defaults to HH:MM:SS and **elapsed** mode by default.

Usage (CLI):
  python audio_visualizer.py --audio path/to/file.mp3 --style bars --palette rainbow \
      --resolution 1280x720 --fps 30 --quality high --codec h264 --bitrate 192k \
      --logo-text "My Track" --background gradient --particles --beat-react \
      --progress-bar --progress-pos bottom --progress-height 10 --progress-countdown \
      --clock digital --clock-mode elapsed --clock-style dark --clock-pos tr --clock-scale 1.0

Run tests:
  python audio_visualizer.py --run-tests

Notes:
- If Tkinter is available, you still get the GUI. Otherwise, the CLI runs.
- ffmpeg is required for final audio+video mux; the script falls back to a video-only file if ffmpeg is not found.
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import tempfile
import subprocess
import datetime as _dt
from dataclasses import dataclass
from threading import Thread, Event
from typing import Callable, Tuple, Optional

# Third-party libs — optional detection for better error messages
try:  # numpy is required
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError("numpy is required: pip install numpy") from e

try:
    import cv2  # for drawing and VideoWriter
    CV2_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    import librosa  # audio analysis
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None  # type: ignore
    LIBROSA_AVAILABLE = False

# Tkinter is optional — we *must not* crash if it's missing
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_AVAILABLE = True
except Exception:
    tk = None  # type: ignore
    filedialog = messagebox = ttk = None  # type: ignore
    TK_AVAILABLE = False

# =============================
# Utility helpers
# =============================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# =============================
# Color utilities
# =============================
class Palette:
    @staticmethod
    def scheme(name: str) -> Callable[[float, int, int], Tuple[int, int, int]]:
        name = (name or "rainbow").lower()
        return getattr(Palette, name, Palette.rainbow)

    @staticmethod
    def _to_bgr(r: float, g: float, b: float) -> Tuple[int, int, int]:
        # clamp to [0,255]
        r = clamp(r, 0, 255)
        g = clamp(g, 0, 255)
        b = clamp(b, 0, 255)
        return (int(b), int(g), int(r))

    @staticmethod
    def rainbow(f: float, i: int, total: int) -> Tuple[int, int, int]:
        hue = (i / max(total, 1)) * 2 * math.pi
        r = 127 + 127 * math.sin(hue)
        g = 127 + 127 * math.sin(hue + 2 * math.pi / 3)
        b = 127 + 127 * math.sin(hue + 4 * math.pi / 3)
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def fire(f: float, i: int, total: int) -> Tuple[int, int, int]:
        r = 255 * f
        g = 180 * f
        b = 60 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def ocean(f: float, i: int, total: int) -> Tuple[int, int, int]:
        r = 60 * f
        g = 180 * f
        b = 255 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def neon(f: float, i: int, total: int) -> Tuple[int, int, int]:
        ch = i % 3
        r = 255 * f if ch == 0 else 50
        g = 255 * f if ch == 1 else 50
        b = 255 * f if ch == 2 else 50
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def purple(f: float, i: int, total: int) -> Tuple[int, int, int]:
        r = 200 * f
        g = 50 * f
        b = 255 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def gradient(f: float, i: int, total: int) -> Tuple[int, int, int]:
        ratio = i / max(total, 1)
        r = 255 * ratio * f
        g = 128 * (1 - ratio) * f
        b = 200 * f
        return Palette._to_bgr(r, g, b)

# =============================
# Audio analysis helpers
# =============================
class AudioAnalysis:
    """Precompute per-frame features: mel-spectrogram aligned to FPS + beat detection.
       For tests, can be constructed with raw y/sr instead of a file path.
    """

    def __init__(self,
                 path: Optional[str],
                 sr: int = 22050,
                 fps: int = 30,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 y: Optional[np.ndarray] = None):
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa is required for audio analysis: pip install librosa")

        self.sr = sr
        self.fps = fps
        self.n_fft = n_fft
        self.n_mels = n_mels
        if y is None:
            if not path:
                raise ValueError("AudioAnalysis requires either path or y array")
            self.y, self.sr = librosa.load(path, sr=sr, mono=True)
        else:
            self.y = y
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))

        # Hop aligned to FPS (ensures consistent energy per frame)
        self.hop_length = max(1, int(self.sr / max(1, self.fps)))

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / max(1e-9, (S_db.max() - S_db.min()))
        self.mels = S_db  # (n_mels, n_frames)
        self.n_frames = int(self.mels.shape[1])

        # Onset-based beat tracking
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop_length)
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length)
        self.beat_frames = set(int(b) for b in beat_frames)

    def frame_features(self, frame_idx: int):
        idx = int(clamp(frame_idx, 0, self.n_frames - 1))
        mel_col = self.mels[:, idx]
        low = float(mel_col[: self.n_mels // 4].mean())
        mid = float(mel_col[self.n_mels // 4: self.n_mels // 2].mean())
        high = float(mel_col[self.n_mels // 2:].mean())
        beat = idx in self.beat_frames
        return mel_col, (low, mid, high), beat

# =============================
# Overlays (progress bar + clocks)
# =============================

def _overlay_progress_bar(img: np.ndarray, progress: float, position: str = 'bottom', height: int = 10,
                          color: Tuple[int, int, int] = (255, 255, 255), label: Optional[str] = None) -> np.ndarray:
    """Draw a simple filled progress bar. progress in [0,1]. If label is set, draw it near the right edge."""
    h, w = img.shape[:2]
    height = max(2, int(height))
    pad = 6
    y0 = (h - height - pad) if position == 'bottom' else pad
    y1 = y0 + height

    # background bar (alpha dark)
    bg = img.copy()
    cv2.rectangle(bg, (pad, y0), (w - pad, y1), (0, 0, 0), -1)
    img = cv2.addWeighted(bg, 0.35, img, 0.65, 0)

    # foreground fill
    x1 = pad + int((w - 2 * pad) * clamp(progress, 0.0, 1.0))
    cv2.rectangle(img, (pad, y0), (x1, y1), color, -1)

    # small handle marker
    cv2.rectangle(img, (x1-2, y0), (x1+2, y1), (240, 240, 240), -1)

    # optional label (e.g., countdown "-MM:SS")
    if label:
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.5
        thick = 1
        size, _ = cv2.getTextSize(label, font, scale, thick)
        tx = w - pad - size[0]
        ty = y0 - 4 if position == 'bottom' else y1 + size[1] + 2
        ty = clamp(ty, 12, h - 6)
        cv2.putText(img, label, (int(tx)+1, int(ty)+1), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
        cv2.putText(img, label, (int(tx), int(ty)), font, scale, (240,240,240), thick, cv2.LINE_AA)

    return img


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_hms_full(seconds: float) -> str:
    """Always HH:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _clock_style_colors(style: str):
    style = (style or 'dark').lower()
    if style == 'light':
        return dict(face=(245,245,245), ring=(200,200,200), tick=(90,90,90), hour=(50,50,50), minute=(70,70,70), second=(40,40,230), text=(10,10,10))
    if style == 'retro':
        return dict(face=(240,230,200), ring=(60,90,120), tick=(40,60,80), hour=(30,30,30), minute=(50,50,50), second=(20,120,220), text=(20,30,40))
    if style == 'neon':
        return dict(face=(10,10,10), ring=(0,230,230), tick=(60,255,255), hour=(0,200,255), minute=(0,255,180), second=(255,80,200), text=(240,240,240))
    if style == 'outline':
        return dict(face=None, ring=(240,240,240), tick=(240,240,240), hour=(240,240,240), minute=(240,240,240), second=(60,200,255), text=(240,240,240))
    # default dark
    return dict(face=(20,20,20), ring=(200,200,200), tick=(200,200,200), hour=(240,240,240), minute=(200,200,200), second=(60,200,255), text=(235,235,235))


def _place_rect(w: int, h: int, rect_w: int, rect_h: int, pos: str, margin: int = 18) -> Tuple[int, int]:
    pos = (pos or 'tr').lower()
    if pos == 'tl':
        return margin, margin
    if pos == 'tr':
        return w - rect_w - margin, margin
    if pos == 'bl':
        return margin, h - rect_h - margin
    # default br
    return w - rect_w - margin, h - rect_h - margin


def _overlay_digital_clock(img: np.ndarray, text: str, pos: str, style: str, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    colors = _clock_style_colors(style)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8 * max(0.4, float(scale))
    thickness = 2
    size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = _place_rect(w, h, size[0] + 20, size[1] + 16, pos)

    # background bubble
    if colors.get('face') is not None:
        cv2.rectangle(img, (x, y), (x + size[0] + 20, y + size[1] + 16), colors['face'], -1)
    cv2.rectangle(img, (x, y), (x + size[0] + 20, y + size[1] + 16), colors['ring'], 1)

    # text center baseline
    tx, ty = x + 10, y + size[1] + 4
    # small shadow
    cv2.putText(img, text, (tx+2, ty+2), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, font_scale, colors['text'], thickness, cv2.LINE_AA)
    return img


def _overlay_analog_clock(img: np.ndarray, dt: _dt.datetime, pos: str, style: str, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    colors = _clock_style_colors(style)
    r = int(min(w, h) * 0.08 * max(0.5, float(scale)))
    cx, cy = _place_rect(w, h, 2*r, 2*r, pos)
    cx += r; cy += r

    if colors.get('face') is not None:
        cv2.circle(img, (cx, cy), r, colors['face'], -1)
    cv2.circle(img, (cx, cy), r, colors['ring'], 2)

    # ticks
    for i in range(12):
        ang = i * math.pi/6.0
        x0 = int(cx + (r - 6) * math.cos(ang))
        y0 = int(cy + (r - 6) * math.sin(ang))
        x1 = int(cx + r * math.cos(ang))
        y1 = int(cy + r * math.sin(ang))
        cv2.line(img, (x0, y0), (x1, y1), colors['tick'], 2)

    sec = dt.second + dt.microsecond/1e6
    minf = dt.minute + sec/60.0
    hourf = (dt.hour % 12) + minf/60.0

    # hands
    def hand(angle, length, color, thick):
        x = int(cx + length * math.cos(angle))
        y = int(cy + length * math.sin(angle))
        cv2.line(img, (cx, cy), (x, y), color, thick, cv2.LINE_AA)

    hand(hourf * math.pi/6.0 - math.pi/2, int(r*0.55), colors['hour'], 4)
    hand(minf * math.pi/30.0 - math.pi/2, int(r*0.75), colors['minute'], 3)
    hand(sec * math.pi/30.0 - math.pi/2, int(r*0.85), colors['second'], 2)

    cv2.circle(img, (cx, cy), 3, colors['ring'], -1)
    return img

# =============================
# Core visual primitives (headless)
# =============================
@dataclass
class RenderSettings:
    audio_path: str
    out_dir: str
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    style: str = "bars"
    palette: str = "rainbow"
    background: str = "black"  # black|white|gradient|blur
    quality: str = "high"       # low|medium|high|ultra
    glow: bool = True
    glow_intensity: int = 3
    blur: bool = False
    blur_amount: int = 5
    particles: bool = False
    particle_count: int = 60
    mirror: bool = False
    beat_react: bool = True
    logo_text: Optional[str] = None
    bitrate: str = "192k"
    codec: str = "h264"          # h264|h265|vp9
    draft_start: float = 0.0      # seconds, optional quick preview segment
    draft_duration: Optional[float] = None  # seconds
    # overlays
    progress_bar: bool = False
    progress_pos: str = 'bottom'   # top|bottom
    progress_height: int = 10
    progress_countdown: bool = False
    clock: Optional[str] = None    # 'digital'|'analog'|None
    clock_mode: str = 'elapsed'    # default changed to 'elapsed'
    clock_style: str = 'dark'      # 'dark'|'light'|'retro'|'neon'|'outline'
    clock_pos: str = 'tr'          # 'tl'|'tr'|'bl'|'br'
    clock_scale: float = 1.0


def _get_bg(w: int, h: int, background: str, frame_num: int = 0) -> np.ndarray:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required: pip install opencv-python")
    bg_type = (background or "black").lower()
    if bg_type == "black":
        return np.zeros((h, w, 3), dtype=np.uint8)
    if bg_type == "white":
        return np.full((h, w, 3), 255, dtype=np.uint8)
    gradient = np.linspace(0, 255, h, dtype=np.uint8)
    if bg_type == "gradient":
        bg = np.repeat(gradient[:, None], w, axis=1)
        return cv2.merge([bg, bg // 2, bg // 3])
    if bg_type == "blur":
        shift = int((frame_num * 2) % 255)
        g = np.roll(gradient, shift)
        bg = np.repeat(g[:, None], w, axis=1)
        return cv2.merge([bg // 3, bg // 2, bg])
    return np.zeros((h, w, 3), dtype=np.uint8)


def _apply_glow(img: np.ndarray, enable: bool, intensity: int) -> np.ndarray:
    if not enable:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), max(1, int(intensity)))
    return cv2.addWeighted(img, 0.6, blur, 0.4, 0)


def _apply_blur(img: np.ndarray, enable: bool, blur_amount: int) -> np.ndarray:
    if not enable:
        return img
    k = int(blur_amount) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_mirror(img: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return img
    h = img.shape[0]
    top = img[: h // 2].copy()
    img[h // 2: h // 2 + top.shape[0]] = cv2.flip(top, 0)
    return img


def _add_text(img: np.ndarray, text: Optional[str], beat: bool = False) -> np.ndarray:
    if not text:
        return img
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0 if beat else 1.5
    thick = 3
    size = cv2.getTextSize(text, font, scale, thick)[0]
    x = (w - size[0]) // 2
    y = h - 50
    cv2.putText(img, text, (x+2, y+2), font, scale, (0, 0, 0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def _add_particles(img: np.ndarray, enable: bool, particles: list, target_count: int,
                   palette: str, energy: np.ndarray, beat: bool) -> np.ndarray:
    if not enable:
        return img
    h, w = img.shape[:2]
    while len(particles) < target_count:
        particles.append({
            'x': float(np.random.uniform(0, w)),
            'y': float(np.random.uniform(0, h)),
            'vx': float(np.random.uniform(-2, 2)),
            'vy': float(np.random.uniform(-2, 2)),
            'size': int(np.random.randint(2, 6)),
            'life': int(np.random.randint(40, 140)),
        })
    avg = float(np.mean(energy)) if energy.size else 0.0
    color_fn = Palette.scheme(palette)
    for p in list(particles):
        if beat:
            p['vx'] *= 1.35
            p['vy'] *= 1.35
        p['x'] += p['vx']
        p['y'] += p['vy']
        if p['x'] < 0 or p['x'] >= w:
            p['vx'] *= -1; p['x'] = clamp(p['x'], 0, w-1)
        if p['y'] < 0 or p['y'] >= h:
            p['vy'] *= -1; p['y'] = clamp(p['y'], 0, h-1)
        p['vx'] *= 0.985; p['vy'] *= 0.985
        p['life'] -= 1
        if p['life'] <= 0:
            particles.remove(p)
            continue
        color = color_fn(avg, len(particles), max(1, target_count))
        cv2.circle(img, (int(p['x']), int(p['y'])), p['size'], color, -1)
    return img

# ------------------------- styles

def frame_bars(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = _get_bg(w, h, background)
    bar_count = min(100, len(mel_col))
    bar_w = max(1, w // bar_count)
    color_fn = Palette.scheme(palette)
    scale = 1.25 if beat else 1.0
    for i in range(bar_count):
        f = float(mel_col[i])
        bh = int(f * h * 0.85 * scale)
        x0 = i * bar_w
        x1 = x0 + bar_w - 2
        y0 = h - bh
        color = color_fn(f, i, bar_count)
        cv2.rectangle(img, (x0, y0), (x1, h), color, -1)
    return img


def frame_circle(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = _get_bg(w, h, background)
    cx, cy = w // 2, h // 2
    base = min(w, h) // 4
    color_fn = Palette.scheme(palette)
    scale = 1.35 if beat else 1.0
    n = min(200, len(mel_col))
    for i in range(n):
        f = float(mel_col[i])
        angle = (i / n) * 2 * math.pi
        radius = base + int(f * 220 * scale)
        x = int(cx + math.cos(angle) * radius)
        y = int(cy + math.sin(angle) * radius)
        color = color_fn(f, i, n)
        cv2.circle(img, (x, y), 4 if beat else 3, color, -1)
    return img


def frame_wave(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = _get_bg(w, h, background)
    color_fn = Palette.scheme(palette)
    n = len(mel_col)
    xs = np.linspace(0, w-1, n).astype(int)
    ys = (h/2 + (mel_col - 0.5) * h * (0.6 if not beat else 0.8)).astype(int)
    for i in range(1, n):
        f = float(mel_col[i])
        color = color_fn(f, i, n)
        cv2.line(img, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 2)
    return img


def frame_spectrum(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = _get_bg(w, h, background)
    color_fn = Palette.scheme(palette)
    n = len(mel_col)
    for i in range(n):
        f = float(mel_col[i])
        y = int((1.0 - f) * (h-1))
        color = color_fn(f, i, n)
        cv2.line(img, (i * w // n, h-1), (i * w // n, y), color, 1)
    return img


def frame_dual(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = frame_bars(mel_col, w, h, palette, background, beat)
    overlay = frame_circle(mel_col, w, h, palette, background, beat)
    return cv2.addWeighted(img, 0.6, overlay, 0.6, 0)


def frame_spiral(mel_col: np.ndarray, w: int, h: int, palette: str, background: str, beat: bool) -> np.ndarray:
    img = _get_bg(w, h, background)
    color_fn = Palette.scheme(palette)
    cx, cy = w // 2, h // 2
    n = len(mel_col)
    twist = 5.0
    scale = 1.4 if beat else 1.0
    for i, f in enumerate(mel_col):
        a = i / n * 2 * math.pi * twist
        r = (i / n) * (min(w, h) * 0.45) + float(f) * 60 * scale
        x = int(cx + math.cos(a) * r)
        y = int(cy + math.sin(a) * r)
        color = color_fn(float(f), i, n)
        cv2.circle(img, (x, y), 2, color, -1)
    return img

STYLE_REGISTRY = {
    'bars': frame_bars,
    'circle': frame_circle,
    'wave': frame_wave,
    'spectrum': frame_spectrum,
    'dual': frame_dual,
    'spiral': frame_spiral,
}

# ------------------------- render orchestration (headless)

def render_video(settings: RenderSettings) -> str:
    if not (CV2_AVAILABLE and LIBROSA_AVAILABLE):
        missing = [name for name, ok in [("cv2", CV2_AVAILABLE), ("librosa", LIBROSA_AVAILABLE)] if not ok]
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")

    width, height = settings.resolution
    fps = max(1, int(settings.fps))
    base = os.path.splitext(os.path.basename(settings.audio_path))[0]
    out_dir = settings.out_dir or os.path.dirname(settings.audio_path)
    os.makedirs(out_dir, exist_ok=True)

    # Optional draft window
    y, sr = librosa.load(settings.audio_path, sr=22050, mono=True)
    total_audio_len = float(len(y)) / sr
    if settings.draft_duration and settings.draft_duration > 0:
        start = max(0.0, settings.draft_start)
        end = min(total_audio_len, start + float(settings.draft_duration))
        y = y[int(start*sr): int(end*sr)]
    analysis = AudioAnalysis(path=None, sr=22050, fps=fps, n_fft=2048, n_mels=128, y=y)

    total_frames = analysis.n_frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_video_path = os.path.join(tempfile.gettempdir(), f"{base}_render_tmp.mp4")
    out = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():  # fallback to a different fourcc
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():  # pragma: no cover
        raise RuntimeError("Failed to open video writer. Check codec/permissions.")

    # Quality supersampling
    ss = 2 if settings.quality.lower() == 'ultra' else 1
    W, H = width * ss, height * ss

    renderer = STYLE_REGISTRY.get(settings.style, frame_bars)
    particles = []

    # choose a color for progress bar derived from palette midpoint
    progress_color = Palette.scheme(settings.palette)(0.8, 50, 100)

    for idx in range(total_frames):
        mel_col, (_, _, _), beat = analysis.frame_features(idx)
        if not settings.beat_react:
            beat = False
        frame = renderer(mel_col, W, H, settings.palette, settings.background, beat)
        frame = _apply_glow(frame, settings.glow, settings.glow_intensity)
        frame = _apply_blur(frame, settings.blur, settings.blur_amount)
        frame = _add_particles(frame, settings.particles, particles, settings.particle_count,
                               settings.palette, mel_col, beat)
        frame = _apply_mirror(frame, settings.mirror)
        frame = _add_text(frame, settings.logo_text, beat)

        # overlays
        # Clock (digital default) — always HH:MM:SS per request
        if settings.clock:
            if settings.clock_mode == 'elapsed':
                elapsed = (idx / fps) + max(0.0, settings.draft_start)
                if settings.clock == 'digital':
                    frame = _overlay_digital_clock(frame, _format_hms_full(elapsed), settings.clock_pos, settings.clock_style, settings.clock_scale)
                else:
                    base_dt = _dt.datetime(2000,1,1) + _dt.timedelta(seconds=float(elapsed))
                    frame = _overlay_analog_clock(frame, base_dt, settings.clock_pos, settings.clock_style, settings.clock_scale)
            else:  # system time
                now = _dt.datetime.now()
                if settings.clock == 'digital':
                    frame = _overlay_digital_clock(frame, now.strftime('%H:%M:%S'), settings.clock_pos, settings.clock_style, settings.clock_scale)
                else:
                    frame = _overlay_analog_clock(frame, now, settings.clock_pos, settings.clock_style, settings.clock_scale)

        # Progress bar + optional countdown label
        if settings.progress_bar:
            progress = (idx + 1) / max(1, total_frames)
            label = None
            if settings.progress_countdown:
                elapsed = (idx + 1) / fps
                remaining = max(0.0, analysis.n_frames / fps - elapsed)
                label = f"-{_format_hms_full(remaining)}"
            frame = _overlay_progress_bar(frame, progress, settings.progress_pos, settings.progress_height, progress_color, label)

        if ss != 1:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(frame)
        # Simple progress
        if idx % max(1, fps) == 0:
            print(f"Progress: {idx+1}/{total_frames} ({(idx+1)/max(1,total_frames)*100:.1f}%)", flush=True)

    out.release()

    # Final mux with audio
    out_path = os.path.join(out_dir, f"{base}_visual.mp4")
    vcodec_map = {
        'h264': 'libx264',
        'h265': 'libx265',
        'vp9': 'libvpx-vp9',
    }
    vcodec = vcodec_map.get(settings.codec.lower(), 'libx264')
    bitrate = settings.bitrate

    cmd = [
        'ffmpeg', '-y',
        '-i', tmp_video_path,
        '-i', settings.audio_path,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', vcodec,
        '-crf', '18' if vcodec in ('libx264', 'libx265') else '30',
        '-preset', 'medium',
        '-c:a', 'aac' if vcodec in ('libx264', 'libx265') else 'libopus',
        '-b:a', bitrate,
        '-shortest',
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        # Fallback: video-only copy
        print("ffmpeg not available or failed; exporting video-only.")
        out_path = os.path.join(out_dir, f"{base}_visual_video_only.mp4")
        os.replace(tmp_video_path, out_path)
    else:
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass

    print(f"Saved: {out_path}")
    return out_path

# =============================
# Optional GUI (only if Tkinter exists)
# =============================
if TK_AVAILABLE:
    _PLATFORM = sys.platform

    class ScrollableFrame(tk.Frame):
        def __init__(self, container, *args, **kwargs):
            super().__init__(container, *args, **kwargs)
            self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
            self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = tk.Frame(self.canvas)
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            )
            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            # Mousewheel across platforms
            if _PLATFORM.startswith("win"):
                self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
            elif _PLATFORM == "darwin":
                self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_macos)
            else:
                self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
                self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        def _on_mousewheel_windows(self, event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_macos(self, event):
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        def _on_mousewheel_linux(self, event):
            direction = -1 if event.num == 4 else 1
            self.canvas.yview_scroll(direction, "units")

    class AudioVisualizerApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Advanced Audio Visualizer Pro v2 (GUI)")
            try:
                self.root.tk.call('tk', 'scaling', self.root.winfo_fpixels('1i') / 72.0)
            except Exception:
                pass
            self.root.geometry("800x860")
            self.file_path: Optional[str] = None
            self.stop_event = Event()
            self.scrollable = ScrollableFrame(root)
            self.scrollable.pack(fill="both", expand=True, padx=6, pady=6)
            container = self.scrollable.scrollable_frame

            title_label = tk.Label(container, text="🎵 Аудио Визуализатор Pro 2.0", font=("Arial", 19, "bold"))
            title_label.pack(pady=12)

            # File row
            file_row = tk.Frame(container)
            file_row.pack(fill="x", padx=12)
            self.button_open = tk.Button(file_row, text="📁 Открыть аудиофайл", command=self.load_file, bg="#4CAF50", fg="white")
            self.button_open.pack(side="left")
            self.status = tk.Label(file_row, text="Файл не выбран", fg="gray")
            self.status.pack(side="left", padx=8)

            # helper for combos
            def combo(frame, label, values, default, row):
                tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
                var = tk.StringVar(value=default)
                ttk.Combobox(frame, textvariable=var, values=values, state="readonly", width=16).grid(row=row, column=1, padx=6)
                return var

            main = tk.LabelFrame(container, text="⚙️ Основные настройки", padx=16, pady=12)
            main.pack(pady=10, padx=12, fill="x")
            self.resolution_var = tk.StringVar(value="1280x720")
            tk.Label(main, text="Разрешение:").grid(row=0, column=0, sticky="w")
            ttk.Combobox(main, textvariable=self.resolution_var,
                         values=["640x480","1280x720","1920x1080","2560x1440","3840x2160"], state="readonly", width=16).grid(row=0, column=1, padx=6)
            tk.Label(main, text="FPS:").grid(row=1, column=0, sticky="w")
            self.fps_var = tk.IntVar(value=30)
            tk.Spinbox(main, from_=15, to=60, textvariable=self.fps_var, width=18).grid(row=1, column=1, padx=6)
            self.style_var = combo(main, "Стиль:", list(STYLE_REGISTRY.keys()), "bars", 2)
            self.palette_var = combo(main, "Палитра:", ["rainbow","fire","ocean","neon","purple","gradient"], "rainbow", 3)

            effects = tk.LabelFrame(container, text="✨ Эффекты", padx=16, pady=12)
            effects.pack(pady=10, padx=12, fill="x")
            self.glow_var = tk.BooleanVar(value=True)
            tk.Checkbutton(effects, text="Свечение", variable=self.glow_var).grid(row=0, column=0, sticky="w")
            tk.Label(effects, text="Интенсивность:").grid(row=0, column=1, padx=(10, 4))
            self.glow_intensity = tk.IntVar(value=3)
            tk.Scale(effects, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.glow_intensity, length=130).grid(row=0, column=2)
            self.blur_var = tk.BooleanVar(value=False)
            tk.Checkbutton(effects, text="Размытие", variable=self.blur_var).grid(row=1, column=0, sticky="w")
            tk.Label(effects, text="Сила:").grid(row=1, column=1, padx=(10, 4))
            self.blur_amount = tk.IntVar(value=5)
            tk.Scale(effects, from_=1, to=15, orient=tk.HORIZONTAL, variable=self.blur_amount, length=130).grid(row=1, column=2)
            self.particles_var = tk.BooleanVar(value=False)
            tk.Checkbutton(effects, text="Частицы", variable=self.particles_var).grid(row=2, column=0, sticky="w")
            tk.Label(effects, text="Количество:").grid(row=2, column=1, padx=(10, 4))
            self.particle_count = tk.IntVar(value=60)
            tk.Scale(effects, from_=10, to=250, orient=tk.HORIZONTAL, variable=self.particle_count, length=130).grid(row=2, column=2)
            self.mirror_var = tk.BooleanVar(value=False)
            tk.Checkbutton(effects, text="Зеркало", variable=self.mirror_var).grid(row=3, column=0, sticky="w", columnspan=3)
            self.beat_react_var = tk.BooleanVar(value=True)
            tk.Checkbutton(effects, text="Реакция на биты", variable=self.beat_react_var).grid(row=4, column=0, sticky="w", columnspan=3)

            overlays = tk.LabelFrame(container, text="🕒 Оверлеи", padx=16, pady=12)
            overlays.pack(pady=10, padx=12, fill="x")

            # progress bar
            self.progress_bar_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Полоса прогресса", variable=self.progress_bar_var).grid(row=0, column=0, sticky="w")
            self.progress_pos_var = combo(overlays, "Позиция бара:", ["bottom","top"], "bottom", 0)
            tk.Label(overlays, text="Высота бара:").grid(row=1, column=0, sticky="w")
            self.progress_height_var = tk.IntVar(value=10)
            tk.Spinbox(overlays, from_=2, to=40, textvariable=self.progress_height_var, width=8).grid(row=1, column=1, sticky="w")
            self.progress_countdown_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Обратный отсчёт у бара", variable=self.progress_countdown_var).grid(row=1, column=2, padx=8, sticky="w")

            # clock (default: digital + elapsed)
            self.clock_enable_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Часы", variable=self.clock_enable_var).grid(row=2, column=0, sticky="w")
            self.clock_type_var = combo(overlays, "Тип часов:", ["digital","analog"], "digital", 2)
            self.clock_mode_var = combo(overlays, "Режим:", ["elapsed","system"], "elapsed", 3)
            self.clock_style_var = combo(overlays, "Стиль:", ["dark","light","retro","neon","outline"], "dark", 4)
            self.clock_pos_var = combo(overlays, "Позиция:", ["tl","tr","bl","br"], "tr", 5)
            tk.Label(overlays, text="Масштаб:").grid(row=6, column=0, sticky="w")
            self.clock_scale_var = tk.DoubleVar(value=1.0)
            tk.Spinbox(overlays, from_=0.5, to=2.0, increment=0.1, textvariable=self.clock_scale_var, width=8).grid(row=6, column=1, sticky="w")

            extra = tk.LabelFrame(container, text="🎬 Дополнительно", padx=16, pady=12)
            extra.pack(pady=10, padx=12, fill="x")
            self.background_var = combo(extra, "Фон:", ["black","white","gradient","blur"], "black", 0)
            self.quality_var = combo(extra, "Качество:", ["low","medium","high","ultra"], "high", 1)
            self.logo_var = tk.BooleanVar(value=False)
            tk.Checkbutton(extra, text="Текст/логотип", variable=self.logo_var).grid(row=2, column=0, sticky="w")
            self.logo_text = tk.Entry(extra, width=22)
            self.logo_text.grid(row=2, column=1, pady=4)
            self.logo_text.insert(0, "My Track")
            tk.Label(extra, text="Аудио битрейт:").grid(row=3, column=0, sticky="w")
            self.bitrate_var = tk.StringVar(value="192k")
            ttk.Combobox(extra, textvariable=self.bitrate_var, values=["128k","192k","256k","320k"], state="readonly", width=16).grid(row=3, column=1)
            tk.Label(extra, text="Видео кодек:").grid(row=4, column=0, sticky="w")
            self.codec_var = tk.StringVar(value="h264")
            ttk.Combobox(extra, textvariable=self.codec_var, values=["h264","h265","vp9"], state="readonly", width=16).grid(row=4, column=1)

            out_row = tk.Frame(container)
            out_row.pack(fill="x", padx=12, pady=6)
            tk.Label(out_row, text="Папка вывода:").pack(side="left")
            self.out_dir = tk.StringVar(value="")
            tk.Entry(out_row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=6)
            tk.Button(out_row, text="Выбрать…", command=self.choose_out_dir).pack(side="left")

            bottom = tk.Frame(root)
            bottom.pack(side="bottom", fill="x", padx=10, pady=10)
            self.progress = ttk.Progressbar(bottom, mode='determinate', length=520)
            self.progress.pack(pady=4)
            self.progress_label = tk.Label(bottom, text="", fg="blue")
            self.progress_label.pack()

            btns = tk.Frame(bottom)
            btns.pack()
            self.button_render = tk.Button(btns, text="🎬 Создать видео", command=self.start_render, state=tk.DISABLED, bg="#2196F3", fg="white")
            self.button_render.pack(side="left", padx=6)

        def choose_out_dir(self):
            d = filedialog.askdirectory()
            if d:
                self.out_dir.set(d)

        def load_file(self):
            path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac")])
            if not path:
                return
            self.file_path = path
            filename = os.path.basename(path)
            self.status.config(text=f"✓ Выбран: {filename}", fg="green")
            self.button_render.config(state=tk.NORMAL)

        def start_render(self):
            if not self.file_path:
                messagebox.showerror("Ошибка", "Сначала выберите аудиофайл.")
                return
            w, h = map(int, self.resolution_var.get().split('x'))
            settings = RenderSettings(
                audio_path=self.file_path,
                out_dir=self.out_dir.get().strip() or os.path.dirname(self.file_path),
                resolution=(w, h),
                fps=int(self.fps_var.get()),
                style=self.style_var.get(),
                palette=self.palette_var.get(),
                background=self.background_var.get(),
                quality=self.quality_var.get(),
                glow=self.glow_var.get(),
                glow_intensity=int(self.glow_intensity.get()),
                blur=self.blur_var.get(),
                blur_amount=int(self.blur_amount.get()),
                particles=self.particles_var.get(),
                particle_count=int(self.particle_count.get()),
                mirror=self.mirror_var.get(),
                beat_react=self.beat_react_var.get(),
                logo_text=(self.logo_text.get().strip() if self.logo_var.get() else None),
                bitrate=self.bitrate_var.get(),
                codec=self.codec_var.get(),
                progress_bar=self.progress_bar_var.get(),
                progress_pos=self.progress_pos_var.get(),
                progress_height=int(self.progress_height_var.get()),
                progress_countdown=self.progress_countdown_var.get(),
                clock=(self.clock_type_var.get() if self.clock_enable_var.get() else None),
                clock_mode=self.clock_mode_var.get(),
                clock_style=self.clock_style_var.get(),
                clock_pos=self.clock_pos_var.get(),
                clock_scale=float(self.clock_scale_var.get()),
            )
            self.progress_label.config(text="Рендер…")
            self.button_render.config(state=tk.DISABLED)
            def _bg():
                try:
                    out_path = render_video(settings)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                else:
                    self.root.after(0, lambda: messagebox.showinfo("Готово", f"Сохранено:\n{out_path}"))
                finally:
                    self.root.after(0, lambda: (self.progress_label.config(text=""), self.button_render.config(state=tk.NORMAL)))
            Thread(target=_bg, daemon=True).start()

# =============================
# CLI
# =============================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audio Visualizer Pro — CLI/GUI hybrid")
    p.add_argument('--audio', help='Path to input audio file (.mp3/.wav/…)')
    p.add_argument('--out-dir', default='', help='Output directory (default: alongside audio)')
    p.add_argument('--resolution', default='1280x720', help='WIDTHxHEIGHT, e.g. 1920x1080')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--style', default='bars', choices=list(STYLE_REGISTRY.keys()))
    p.add_argument('--palette', default='rainbow', choices=['rainbow','fire','ocean','neon','purple','gradient'])
    p.add_argument('--background', default='black', choices=['black','white','gradient','blur'])
    p.add_argument('--quality', default='high', choices=['low','medium','high','ultra'])
    p.add_argument('--glow', action='store_true')
    p.add_argument('--glow-intensity', type=int, default=3)
    p.add_argument('--blur', action='store_true')
    p.add_argument('--blur-amount', type=int, default=5)
    p.add_argument('--particles', action='store_true')
    p.add_argument('--particle-count', type=int, default=60)
    p.add_argument('--mirror', action='store_true')
    p.add_argument('--no-beat-react', dest='beat_react', action='store_false', help='Disable beat reaction')
    p.set_defaults(beat_react=True)
    p.add_argument('--logo-text', default=None)
    p.add_argument('--bitrate', default='192k')
    p.add_argument('--codec', default='h264', choices=['h264','h265','vp9'])
    p.add_argument('--draft-start', type=float, default=0.0, help='Draft render start (sec)')
    p.add_argument('--draft-duration', type=float, default=None, help='Draft render duration (sec)')
    # overlays
    p.add_argument('--progress-bar', action='store_true', help='Enable video progress bar overlay')
    p.add_argument('--progress-pos', default='bottom', choices=['top','bottom'])
    p.add_argument('--progress-height', type=int, default=10)
    p.add_argument('--progress-countdown', action='store_true', help='Show reverse countdown near the progress bar')
    p.add_argument('--clock', default='digital', choices=['digital','analog', 'none'])
    p.add_argument('--clock-mode', default='elapsed', choices=['system','elapsed'])
    p.add_argument('--clock-style', default='dark', choices=['dark','light','retro','neon','outline'])
    p.add_argument('--clock-pos', default='tr', choices=['tl','tr','bl','br'])
    p.add_argument('--clock-scale', type=float, default=1.0)

    p.add_argument('--run-tests', action='store_true', help='Run unit tests and exit')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_tests:
        import unittest
        # run tests in this module only
        unittest.main(module=__name__, argv=[sys.argv[0]], exit=False)
        return

    # If no audio provided and Tk available — launch GUI
    if not args.audio and TK_AVAILABLE:
        root = tk.Tk()
        app = AudioVisualizerApp(root)
        root.mainloop()
        return

    if not args.audio:
        print("Error: --audio is required in headless mode (Tkinter not available)")
        print("Tip: run with --run-tests to verify environment, or install Tkinter to use the GUI.")
        sys.exit(2)

    try:
        w, h = map(int, args.resolution.lower().split('x'))
    except Exception:
        print("Invalid --resolution. Use WIDTHxHEIGHT, e.g. 1280x720")
        sys.exit(2)

    # map 'none' to None
    clock_val = None if (args.clock == 'none') else args.clock

    settings = RenderSettings(
        audio_path=args.audio,
        out_dir=args.out_dir,
        resolution=(w, h),
        fps=int(args.fps),
        style=args.style,
        palette=args.palette,
        background=args.background,
        quality=args.quality,
        glow=bool(args.glow),
        glow_intensity=int(args.glow_intensity),
        blur=bool(args.blur),
        blur_amount=int(args.blur_amount),
        particles=bool(args.particles),
        particle_count=int(args.particle_count),
        mirror=bool(args.mirror),
        beat_react=bool(args.beat_react),
        logo_text=(args.logo_text if args.logo_text else None),
        bitrate=args.bitrate,
        codec=args.codec,
        draft_start=float(args.draft_start),
        draft_duration=(float(args.draft_duration) if args.draft_duration is not None else None),
        progress_bar=bool(args.progress_bar),
        progress_pos=args.progress_pos,
        progress_height=int(args.progress_height),
        progress_countdown=bool(args.progress_countdown),
        clock=clock_val,
        clock_mode=args.clock_mode,
        clock_style=args.clock_style,
        clock_pos=args.clock_pos,
        clock_scale=float(args.clock_scale),
    )
    try:
        render_video(settings)
    except Exception as e:
        print(f"Render failed: {e}")
        sys.exit(1)

# =============================
# Tests
# =============================
import unittest

class TestUtils(unittest.TestCase):
    def test_clamp_basic(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_palette_range(self):
        fn = Palette.scheme('rainbow')
        bgr = fn(0.5, 5, 10)
        self.assertEqual(len(bgr), 3)
        self.assertTrue(all(0 <= c <= 255 for c in bgr))

    def test_format_hms_full(self):
        self.assertEqual(_format_hms_full(5), "00:00:05")
        self.assertEqual(_format_hms_full(65), "00:01:05")
        self.assertEqual(_format_hms_full(3661), "01:01:01")

@unittest.skipUnless(CV2_AVAILABLE, "cv2 not installed")
class TestFrames(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 320, 180
        self.mel = np.clip(np.linspace(0, 1, 128), 0, 1)
    def test_bars_shape(self):
        img = frame_bars(self.mel, self.w, self.h, 'rainbow', 'black', False)
        self.assertEqual(img.shape, (self.h, self.w, 3))
    def test_spiral_shape(self):
        img = frame_spiral(self.mel, self.w, self.h, 'ocean', 'gradient', True)
        self.assertEqual(img.shape, (self.h, self.w, 3))
    def test_progress_overlay(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out = _overlay_progress_bar(img.copy(), 0.5, 'bottom', 8)
        # bottom row should not be all zeros after overlay
        self.assertTrue(out[-10:].sum() > 0)
    def test_progress_countdown_label(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out = _overlay_progress_bar(img.copy(), 0.5, 'bottom', 8, (255,255,255), label='-00:10:00')
        self.assertEqual(out.shape, (self.h, self.w, 3))
    def test_digital_clock_overlay(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out = _overlay_digital_clock(img.copy(), '12:34:56', 'tr', 'dark', 1.0)
        self.assertEqual(out.shape, (self.h, self.w, 3))
    def test_analog_clock_overlay(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        now = _dt.datetime(2024, 1, 1, 12, 34, 56)
        out = _overlay_analog_clock(img.copy(), now, 'tl', 'light', 1.0)
        self.assertEqual(out.shape, (self.h, self.w, 3))

@unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
class TestAudioAnalysis(unittest.TestCase):
    def test_analysis_hop_and_frames(self):
        sr = 22050
        t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
        y = 0.2*np.sin(2*math.pi*440*t)
        aa = AudioAnalysis(path=None, sr=sr, fps=25, y=y)
        self.assertGreater(aa.n_frames, 0)
        self.assertEqual(aa.hop_length, max(1, int(sr/25)))

if __name__ == '__main__':
    main()
