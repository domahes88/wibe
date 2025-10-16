"""
Audio Visualizer Pro — Headless/GUI Hybrid
Fix: gracefully handle environments without Tkinter by providing a CLI (headless) mode.

New in this build:
- Video **progress bar** overlay (shows track progress), position + size configurable.
- **Clock overlays**: digital and analog. Modes: system time or track elapsed time. Multiple styles and positions.
- **Reverse countdown** near the progress bar; digital clock defaults to HH:MM:SS and **elapsed** mode by default.
- **Batch processing** — render **multiple audio files** and (optionally) **merge into one video** with **chapters** (bookmarks) named after filenames.
- **Gap / Slate between tracks** when merging — configurable duration and label format (e.g. “Next: <title>”).
- **NEW**: More robust merging (no missing tracks with same basenames), proper ffmpeg-concat escaping.
- **NEW**: Parallel batch rendering (per-file) with configurable workers.
- **NEW**: Memory-friendly streaming & cleanup options for long batches.
- **NEW**: Logo overlay (image) with time window, position, scale, opacity, and optional motion along progress.

Usage (CLI):
  # Single file
  python audio_visualizer.py --audio song1.mp3 --style bars --palette rainbow \
      --resolution 1280x720 --fps 30 --quality high --codec h264 --bitrate 192k \
      --progress-bar --progress-countdown --clock digital --clock-mode elapsed

  # Multiple files → one video with chapters and slates between tracks
  python audio_visualizer.py --audio song1.mp3 song2.mp3 song3.wav \
      --merge --chapters --gap-duration 1.0 --gap-slate --gap-label-format "Next: {title}" \
      --workers 3 --cleanup-intermediates --out-dir out/

  # Multiple files → render separately (no merge) with parallelism
  python audio_visualizer.py --audio folder/*.mp3 --no-merge --workers 4

  # Logo overlay example (shows from 3s to 15s at top-right, 60% opacity, 0.5x scale)
  python audio_visualizer.py --audio song.mp3 --logo-img logo.png --logo-pos tr --logo-opacity 0.6 \
      --logo-scale 0.5 --logo-start 3 --logo-end 15 --progress-bar

Run tests:
  python audio_visualizer.py --run-tests

Notes:
- If Tkinter is available, you still get the GUI. Otherwise, the CLI runs.
- ffmpeg is required for final audio+video mux; the script falls back to a video-only file if ffmpeg is not found for single segments. For merging/chapters ffmpeg is required.
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import tempfile
import glob
import subprocess
import datetime as _dt
import hashlib
import gc
from dataclasses import dataclass
from threading import Thread, Event
from typing import Callable, Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

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


def _escape_ffconcat_path(p: str) -> str:
    """Escape single quotes for ffmpeg concat demuxer line: file '<path>'"""
    return p.replace("'", "'\\''")


def _unique_out_path(dir_: str, base_name: str, suffix: str) -> str:
    """Ensure no overwrites when inputs share the same basename.
    Returns a path like dir_/f"{base}_{suffix}.mp4"; if exists, adds _1, _2...
    """
    base, ext = os.path.splitext(base_name)
    candidate = os.path.join(dir_, f"{base}_{suffix}{ext}")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        cand = os.path.join(dir_, f"{base}_{suffix}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

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
            y_f, self.sr = librosa.load(path, sr=sr, mono=True)
            self.y = y_f.astype(np.float32, copy=False)
        else:
            self.y = y.astype(np.float32, copy=False)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))

        # Hop aligned to FPS
        self.hop_length = max(1, int(self.sr / max(1, self.fps)))

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / max(1e-9, (S_db.max() - S_db.min()))
        self.mels = S_db.astype(np.float32, copy=False)  # (n_mels, n_frames)
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
# Overlays (progress bar + clocks + logo)
# =============================

def _overlay_progress_bar(img: np.ndarray, progress: float, position: str = 'bottom', height: int = 10,
                          color: Tuple[int, int, int] = (255, 255, 255), label: Optional[str] = None) -> np.ndarray:
    h, w = img.shape[:2]
    height = max(2, int(height))
    pad = 6
    y0 = (h - height - pad) if position == 'bottom' else pad
    y1 = y0 + height

    bg = img.copy()
    cv2.rectangle(bg, (pad, y0), (w - pad, y1), (0, 0, 0), -1)
    img = cv2.addWeighted(bg, 0.35, img, 0.65, 0)

    x1 = pad + int((w - 2 * pad) * clamp(progress, 0.0, 1.0))
    cv2.rectangle(img, (pad, y0), (x1, y1), color, -1)
    cv2.rectangle(img, (x1-2, y0), (x1+2, y1), (240, 240, 240), -1)

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
    return dict(face=(20,20,20), ring=(200,200,200), tick=(200,200,200), hour=(240,240,240), minute=(200,200,200), second=(60,200,255), text=(235,235,235))


def _place_rect(w: int, h: int, rect_w: int, rect_h: int, pos: str, margin: int = 18) -> Tuple[int, int]:
    pos = (pos or 'tr').lower()
    if pos == 'tl':
        return margin, margin
    if pos == 'tr':
        return w - rect_w - margin, margin
    if pos == 'bl':
        return margin, h - rect_h - margin
    return w - rect_w - margin, h - rect_h - margin


def _overlay_digital_clock(img: np.ndarray, text: str, pos: str, style: str, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    colors = _clock_style_colors(style)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8 * max(0.4, float(scale))
    thickness = 2
    size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = _place_rect(w, h, size[0] + 20, size[1] + 16, pos)
    if colors.get('face') is not None:
        cv2.rectangle(img, (x, y), (x + size[0] + 20, y + size[1] + 16), colors['face'], -1)
    cv2.rectangle(img, (x, y), (x + size[0] + 20, y + size[1] + 16), colors['ring'], 1)
    tx, ty = x + 10, y + size[1] + 4
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
    def hand(angle, length, color, thick):
        x = int(cx + length * math.cos(angle))
        y = int(cy + length * math.sin(angle))
        cv2.line(img, (cx, cy), (x, y), color, thick, cv2.LINE_AA)
    hand(hourf * math.pi/6.0 - math.pi/2, int(r*0.55), colors['hour'], 4)
    hand(minf * math.pi/30.0 - math.pi/2, int(r*0.75), colors['minute'], 3)
    hand(sec * math.pi/30.0 - math.pi/2, int(r*0.85), colors['second'], 2)
    cv2.circle(img, (cx, cy), 3, colors['ring'], -1)
    return img

# ------------------------- Logo overlay helpers

def _overlay_logo(img: np.ndarray, logo_rgba: Optional[np.ndarray], pos: str, scale: float, opacity: float,
                  progress: float, follow_progress: bool) -> np.ndarray:
    if logo_rgba is None:
        return img
    h, w = img.shape[:2]
    # Resize logo
    lh, lw = logo_rgba.shape[:2]
    target_w = int(min(w, h) * max(0.1, min(2.0, float(scale))) )
    new_w = max(1, target_w)
    new_h = max(1, int(lh * (new_w / lw)))
    logo = cv2.resize(logo_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Compute position
    if follow_progress:
        # move along bottom bar from left to right
        x = int((w - new_w) * clamp(progress, 0.0, 1.0))
        y = h - new_h - 18
    else:
        x, y = _place_rect(w, h, new_w, new_h, pos)
    # Split channels, handle alpha
    if logo.shape[2] == 4:
        b,g,r,a = cv2.split(logo)
        a = (a.astype(np.float32)/255.0) * float(opacity)
        overlay = cv2.merge([b,g,r])
    else:
        overlay = logo
        a = np.full((logo.shape[0], logo.shape[1]), float(opacity), dtype=np.float32)
    # Blend
    roi = img[y:y+overlay.shape[0], x:x+overlay.shape[1]].astype(np.float32)
    if roi.shape[0] <= 0 or roi.shape[1] <= 0:
        return img
    ov = overlay.astype(np.float32)
    for c in range(3):
        roi[:,:,c] = ov[:,:,c] * a + roi[:,:,c] * (1 - a)
    img[y:y+overlay.shape[0], x:x+overlay.shape[1]] = np.clip(roi, 0, 255).astype(np.uint8)
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
    # Text logo (legacy)
    logo_text: Optional[str] = None
    # Image logo (new)
    logo_img_path: Optional[str] = None
    logo_pos: str = 'tr'
    logo_scale: float = 1.0
    logo_opacity: float = 0.8
    logo_start: float = 0.0
    logo_end: Optional[float] = None
    logo_follow_progress: bool = False

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
    clock_mode: str = 'elapsed'    # default 'elapsed'
    clock_style: str = 'dark'      # 'dark'|'light'|'retro'|'neon'|'outline'
    clock_pos: str = 'tr'          # 'tl'|'tr'|'bl'|'br'
    clock_scale: float = 1.0

    # Batch parallelism
    workers: int = 1
    cleanup_intermediates: bool = False


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

# ------------------------- render orchestration (single segment)

def render_video(settings: RenderSettings) -> str:
    if not (CV2_AVAILABLE and LIBROSA_AVAILABLE):
        missing = [name for name, ok in [("cv2", CV2_AVAILABLE), ("librosa", LIBROSA_AVAILABLE)] if not ok]
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")

    width, height = settings.resolution
    fps = max(1, int(settings.fps))
    base_name = os.path.splitext(os.path.basename(settings.audio_path))[0]
    out_dir = settings.out_dir or os.path.dirname(settings.audio_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load audio (float32) and optionally trim for draft
    y, sr = librosa.load(settings.audio_path, sr=22050, mono=True)
    y = y.astype(np.float32, copy=False)
    total_audio_len = float(len(y)) / sr
    if settings.draft_duration and settings.draft_duration > 0:
        start = max(0.0, settings.draft_start)
        end = min(total_audio_len, start + float(settings.draft_duration))
        y = y[int(start*sr): int(end*sr)]
    analysis = AudioAnalysis(path=None, sr=22050, fps=fps, n_fft=2048, n_mels=128, y=y)

    total_frames = analysis.n_frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # unique temp and output names (avoid collisions for same basenames)
    hash6 = hashlib.sha1(os.path.abspath(settings.audio_path).encode('utf-8')).hexdigest()[:6]
    tmp_video_path = os.path.join(tempfile.gettempdir(), f"{base_name}_{hash6}_tmp.mp4")
    out_base_suggest = f"{base_name}_visual.mp4"
    out_path = _unique_out_path(out_dir, out_base_suggest, hash6)

    out = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(tmp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():  # pragma: no cover
        raise RuntimeError("Failed to open video writer. Check codec/permissions.")

    # Quality supersampling
    ss = 2 if settings.quality.lower() == 'ultra' else 1
    W, H = width * ss, height * ss

    renderer = STYLE_REGISTRY.get(settings.style, frame_bars)
    particles = []

    # prepare logo once
    logo_rgba = None
    if settings.logo_img_path and os.path.isfile(settings.logo_img_path):
        logo_rgba = cv2.imread(settings.logo_img_path, cv2.IMREAD_UNCHANGED)

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
        if settings.clock:
            if settings.clock_mode == 'elapsed':
                elapsed = (idx / fps) + max(0.0, settings.draft_start)
                if settings.clock == 'digital':
                    frame = _overlay_digital_clock(frame, _format_hms_full(elapsed), settings.clock_pos, settings.clock_style, settings.clock_scale)
                else:
                    base_dt = _dt.datetime(2000,1,1) + _dt.timedelta(seconds=float(elapsed))
                    frame = _overlay_analog_clock(frame, base_dt, settings.clock_pos, settings.clock_style, settings.clock_scale)
            else:
                now = _dt.datetime.now()
                if settings.clock == 'digital':
                    frame = _overlay_digital_clock(frame, now.strftime('%H:%M:%S'), settings.clock_pos, settings.clock_style, settings.clock_scale)
                else:
                    frame = _overlay_analog_clock(frame, now, settings.clock_pos, settings.clock_style, settings.clock_scale)

        # logo image time-window & motion
        if settings.logo_img_path:
            elapsed = (idx / fps) + max(0.0, settings.draft_start)
            if (elapsed >= settings.logo_start) and (settings.logo_end is None or elapsed <= settings.logo_end):
                frame = _overlay_logo(frame, logo_rgba, settings.logo_pos, settings.logo_scale, settings.logo_opacity,
                                      progress=(idx+1)/max(1,total_frames), follow_progress=bool(settings.logo_follow_progress))

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
        if idx % max(1, fps) == 0:
            print(f"Progress: {idx+1}/{total_frames} ({(idx+1)/max(1,total_frames)*100:.1f}%)", flush=True)

    out.release()

    # Final mux with audio
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
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass
    except Exception:
        print("ffmpeg not available or failed; exporting video-only.")
        # fall back to video-only unique path
        try:
            os.replace(tmp_video_path, out_path)
        except Exception:
            pass

    # Free heavy arrays sooner
    del analysis, y
    gc.collect()

    print(f"Saved: {out_path}")
    return out_path

# ------------------------- batch helpers & chapters/slates

def _ffmetadata_for_chapters(titles: List[str], durations_sec: List[float]) -> str:
    assert len(titles) == len(durations_sec) and len(titles) > 0
    lines = [";FFMETADATA1"]
    time_ms = 0
    for i, (title, dur) in enumerate(zip(titles, durations_sec)):
        start = int(round(time_ms))
        end = int(round(time_ms + max(0.0, dur) * 1000)) - 1
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start}")
        lines.append(f"END={max(start, end)}")
        safe_title = str(title).replace('\n', ' ').strip()
        lines.append(f"title={safe_title}")
        time_ms += max(0.0, dur) * 1000
    return "\n".join(lines) + "\n"


def _generate_slate_video(width: int, height: int, fps: int, duration: float, label: str, background: str = 'black') -> str:
    if duration <= 0:
        raise ValueError("Slate duration must be > 0")
    frames = max(1, int(round(duration * max(1, fps))))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = os.path.join(tempfile.gettempdir(), f"slate_{abs(hash((label, frames, width, height, fps)))}.mp4")
    out = cv2.VideoWriter(path, fourcc, max(1, fps), (width, height))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter for slate")
    for i in range(frames):
        frame = _get_bg(width, height, background, i)
        fade_in = clamp(i / 12.0, 0, 1)
        fade_out = clamp((frames - 1 - i) / 12.0, 0, 1)
        alpha = min(fade_in, fade_out)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.0
        thick = 2
        tw, th = cv2.getTextSize(label, font, scale, thick)[0]
        x = (width - tw) // 2
        y = (height + th) // 2
        overlay = frame.copy()
        cv2.putText(overlay, label, (x+2, y+2), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, y), font, (scale), (240,240,240), thick, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        out.write(frame)
    out.release()
    return path


def _render_one_for_batch(ap: str, base_settings: RenderSettings) -> Tuple[str, str, float]:
    s = RenderSettings(
        audio_path=ap,
        out_dir=base_settings.out_dir,
        resolution=base_settings.resolution,
        fps=base_settings.fps,
        style=base_settings.style,
        palette=base_settings.palette,
        background=base_settings.background,
        quality=base_settings.quality,
        glow=base_settings.glow,
        glow_intensity=base_settings.glow_intensity,
        blur=base_settings.blur,
        blur_amount=base_settings.blur_amount,
        particles=base_settings.particles,
        particle_count=base_settings.particle_count,
        mirror=base_settings.mirror,
        beat_react=base_settings.beat_react,
        logo_text=base_settings.logo_text,
        logo_img_path=base_settings.logo_img_path,
        logo_pos=base_settings.logo_pos,
        logo_scale=base_settings.logo_scale,
        logo_opacity=base_settings.logo_opacity,
        logo_start=base_settings.logo_start,
        logo_end=base_settings.logo_end,
        logo_follow_progress=base_settings.logo_follow_progress,
        bitrate=base_settings.bitrate,
        codec=base_settings.codec,
        draft_start=base_settings.draft_start,
        draft_duration=base_settings.draft_duration,
        progress_bar=base_settings.progress_bar,
        progress_pos=base_settings.progress_pos,
        progress_height=base_settings.progress_height,
        progress_countdown=base_settings.progress_countdown,
        clock=base_settings.clock,
        clock_mode=base_settings.clock_mode,
        clock_style=base_settings.clock_style,
        clock_pos=base_settings.clock_pos,
        clock_scale=base_settings.clock_scale,
    )
    out_path = render_video(s)
    title = os.path.splitext(os.path.basename(ap))[0]
    try:
        y, sr = librosa.load(ap, sr=22050, mono=True)
        dur = float(len(y))/sr
    except Exception:
        dur = 0.0
    # try to free RAM used by librosa
    try:
        del y
    except Exception:
        pass
    gc.collect()
    return out_path, title, dur


def render_batch(audio_paths: List[str], base_settings: RenderSettings, *, merge: bool = True, add_chapters: bool = True,
                 gap_duration: float = 0.0, gap_slate: bool = True, gap_label_format: str = "Next: {title}") -> List[str] | str:
    if not audio_paths:
        raise ValueError("No input files provided for batch")

    # Parallel per-file rendering
    workers = max(1, int(base_settings.workers))
    outs: List[str] = []
    titles: List[str] = []
    durations: List[float] = []

    if workers == 1:
        for ap in audio_paths:
            out_path, title, dur = _render_one_for_batch(ap, base_settings)
            outs.append(out_path); titles.append(title); durations.append(dur)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_render_one_for_batch, ap, base_settings): ap for ap in audio_paths}
            for fut in as_completed(futs):
                out_path, title, dur = fut.result()
                outs.append(out_path); titles.append(title); durations.append(dur)

    if not merge:
        return outs

    # Build concat list with optional slates
    list_file = os.path.join(tempfile.gettempdir(), "concat_list.txt")
    width, height = base_settings.resolution
    fps = max(1, int(base_settings.fps))
    slate_paths: List[str] = []

    # Keep original order of inputs for concat
    with open(list_file, 'w', encoding='utf-8') as f:
        for i, ap in enumerate(audio_paths):
            # find the corresponding out file by title (unique by hash in name), fallback to first match
            title_i = os.path.splitext(os.path.basename(ap))[0]
            # pick the out that contains matching hash of ap to be safest
            hash6 = hashlib.sha1(os.path.abspath(ap).encode('utf-8')).hexdigest()[:6]
            match_path = None
            for p in outs:
                if hash6 in os.path.basename(p):
                    match_path = p; break
            match_path = match_path or outs[i]
            f.write(f"file '{_escape_ffconcat_path(match_path)}'\n")
            if gap_duration > 0 and i < len(audio_paths) - 1:
                if gap_slate:
                    label = (gap_label_format or "Next: {title}").format(index=i+2, title=os.path.splitext(os.path.basename(audio_paths[i+1]))[0])
                    slate = _generate_slate_video(width, height, fps, gap_duration, label, background=base_settings.background)
                    slate_paths.append(slate)
                    f.write(f"file '{_escape_ffconcat_path(slate)}'\n")
                else:
                    slate = _generate_slate_video(width, height, fps, gap_duration, "", background=base_settings.background)
                    slate_paths.append(slate)
                    f.write(f"file '{_escape_ffconcat_path(slate)}'\n")

    merged = os.path.join(base_settings.out_dir or os.path.dirname(outs[0]), "merged_with_chapters_temp.mp4")
    cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', merged]
    subprocess.run(cmd_concat, check=True)

    # Chapters are based only on track durations (not counting slates)
    if add_chapters:
        ffmeta_text = _ffmetadata_for_chapters(titles, durations)
        ffmeta_path = os.path.join(tempfile.gettempdir(), 'chapters.ffmeta')
        with open(ffmeta_path, 'w', encoding='utf-8') as f:
            f.write(ffmeta_text)
        final_path = os.path.join(base_settings.out_dir or os.path.dirname(outs[0]), 'merged_with_chapters.mp4')
        cmd_meta = ['ffmpeg', '-y', '-i', merged, '-i', ffmeta_path, '-map_metadata', '1', '-c', 'copy', final_path]
        subprocess.run(cmd_meta, check=True)
        try:
            os.remove(merged)
        except Exception:
            pass
        if base_settings.cleanup_intermediates:
            for p in outs + slate_paths:
                try: os.remove(p)
                except Exception: pass
        print(f"Saved (merged with chapters): {final_path}")
        return final_path

    if base_settings.cleanup_intermediates:
        for p in outs + slate_paths:
            try: os.remove(p)
            except Exception: pass
    print(f"Saved (merged): {merged}")
    return merged

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
            self.root.geometry("880x1060")
            self.files: List[str] = []
            self.stop_event = Event()
            self.scrollable = ScrollableFrame(root)
            self.scrollable.pack(fill="both", expand=True, padx=6, pady=6)
            container = self.scrollable.scrollable_frame

            title_label = tk.Label(container, text="🎵 Аудио Визуализатор Pro 2.1", font=("Arial", 19, "bold"))
            title_label.pack(pady=12)

            file_row = tk.Frame(container)
            file_row.pack(fill="x", padx=12)
            tk.Button(file_row, text="📁 Открыть файл", command=self.load_one, bg="#4CAF50", fg="white").pack(side="left")
            tk.Button(file_row, text="📚 Открыть несколько", command=self.load_many).pack(side="left", padx=6)
            self.status = tk.Label(file_row, text="Файлы не выбраны", fg="gray")
            self.status.pack(side="left", padx=8)

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
            self.progress_bar_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Полоса прогресса", variable=self.progress_bar_var).grid(row=0, column=0, sticky="w")
            self.progress_pos_var = combo(overlays, "Позиция бара:", ["bottom","top"], "bottom", 0)
            tk.Label(overlays, text="Высота бара:").grid(row=1, column=0, sticky="w")
            self.progress_height_var = tk.IntVar(value=10)
            tk.Spinbox(overlays, from_=2, to=40, textvariable=self.progress_height_var, width=8).grid(row=1, column=1, sticky="w")
            self.progress_countdown_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Обратный отсчёт у бара", variable=self.progress_countdown_var).grid(row=1, column=2, padx=8, sticky="w")

            self.clock_enable_var = tk.BooleanVar(value=True)
            tk.Checkbutton(overlays, text="Часы", variable=self.clock_enable_var).grid(row=2, column=0, sticky="w")
            self.clock_type_var = combo(overlays, "Тип часов:", ["digital","analog"], "digital", 2)
            self.clock_mode_var = combo(overlays, "Режим:", ["elapsed","system"], "elapsed", 3)
            self.clock_style_var = combo(overlays, "Стиль:", ["dark","light","retro","neon","outline"], "dark", 4)
            self.clock_pos_var = combo(overlays, "Позиция:", ["tl","tr","bl","br"], "tr", 5)
            tk.Label(overlays, text="Масштаб:").grid(row=6, column=0, sticky="w")
            self.clock_scale_var = tk.DoubleVar(value=1.0)
            tk.Spinbox(overlays, from_=0.5, to=2.0, increment=0.1, textvariable=self.clock_scale_var, width=8).grid(row=6, column=1, sticky="w")

            logo = tk.LabelFrame(container, text="🖼️ Логотип", padx=16, pady=12)
            logo.pack(pady=10, padx=12, fill="x")
            self.logo_img_path = tk.StringVar(value="")
            tk.Label(logo, text="Файл:").grid(row=0, column=0, sticky="w")
            tk.Entry(logo, textvariable=self.logo_img_path, width=32).grid(row=0, column=1, sticky="w")
            tk.Button(logo, text="Выбрать…", command=lambda: self._choose_logo()).grid(row=0, column=2, padx=6)
            self.logo_pos_var = combo(logo, "Позиция:", ["tl","tr","bl","br"], "tr", 1)
            tk.Label(logo, text="Масштаб:").grid(row=2, column=0, sticky="w")
            self.logo_scale_var = tk.DoubleVar(value=1.0)
            tk.Spinbox(logo, from_=0.25, to=2.5, increment=0.05, textvariable=self.logo_scale_var, width=8).grid(row=2, column=1, sticky="w")
            tk.Label(logo, text="Непрозрачность:").grid(row=2, column=2, sticky="w")
            self.logo_opacity_var = tk.DoubleVar(value=0.8)
            tk.Spinbox(logo, from_=0.0, to=1.0, increment=0.05, textvariable=self.logo_opacity_var, width=8).grid(row=2, column=3, sticky="w")
            tk.Label(logo, text="Показ с (сек):").grid(row=3, column=0, sticky="w")
            self.logo_start_var = tk.DoubleVar(value=0.0)
            tk.Spinbox(logo, from_=0.0, to=9999.0, increment=0.5, textvariable=self.logo_start_var, width=8).grid(row=3, column=1, sticky="w")
            tk.Label(logo, text="по (сек):").grid(row=3, column=2, sticky="w")
            self.logo_end_var = tk.DoubleVar(value=0.0)
            tk.Spinbox(logo, from_=0.0, to=9999.0, increment=0.5, textvariable=self.logo_end_var, width=8).grid(row=3, column=3, sticky="w")
            self.logo_follow_var = tk.BooleanVar(value=False)
            tk.Checkbutton(logo, text="Двигать вдоль прогресса", variable=self.logo_follow_var).grid(row=4, column=0, sticky="w")

            batch = tk.LabelFrame(container, text="📚 Пакетная обработка", padx=16, pady=12)
            batch.pack(pady=10, padx=12, fill="x")
            self.merge_var = tk.BooleanVar(value=True)
            tk.Checkbutton(batch, text="Объединить в один видеофайл", variable=self.merge_var).grid(row=0, column=0, sticky="w")
            self.chapters_var = tk.BooleanVar(value=True)
            tk.Checkbutton(batch, text="Добавить закладки (имена файлов)", variable=self.chapters_var).grid(row=0, column=1, sticky="w", padx=10)
            tk.Label(batch, text="Потоки (файлы):").grid(row=1, column=0, sticky="w")
            self.workers_var = tk.IntVar(value=min(4, max(1, cpu_count()-1)))
            tk.Spinbox(batch, from_=1, to=max(1, cpu_count()), textvariable=self.workers_var, width=8).grid(row=1, column=1, sticky="w")
            tk.Label(batch, text="Пауза между треками (сек):").grid(row=2, column=0, sticky="w")
            self.gap_duration_var = tk.DoubleVar(value=0.0)
            tk.Spinbox(batch, from_=0.0, to=5.0, increment=0.1, textvariable=self.gap_duration_var, width=8).grid(row=2, column=1, sticky="w")
            self.gap_slate_var = tk.BooleanVar(value=True)
            tk.Checkbutton(batch, text="Заставка с названием следующего трека", variable=self.gap_slate_var).grid(row=2, column=2, sticky="w", padx=8)
            tk.Label(batch, text="Формат подписи заставки:").grid(row=3, column=0, sticky="w")
            self.gap_label_entry = tk.Entry(batch, width=28)
            self.gap_label_entry.grid(row=3, column=1, sticky="w")
            self.gap_label_entry.insert(0, "Next: {title}")
            self.cleanup_var = tk.BooleanVar(value=False)
            tk.Checkbutton(batch, text="Удалить промежуточные файлы после мерджа", variable=self.cleanup_var).grid(row=4, column=0, sticky="w")

            extra = tk.LabelFrame(container, text="🎬 Дополнительно", padx=16, pady=12)
            extra.pack(pady=10, padx=12, fill="x")
            self.background_var = combo(extra, "Фон:", ["black","white","gradient","blur"], "black", 0)
            self.quality_var = combo(extra, "Качество:", ["low","medium","high","ultra"], "high", 1)
            self.logo_text_enable = tk.BooleanVar(value=False)
            tk.Checkbutton(extra, text="Текст/логотип (надпись)", variable=self.logo_text_enable).grid(row=2, column=0, sticky="w")
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
            self.button_render = tk.Button(btns, text="🎬 Старт", command=self.start_render, state=tk.DISABLED, bg="#2196F3", fg="white")
            self.button_render.pack(side="left", padx=6)

        def _choose_logo(self):
            path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
            if path:
                self.logo_img_path.set(path)

        def choose_out_dir(self):
            d = filedialog.askdirectory()
            if d:
                self.out_dir.set(d)

        def load_one(self):
            path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac")])
            if not path:
                return
            self.files = [path]
            self.status.config(text=f"✓ 1 файл: {os.path.basename(path)}", fg="green")
            self.button_render.config(state=tk.NORMAL)

        def load_many(self):
            paths = filedialog.askopenfilenames(filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac")])
            if not paths:
                return
            self.files = list(paths)
            self.status.config(text=f"✓ Выбрано файлов: {len(self.files)}", fg="green")
            self.button_render.config(state=tk.NORMAL)

        def start_render(self):
            if not self.files:
                messagebox.showerror("Ошибка", "Сначала выберите аудиофайлы.")
                return
            w, h = map(int, self.resolution_var.get().split('x'))
            base = RenderSettings(
                audio_path=self.files[0],
                out_dir=self.out_dir.get().strip() or os.path.dirname(self.files[0]),
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
                logo_text=(self.logo_text.get().strip() if self.logo_text_enable.get() else None),
                logo_img_path=(self.logo_img_path.get().strip() or None),
                logo_pos=self.logo_pos_var.get(),
                logo_scale=float(self.logo_scale_var.get()),
                logo_opacity=float(self.logo_opacity_var.get()),
                logo_start=float(self.logo_start_var.get()),
                logo_end=(float(self.logo_end_var.get()) if self.logo_end_var.get() > 0 else None),
                logo_follow_progress=bool(self.logo_follow_var.get()),
                bitrate=self.bitrate_var.get(),
                codec=self.codec_var.get(),
                progress_bar=self.progress_bar_var.get(),
                progress_pos=self.progress_pos_var.get(),
                progress_height=int(self.progress_height_var.get()),
                progress_countdown=self.progress_countdown_var.get(),
                clock=self.clock_type_var.get() if self.clock_enable_var.get() else None,
                clock_mode=self.clock_mode_var.get(),
                clock_style=self.clock_style_var.get(),
                clock_pos=self.clock_pos_var.get(),
                clock_scale=float(self.clock_scale_var.get()),
                workers=int(self.workers_var.get()),
                cleanup_intermediates=bool(self.cleanup_var.get()),
            )
            merge = bool(self.merge_var.get()) if len(self.files) > 1 else False
            chapters = bool(self.chapters_var.get()) if merge else False
            gap_dur = float(self.gap_duration_var.get()) if merge else 0.0
            gap_slate = bool(self.gap_slate_var.get()) if merge else False
            gap_label_fmt = self.gap_label_entry.get().strip() or "Next: {title}"

            self.progress_label.config(text="Рендер…")
            self.button_render.config(state=tk.DISABLED)

            def _bg():
                try:
                    if merge:
                        out_path = render_batch(self.files, base, merge=True, add_chapters=chapters,
                                                gap_duration=gap_dur, gap_slate=gap_slate, gap_label_format=gap_label_fmt)
                        self.root.after(0, lambda: messagebox.showinfo("Готово", f"Сохранено:\n{out_path}"))
                    else:
                        outs = render_batch(self.files, base, merge=False)
                        self.root.after(0, lambda: messagebox.showinfo("Готово", "\n".join(outs)))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                finally:
                    self.root.after(0, lambda: (self.progress_label.config(text=""), self.button_render.config(state=tk.NORMAL)))

            Thread(target=_bg, daemon=True).start()

# =============================
# CLI
# =============================

def _tk_runtime_available() -> bool:
    if not TK_AVAILABLE:
        return False
    try:
        _r = tk.Tk()
        _r.withdraw()
        _r.update_idletasks()
        _r.destroy()
        return True
    except Exception:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audio Visualizer Pro — CLI/GUI hybrid")
    p.add_argument('--audio', nargs='+', help='Path(s) to input audio file(s). Globs allowed (quoted).')
    p.add_argument('--out-dir', default='', help='Output directory (default: alongside first audio)')
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
    # Logo image options
    p.add_argument('--logo-img', dest='logo_img_path', default=None)
    p.add_argument('--logo-pos', default='tr', choices=['tl','tr','bl','br'])
    p.add_argument('--logo-scale', type=float, default=1.0)
    p.add_argument('--logo-opacity', type=float, default=0.8)
    p.add_argument('--logo-start', type=float, default=0.0)
    p.add_argument('--logo-end', type=float, default=None)
    p.add_argument('--logo-follow-progress', action='store_true')

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

    # batch
    p.add_argument('--merge', dest='merge', action='store_true', help='Merge multiple inputs into one output with same settings')
    p.add_argument('--no-merge', dest='merge', action='store_false', help='Render multiple inputs separately')
    p.set_defaults(merge=True)
    p.add_argument('--chapters', dest='chapters', action='store_true', help='Add chapters from file names on merged output')
    p.add_argument('--no-chapters', dest='chapters', action='store_false')
    p.set_defaults(chapters=True)
    p.add_argument('--gap-duration', type=float, default=0.0, help='Gap/slate duration between tracks (seconds). 0 disables')
    p.add_argument('--gap-slate', dest='gap_slate', action='store_true', help='Show a slate with next track title during the gap')
    p.add_argument('--no-gap-slate', dest='gap_slate', action='store_false')
    p.set_defaults(gap_slate=True)
    p.add_argument('--gap-label-format', default='Next: {title}', help='Label template for slate. Vars: {index} (1-based next track), {title}')
    p.add_argument('--workers', type=int, default=1, help='Parallel workers for per-file rendering')
    p.add_argument('--cleanup-intermediates', action='store_true', help='Delete per-track videos after merging')

    p.add_argument('--run-tests', action='store_true', help='Run unit tests and exit')
    return p.parse_args(argv)


def _expand_inputs(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for item in inputs:
        if any(ch in item for ch in ['*', '?', '[']):
            out.extend(sorted(glob.glob(item)))
        else:
            out.append(item)
    out = [p for p in out if os.path.isfile(p)]
    if not out:
        raise FileNotFoundError("No matching input files found")
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.run_tests:
        import unittest
        unittest.main(module=__name__, argv=[sys.argv[0]], exit=False)
        return

    # If no audio provided and Tk usable — launch GUI, else headless
    if (not args.audio) and _tk_runtime_available():
        root = tk.Tk()
        app = AudioVisualizerApp(root)
        root.mainloop()
        return

    if not args.audio:
        print("Error: --audio is required in headless mode (Tk not available or GUI cannot start)")
        print("Tip: run with --run-tests to verify environment, or install a desktop Tk (python3-tk) to use the GUI.")
        sys.exit(2)

    try:
        w, h = map(int, args.resolution.lower().split('x'))
    except Exception:
        print("Invalid --resolution. Use WIDTHxHEIGHT, e.g. 1280x720")
        sys.exit(2)

    clock_val = None if (args.clock == 'none') else args.clock
    inputs = _expand_inputs(args.audio)

    base_settings = RenderSettings(
        audio_path=inputs[0],
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
        logo_img_path=(args.logo_img_path if args.logo_img_path else None),
        logo_pos=args.logo_pos,
        logo_scale=float(args.logo_scale),
        logo_opacity=float(args.logo_opacity),
        logo_start=float(args.logo_start),
        logo_end=(float(args.logo_end) if args.logo_end is not None else None),
        logo_follow_progress=bool(args.logo_follow_progress),
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
        workers=int(args.workers),
        cleanup_intermediates=bool(args.cleanup_intermediates),
    )

    try:
        if len(inputs) == 1:
            render_video(base_settings)
        else:
            if not args.merge:
                outs = render_batch(inputs, base_settings, merge=False)
                for p in outs:
                    print(p)
            else:
                final = render_batch(inputs, base_settings, merge=True, add_chapters=bool(args.chapters),
                                     gap_duration=float(args.gap_duration), gap_slate=bool(args.gap_slate),
                                     gap_label_format=str(args.gap_label_format))
                print(final)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e)
        sys.exit(1)
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

    def test_ffconcat_escape(self):
        p = "/tmp/ab'cd x.mp4"
        esc = _escape_ffconcat_path(p)
        self.assertNotEqual(p, esc)
        self.assertIn("\\''", esc)

class TestFFMetadata(unittest.TestCase):
    def test_ffmetadata_chapters(self):
        titles = ['a', 'b']
        durs = [1.0, 2.5]
        meta = _ffmetadata_for_chapters(titles, durs)
        self.assertIn(';FFMETADATA1', meta)
        self.assertIn('title=a', meta)
        self.assertIn('title=b', meta)
        self.assertIn('START=0', meta)
        self.assertIn('END=999', meta)  # ~1s

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
        self.assertTrue(out[-10:].sum() > 0)
    def test_progress_countdown_label(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out = _overlay_progress_bar(img.copy(), 0.5, 'bottom', 8, (255,255,255), label='-00:10:00')
        self.assertEqual(out.shape, (self.h, self.w, 3))
    def test_digital_clock_overlay(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        out = _overlay_digital_clock(img.copy(), '12:34:56', 'tr', 'dark', 1.0)
        self.assertEqual(out.shape, (self.h, self.w, 3))
    def test_logo_overlay(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Synthetic small logo (RGBA)
        logo = np.zeros((20, 40, 4), dtype=np.uint8)
        logo[..., :3] = 255
        logo[..., 3] = 128
        out = _overlay_logo(img.copy(), logo, 'tr', 1.0, 0.5, 0.3, False)
        self.assertEqual(out.shape, (self.h, self.w, 3))

@unittest.skipUnless(CV2_AVAILABLE, "cv2 not installed")
class TestSlate(unittest.TestCase):
    def test_generate_slate_video(self):
        path = _generate_slate_video(320, 180, 24, 0.5, "Next: Track")
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 1000)
        try:
            os.remove(path)
        except Exception:
            pass

@unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
class TestAudioAnalysis(unittest.TestCase):
    def test_analysis_hop_and_frames(self):
        sr = 22050
        t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
        y = 0.2*np.sin(2*math.pi*440*t).astype(np.float32)
        aa = AudioAnalysis(path=None, sr=sr, fps=25, y=y)
        self.assertGreater(aa.n_frames, 0)
        self.assertEqual(aa.hop_length, max(1, int(sr/25)))

if __name__ == '__main__':
    main()
