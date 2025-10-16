import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import librosa
import os
import sys
import math
import tempfile
import subprocess
from threading import Thread, Event

# =============================
# Utility helpers
# =============================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Cross‑platform mousewheel constants
_PLATFORM = sys.platform

# =============================
# Scrollable container
# =============================
class ScrollableFrame(tk.Frame):
    """Прокручиваемый фрейм с кроссплатформенной поддержкой колесика"""
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

# =============================
# Color utilities
# =============================
class Palette:
    @staticmethod
    def scheme(name):
        name = (name or "rainbow").lower()
        return getattr(Palette, name, Palette.rainbow)

    @staticmethod
    def _to_bgr(r, g, b):
        return (int(b), int(g), int(r))

    @staticmethod
    def rainbow(f, i, total):
        hue = (i / max(total, 1)) * 2 * math.pi
        r = 127 + 127 * math.sin(hue)
        g = 127 + 127 * math.sin(hue + 2 * math.pi / 3)
        b = 127 + 127 * math.sin(hue + 4 * math.pi / 3)
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def fire(f, i, total):
        r = 255 * f
        g = 180 * f
        b = 60 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def ocean(f, i, total):
        r = 60 * f
        g = 180 * f
        b = 255 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def neon(f, i, total):
        ch = i % 3
        r = 255 * f if ch == 0 else 50
        g = 255 * f if ch == 1 else 50
        b = 255 * f if ch == 2 else 50
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def purple(f, i, total):
        r = 200 * f
        g = 50 * f
        b = 255 * f
        return Palette._to_bgr(r, g, b)

    @staticmethod
    def gradient(f, i, total):
        ratio = i / max(total, 1)
        r = 255 * ratio * f
        g = 128 * (1 - ratio) * f
        b = 200 * f
        return Palette._to_bgr(r, g, b)

# =============================
# Audio analysis helpers
# =============================
class AudioAnalysis:
    """Предварительный расчёт признаков для кадров.
       Маппинг fps -> hop_length, мел‑спектр и детекция ударов."""
    def __init__(self, path, sr=22050, fps=30, n_fft=2048, n_mels=128):
        self.path = path
        self.sr = sr
        self.fps = fps
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.y, self.sr = librosa.load(self.path, sr=self.sr, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

        # Hop aligned to fps (ensures consistent energy per frame)
        self.hop_length = max(1, int(self.sr / self.fps))

        # Mel spectrogram (energy compressed), then normalize per‑frame
        S = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / max(1e-9, (S_db.max() - S_db.min()))
        self.mels = S_db  # shape: (n_mels, n_frames)
        self.n_frames = self.mels.shape[1]

        # Beat tracking on onset envelope improves stability vs FFT thresholding
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop_length)
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length)
        self.beat_frames = set(int(b) for b in beat_frames)

    def frame_features(self, frame_idx):
        idx = clamp(frame_idx, 0, self.n_frames - 1)
        mel_col = self.mels[:, idx]
        # Low‑to‑high bands energy summaries
        low = mel_col[: self.n_mels // 4].mean()
        mid = mel_col[self.n_mels // 4: self.n_mels // 2].mean()
        high = mel_col[self.n_mels // 2 :].mean()
        beat = idx in self.beat_frames
        return mel_col, (low, mid, high), beat

# =============================
# Main app
# =============================
class AudioVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Audio Visualizer Pro v2")
        # HiDPI scaling – improves crispness on 125–200% scale
        try:
            self.root.tk.call('tk', 'scaling', self.root.winfo_fpixels('1i') / 72.0)
        except Exception:
            pass

        self.root.geometry("720x720")
        self.file_path = None
        self.is_rendering = False
        self.stop_event = Event()
        self.analysis = None

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

        # Main settings
        main_frame = tk.LabelFrame(container, text="⚙️ Основные настройки", padx=16, pady=12)
        main_frame.pack(pady=10, padx=12, fill="x")

        tk.Label(main_frame, text="Разрешение:").grid(row=0, column=0, sticky="w")
        self.resolution_var = tk.StringVar(value="1280x720")
        ttk.Combobox(main_frame, textvariable=self.resolution_var,
                     values=["640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"], state="readonly", width=14).grid(row=0, column=1, padx=6)

        tk.Label(main_frame, text="FPS:").grid(row=1, column=0, sticky="w")
        self.fps_var = tk.IntVar(value=30)
        tk.Spinbox(main_frame, from_=15, to=60, textvariable=self.fps_var, width=16).grid(row=1, column=1, padx=6)

        tk.Label(main_frame, text="Стиль:").grid(row=2, column=0, sticky="w")
        self.style_var = tk.StringVar(value="bars")
        ttk.Combobox(main_frame, textvariable=self.style_var,
                     values=["bars", "circle", "wave", "spectrum", "dual", "spiral"], state="readonly", width=14).grid(row=2, column=1, padx=6)

        tk.Label(main_frame, text="Цвета:").grid(row=3, column=0, sticky="w")
        self.color_var = tk.StringVar(value="rainbow")
        ttk.Combobox(main_frame, textvariable=self.color_var,
                     values=["rainbow", "fire", "ocean", "neon", "purple", "gradient"], state="readonly", width=14).grid(row=3, column=1, padx=6)

        # Effects
        effects_frame = tk.LabelFrame(container, text="✨ Визуальные эффекты", padx=16, pady=12)
        effects_frame.pack(pady=10, padx=12, fill="x")

        self.glow_var = tk.BooleanVar(value=True)
        tk.Checkbutton(effects_frame, text="Свечение (Glow)", variable=self.glow_var).grid(row=0, column=0, sticky="w")
        tk.Label(effects_frame, text="Интенсивность:").grid(row=0, column=1, padx=(10, 4))
        self.glow_intensity = tk.IntVar(value=3)
        tk.Scale(effects_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.glow_intensity, length=110).grid(row=0, column=2)

        self.blur_var = tk.BooleanVar(value=False)
        tk.Checkbutton(effects_frame, text="Размытие (Blur)", variable=self.blur_var).grid(row=1, column=0, sticky="w")
        tk.Label(effects_frame, text="Сила:").grid(row=1, column=1, padx=(10, 4))
        self.blur_amount = tk.IntVar(value=5)
        tk.Scale(effects_frame, from_=1, to=15, orient=tk.HORIZONTAL, variable=self.blur_amount, length=110).grid(row=1, column=2)

        self.particles_var = tk.BooleanVar(value=False)
        tk.Checkbutton(effects_frame, text="Частицы", variable=self.particles_var).grid(row=2, column=0, sticky="w")
        tk.Label(effects_frame, text="Количество:").grid(row=2, column=1, padx=(10, 4))
        self.particle_count = tk.IntVar(value=60)
        tk.Scale(effects_frame, from_=10, to=250, orient=tk.HORIZONTAL, variable=self.particle_count, length=110).grid(row=2, column=2)

        self.mirror_var = tk.BooleanVar(value=False)
        tk.Checkbutton(effects_frame, text="Зеркальное отражение", variable=self.mirror_var).grid(row=3, column=0, sticky="w", columnspan=3)

        self.beat_react_var = tk.BooleanVar(value=True)
        tk.Checkbutton(effects_frame, text="Реакция на биты", variable=self.beat_react_var).grid(row=4, column=0, sticky="w", columnspan=3)

        # Extras
        extra_frame = tk.LabelFrame(container, text="🎬 Дополнительно", padx=16, pady=12)
        extra_frame.pack(pady=10, padx=12, fill="x")

        tk.Label(extra_frame, text="Фон:").grid(row=0, column=0, sticky="w")
        self.background_var = tk.StringVar(value="black")
        ttk.Combobox(extra_frame, textvariable=self.background_var,
                     values=["black", "white", "gradient", "blur"], state="readonly", width=14).grid(row=0, column=1)

        tk.Label(extra_frame, text="Качество:").grid(row=1, column=0, sticky="w")
        self.quality_var = tk.StringVar(value="high")
        ttk.Combobox(extra_frame, textvariable=self.quality_var,
                     values=["low", "medium", "high", "ultra"], state="readonly", width=14).grid(row=1, column=1)

        self.logo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(extra_frame, text="Добавить текст/логотип", variable=self.logo_var).grid(row=2, column=0, sticky="w")
        self.logo_text = tk.Entry(extra_frame, width=22)
        self.logo_text.grid(row=2, column=1, pady=4)
        self.logo_text.insert(0, "My Track")

        tk.Label(extra_frame, text="Битрейт аудио:").grid(row=3, column=0, sticky="w")
        self.bitrate_var = tk.StringVar(value="192k")
        ttk.Combobox(extra_frame, textvariable=self.bitrate_var, values=["128k", "192k", "256k", "320k"], state="readonly", width=14).grid(row=3, column=1)

        tk.Label(extra_frame, text="Видео кодек:").grid(row=4, column=0, sticky="w")
        self.codec_var = tk.StringVar(value="h264")
        ttk.Combobox(extra_frame, textvariable=self.codec_var, values=["h264", "h265", "vp9"], state="readonly", width=14).grid(row=4, column=1)

        # Output path
        out_row = tk.Frame(container)
        out_row.pack(fill="x", padx=12, pady=6)
        tk.Label(out_row, text="Папка вывода:").pack(side="left")
        self.out_dir = tk.StringVar(value="")
        tk.Entry(out_row, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(out_row, text="Выбрать…", command=self.choose_out_dir).pack(side="left")

        # Bottom controls
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
        self.button_stop = tk.Button(btns, text="⏹ Остановить", command=self.request_stop, state=tk.DISABLED)
        self.button_stop.pack(side="left", padx=6)

        # Preview canvas (optional minimal live preview while рендер)
        self.preview_var = tk.BooleanVar(value=True)
        tk.Checkbutton(container, text="Показывать превью при рендере (нагрузка ↑)", variable=self.preview_var).pack(anchor="w", padx=14)
        self.preview = tk.Label(container)
        self.preview.pack(padx=14, pady=6)

        self.particles = []

    # ------------------------- UI actions
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

    # ------------------------- visual helpers
    def _get_bg(self, w, h, frame_num=0):
        bg_type = self.background_var.get()
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

    def _apply_glow(self, img):
        if not self.glow_var.get():
            return img
        intensity = self.glow_intensity.get()
        blur = cv2.GaussianBlur(img, (0, 0), intensity)
        return cv2.addWeighted(img, 0.6, blur, 0.4, 0)

    def _apply_blur(self, img):
        if not self.blur_var.get():
            return img
        k = self.blur_amount.get() * 2 + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def _apply_mirror(self, img):
        if not self.mirror_var.get():
            return img
        h = img.shape[0]
        top = img[: h // 2].copy()
        img[h // 2 : h // 2 + top.shape[0]] = cv2.flip(top, 0)
        return img

    def _add_text(self, img, beat=False):
        if not self.logo_var.get():
            return img
        text = self.logo_text.get().strip()
        if not text:
            return img
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0 if beat else 1.5
        thick = 3
        size = cv2.getTextSize(text, font, scale, thick)[0]
        x = (w - size[0]) // 2
        y = h - 50
        cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)
        return img

    def _add_particles(self, img, energy, beat=False):
        if not self.particles_var.get():
            return img
        h, w = img.shape[:2]
        target = self.particle_count.get()
        while len(self.particles) < target:
            self.particles.append({
                'x': np.random.uniform(0, w),
                'y': np.random.uniform(0, h),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'size': np.random.randint(2, 6),
                'life': np.random.randint(40, 140)
            })
        avg = float(np.mean(energy)) if len(energy) else 0.0
        color_fn = Palette.scheme(self.color_var.get())
        for p in list(self.particles):
            if beat:
                p['vx'] *= 1.35
                p['vy'] *= 1.35
            p['x'] += p['vx']
            p['y'] += p['vy']
            if p['x'] < 0 or p['x'] >= w: p['vx'] *= -1; p['x'] = clamp(p['x'], 0, w-1)
            if p['y'] < 0 or p['y'] >= h: p['vy'] *= -1; p['y'] = clamp(p['y'], 0, h-1)
            p['vx'] *= 0.985; p['vy'] *= 0.985
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            color = color_fn(avg, len(self.particles), max(1, target))
            cv2.circle(img, (int(p['x']), int(p['y'])), p['size'], color, -1)
        return img

    # ------------------------- styles
    def frame_bars(self, mel_col, w, h, beat=False):
        img = self._get_bg(w, h)
        bar_count = min(100, len(mel_col))
        bar_w = max(1, w // bar_count)
        color_fn = Palette.scheme(self.color_var.get())
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

    def frame_circle(self, mel_col, w, h, beat=False):
        img = self._get_bg(w, h)
        cx, cy = w // 2, h // 2
        base = min(w, h) // 4
        color_fn = Palette.scheme(self.color_var.get())
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

    def frame_wave(self, mel_col, w, h, beat=False):
        img = self._get_bg(w, h)
        color_fn = Palette.scheme(self.color_var.get())
        n = len(mel_col)
        xs = np.linspace(0, w-1, n).astype(int)
        ys = (h/2 + (mel_col - 0.5) * h * (0.6 if not beat else 0.8)).astype(int)
        for i in range(1, n):
            f = float(mel_col[i])
            color = color_fn(f, i, n)
            cv2.line(img, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 2)
        return img

    def frame_spectrum(self, mel_col, w, h, beat=False):
        img = self._get_bg(w, h)
        color_fn = Palette.scheme(self.color_var.get())
        n = len(mel_col)
        for i in range(n):
            f = float(mel_col[i])
            y = int((1.0 - f) * (h-1))
            color = color_fn(f, i, n)
            cv2.line(img, (i * w // n, h-1), (i * w // n, y), color, 1)
        return img

    def frame_dual(self, mel_col, w, h, beat=False):
        # Bars + circle hybrid
        img = self.frame_bars(mel_col, w, h, beat)
        overlay = self.frame_circle(mel_col, w, h, beat)
        return cv2.addWeighted(img, 0.6, overlay, 0.6, 0)

    def frame_spiral(self, mel_col, w, h, beat=False):
        img = self._get_bg(w, h)
        color_fn = Palette.scheme(self.color_var.get())
        cx, cy = w // 2, h // 2
        n = len(mel_col)
        twist = 5.0
        scale = 1.4 if beat else 1.0
        for i, f in enumerate(mel_col):
            a = i / n * 2 * math.pi * twist
            r = (i / n) * (min(w, h) * 0.45) + f * 60 * scale
            x = int(cx + math.cos(a) * r)
            y = int(cy + math.sin(a) * r)
            color = color_fn(float(f), i, n)
            cv2.circle(img, (x, y), 2, color, -1)
        return img

    # ------------------------- render orchestration
    def start_render(self):
        if self.is_rendering:
            return
        if not self.file_path:
            messagebox.showerror("Ошибка", "Сначала выберите аудиофайл.")
            return
        self.is_rendering = True
        self.stop_event.clear()
        self.button_render.config(state=tk.DISABLED)
        self.button_open.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self.particles = []

        # Prepare analysis upfront aligned to current FPS
        try:
            fps = int(self.fps_var.get())
            self.analysis = AudioAnalysis(self.file_path, sr=22050, fps=fps, n_fft=2048, n_mels=128)
        except Exception as e:
            self.is_rendering = False
            self.button_render.config(state=tk.NORMAL)
            self.button_open.config(state=tk.NORMAL)
            self.button_stop.config(state=tk.DISABLED)
            messagebox.showerror("Ошибка анализа", str(e))
            return

        thread = Thread(target=self._render_thread, daemon=True)
        thread.start()

    def request_stop(self):
        self.stop_event.set()

    def _render_thread(self):
        try:
            width, height = map(int, self.resolution_var.get().split('x'))
            fps = int(self.fps_var.get())
            style = self.style_var.get()
            total_frames = self.analysis.n_frames

            # Output path
            base = os.path.splitext(os.path.basename(self.file_path))[0]
            out_dir = self.out_dir.get().strip() or os.path.dirname(self.file_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}_visual.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path_tmp = os.path.join(tempfile.gettempdir(), f"{base}_render_tmp.mp4")
            out = cv2.VideoWriter(video_path_tmp, fourcc, fps, (width, height))

            # Supersampling for 'ultra' quality
            quality = self.quality_var.get()
            ss = 1
            if quality == 'ultra':
                ss = 2
            W, H = width * ss, height * ss

            # Select renderer
            renderer = {
                'bars': self.frame_bars,
                'circle': self.frame_circle,
                'wave': self.frame_wave,
                'spectrum': self.frame_spectrum,
                'dual': self.frame_dual,
                'spiral': self.frame_spiral,
            }.get(style, self.frame_bars)

            for idx in range(total_frames):
                if self.stop_event.is_set():
                    break
                mel_col, (low, mid, high), beat = self.analysis.frame_features(idx)
                if not self.beat_react_var.get():
                    beat = False

                frame = renderer(mel_col, W, H, beat)
                frame = self._apply_glow(frame)
                frame = self._apply_blur(frame)
                frame = self._add_particles(frame, mel_col, beat)
                frame = self._apply_mirror(frame)
                frame = self._add_text(frame, beat)

                # Downscale if supersampled
                if ss != 1:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                out.write(frame)

                prog = (idx + 1) / max(1, total_frames) * 100
                self.root.after(0, lambda p=prog: self.progress.config(value=p))
                self.root.after(0, lambda i=idx+1, t=total_frames, p=prog: self.progress_label.config(text=f"Кадр {i}/{t} ({p:.1f}%)"))

                # Lightweight preview
                if self.preview_var.get() and (idx % max(1, int(self.analysis.fps/6)) == 0):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(rgb, (min(480, width), int(min(480, width) * height / width)))
                    imgtk = tk.PhotoImage(master=self.preview, data=cv2.imencode('.png', img)[1].tobytes())
                    # Keep ref to avoid GC
                    self.preview.imgtk = imgtk
                    self.root.after(0, lambda im=imgtk: self.preview.config(image=im))

            out.release()

            if self.stop_event.is_set():
                try:
                    os.remove(video_path_tmp)
                except Exception:
                    pass
                self.root.after(0, lambda: messagebox.showinfo("Остановлено", "Рендер был остановлен пользователем."))
                return

            # Mux audio via ffmpeg (copy video, encode audio)
            bitrate = self.bitrate_var.get()
            codec = self.codec_var.get()
            vcodec = {
                'h264': 'libx264',
                'h265': 'libx265',
                'vp9': 'libvpx-vp9',
            }.get(codec, 'libx264')

            cmd = [
                'ffmpeg', '-y',
                '-i', video_path_tmp,
                '-i', self.file_path,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', vcodec,
                '-crf', '18' if codec in ('h264','h265') else '30',
                '-preset', 'medium',
                '-c:a', 'aac' if codec in ('h264','h265') else 'libopus',
                '-b:a', bitrate,
                '-shortest',
                out_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                # Fallback: try simple copy mux
                fallback = [
                    'ffmpeg', '-y', '-i', video_path_tmp, '-i', self.file_path,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', bitrate, '-shortest', out_path
                ]
                subprocess.run(fallback, check=False)

            try:
                os.remove(video_path_tmp)
            except Exception:
                pass

            self.root.after(0, lambda: messagebox.showinfo("✅ Готово!", f"Видео сохранено:\n{out_path}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось создать видео:\n{e}"))
        finally:
            self.root.after(0, self._finish_render)

    def _finish_render(self):
        self.is_rendering = False
        self.progress.config(value=0)
        self.progress_label.config(text="")
        self.button_render.config(state=tk.NORMAL)
        self.button_open.config(state=tk.NORMAL)
        self.button_stop.config(state=tk.DISABLED)

# ------------------------- main
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioVisualizerApp(root)
    root.mainloop()
