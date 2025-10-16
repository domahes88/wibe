import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import librosa
import os
from threading import Thread

class AudioVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procedural Audio Visualizer")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        self.file_path = None
        self.is_rendering = False
        
        # Заголовок
        title_label = tk.Label(root, text="Аудио Визуализатор", font=("Arial", 16, "bold"))
        title_label.pack(pady=15)
        
        # Кнопка выбора файла
        self.button_open = tk.Button(
            root, 
            text="📁 Открыть аудиофайл", 
            command=self.load_file,
            font=("Arial", 11),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        self.button_open.pack(pady=10)
        
        # Статус файла
        self.status = tk.Label(root, text="Файл не выбран", fg="gray", font=("Arial", 10))
        self.status.pack(pady=5)
        
        # Фрейм настроек
        settings_frame = tk.LabelFrame(root, text="Настройки визуализации", padx=20, pady=15)
        settings_frame.pack(pady=15, padx=20, fill="both")
        
        # Разрешение видео
        tk.Label(settings_frame, text="Разрешение:").grid(row=0, column=0, sticky="w", pady=5)
        self.resolution_var = tk.StringVar(value="1280x720")
        resolution_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.resolution_var, 
            values=["640x480", "1280x720", "1920x1080", "2560x1440"],
            state="readonly",
            width=15
        )
        resolution_combo.grid(row=0, column=1, pady=5)
        
        # FPS
        tk.Label(settings_frame, text="FPS:").grid(row=1, column=0, sticky="w", pady=5)
        self.fps_var = tk.IntVar(value=30)
        fps_spin = tk.Spinbox(settings_frame, from_=15, to=60, textvariable=self.fps_var, width=17)
        fps_spin.grid(row=1, column=1, pady=5)
        
        # Стиль визуализации
        tk.Label(settings_frame, text="Стиль:").grid(row=2, column=0, sticky="w", pady=5)
        self.style_var = tk.StringVar(value="bars")
        style_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.style_var,
            values=["bars", "circle", "wave", "spectrum"],
            state="readonly",
            width=15
        )
        style_combo.grid(row=2, column=1, pady=5)
        
        # Цветовая схема
        tk.Label(settings_frame, text="Цвета:").grid(row=3, column=0, sticky="w", pady=5)
        self.color_var = tk.StringVar(value="rainbow")
        color_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.color_var,
            values=["rainbow", "fire", "ocean", "neon", "purple"],
            state="readonly",
            width=15
        )
        color_combo.grid(row=3, column=1, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(root, mode='determinate', length=400)
        self.progress.pack(pady=10)
        
        self.progress_label = tk.Label(root, text="", fg="blue", font=("Arial", 9))
        self.progress_label.pack()
        
        # Кнопка рендера
        self.button_render = tk.Button(
            root, 
            text="🎬 Создать видео", 
            command=self.start_render,
            state=tk.DISABLED,
            font=("Arial", 11, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10
        )
        self.button_render.pack(pady=10)
        
    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac")]
        )
        if not path:
            return
        self.file_path = path
        filename = os.path.basename(path)
        self.status.config(text=f"✓ Выбран: {filename}", fg="green")
        self.button_render.config(state=tk.NORMAL)
    
    def get_color_scheme(self, f, i, total):
        """Генерация цветов по выбранной схеме"""
        scheme = self.color_var.get()
        
        if scheme == "rainbow":
            hue = i / total
            r = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * hue)))
            g = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * (hue + 0.33))))
            b = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * (hue + 0.66))))
        elif scheme == "fire":
            r = int(f * 255)
            g = int(f * 180)
            b = int(f * 50)
        elif scheme == "ocean":
            r = int(f * 50)
            g = int(f * 180)
            b = int(f * 255)
        elif scheme == "neon":
            r = int(255 * f if i % 3 == 0 else 50)
            g = int(255 * f if i % 3 == 1 else 50)
            b = int(255 * f if i % 3 == 2 else 50)
        elif scheme == "purple":
            r = int(f * 200)
            g = int(f * 50)
            b = int(f * 255)
        
        return (b, g, r)  # OpenCV использует BGR вместо RGB
    
    def make_frame_bars(self, freq, width, height):
        """Стиль: вертикальные столбцы"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        bar_count = 100
        bar_width = width // bar_count
        
        for i in range(bar_count):
            f = freq[i * (len(freq) // bar_count)] if i < len(freq) else 0
            h = int(f * height * 0.8)
            color = self.get_color_scheme(f, i, bar_count)
            x_start = i * bar_width
            x_end = x_start + bar_width - 2
            y_start = height - h
            cv2.rectangle(img, (x_start, y_start), (x_end, height), color, -1)
        
        return img
    
    def make_frame_circle(self, freq, width, height):
        """Стиль: круговая визуализация"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        radius_base = min(width, height) // 4
        
        for i, f in enumerate(freq[:200]):
            angle = (i / 200) * 2 * np.pi
            radius = radius_base + int(f * 200)
            x = int(cx + np.cos(angle) * radius)
            y_pos = int(cy + np.sin(angle) * radius)
            color = self.get_color_scheme(f, i, 200)
            
            cv2.circle(img, (x, y_pos), 4, color, -1)
        
        return img
    
    def make_frame_wave(self, segment, width, height):
        """Стиль: волновая визуализация"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(segment) > 0:
            segment = segment / (np.max(np.abs(segment)) + 1e-10)
            step = len(segment) // width
            
            points = []
            for x in range(width):
                idx = x * step
                if idx < len(segment):
                    y = int(height // 2 + segment[idx] * height * 0.4)
                    points.append([x, y])
                    color = self.get_color_scheme(abs(segment[idx]), x, width)
                    cv2.circle(img, (x, y), 2, color, -1)
            
            if len(points) > 1:
                points_np = np.array(points, dtype=np.int32)
                cv2.polylines(img, [points_np], False, (100, 100, 100), 1)
        
        return img
    
    def make_frame_spectrum(self, freq, width, height):
        """Стиль: спектральная визуализация"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(len(freq)):
            if i >= width:
                break
            f = freq[i]
            h = int(f * height)
            color = self.get_color_scheme(f, i, len(freq))
            
            cv2.line(img, (i, height), (i, height - h), color, 1)
        
        return img
    
    def start_render(self):
        """Запуск рендера в отдельном потоке"""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.button_render.config(state=tk.DISABLED)
        self.button_open.config(state=tk.DISABLED)
        
        thread = Thread(target=self.render_video, daemon=True)
        thread.start()
    
    def render_video(self):
        try:
            if not self.file_path:
                messagebox.showerror("Ошибка", "Сначала выберите аудиофайл.")
                return
            
            # Загрузка аудио
            y, sr = librosa.load(self.file_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Получение параметров
            width, height = map(int, self.resolution_var.get().split('x'))
            fps = self.fps_var.get()
            style = self.style_var.get()
            
            # Создание видео файла
            out_path = os.path.splitext(self.file_path)[0] + "_visual.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            total_frames = int(duration * fps)
            window_size = 2048
            
            # Генерация кадров
            for frame_num in range(total_frames):
                t = frame_num / fps
                idx = int(t * sr)
                
                # Обновление прогресса
                progress = (frame_num / total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress.config(value=p))
                self.root.after(0, lambda f=frame_num, t=total_frames: 
                               self.progress_label.config(text=f"Кадр {f}/{t}"))
                
                # Извлечение данных
                segment = y[max(0, idx-window_size):idx]
                if len(segment) == 0:
                    segment = np.zeros(window_size)
                
                freq = np.abs(np.fft.fft(segment))[:512]
                freq = freq / (np.max(freq) + 1e-10)
                
                # Генерация кадра
                if style == "bars":
                    frame = self.make_frame_bars(freq, width, height)
                elif style == "circle":
                    frame = self.make_frame_circle(freq, width, height)
                elif style == "wave":
                    frame = self.make_frame_wave(segment, width, height)
                else:  # spectrum
                    frame = self.make_frame_spectrum(freq, width, height)
                
                out.write(frame)
            
            out.release()
            
            # Добавление аудио к видео с помощью ffmpeg
            temp_video = out_path.replace(".mp4", "_temp.mp4")
            os.rename(out_path, temp_video)
            
            # Команда ffmpeg для объединения видео и аудио
            cmd = f'ffmpeg -i "{temp_video}" -i "{self.file_path}" -c:v copy -c:a aac -shortest "{out_path}" -y'
            os.system(cmd)
            
            # Удаление временного файла
            if os.path.exists(temp_video):
                os.remove(temp_video)
            
            self.root.after(0, lambda: messagebox.showinfo("Готово", f"Видео сохранено:\n{out_path}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось создать видео:\n{str(e)}"))
        
        finally:
            self.root.after(0, self.finish_render)
    
    def finish_render(self):
        """Завершение рендера"""
        self.is_rendering = False
        self.progress.config(value=0)
        self.progress_label.config(text="")
        self.button_render.config(state=tk.NORMAL)
        self.button_open.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioVisualizerApp(root)
    root.mainloop()
