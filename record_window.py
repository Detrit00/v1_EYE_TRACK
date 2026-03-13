# record_window.py
import tkinter as tk
from tkinter import messagebox
import cv2
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from utils import center_window
from database import EyeDatabase
from eye_tracker import EyeTracker
import os
from datetime import datetime

class RecordWindow:
    def __init__(self, window, main_window):
        self.window = window
        self.main_window = main_window
        self.window.title("Запись видео - Айтрекинг")

        center_window(self.window, 800, 600)

        self.db = EyeDatabase()
        self.tracker = EyeTracker()
        self.is_recording = False
        self.is_video_recording = False
        self.recording_thread = None
        self.video_writer_raw = None
        self.video_filename_raw = None
        self.video_start_time = 0

        self.setup_ui()

        self.download_model()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        control_frame = tk.Frame(self.window, bg="#f0f0f0", height=60)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.record_btn = tk.Button(
            control_frame,
            text="Начать запись данных",
            font=("Arial", 11, "bold"),
            width=20,
            height=1,
            bg="#4CAF50",
            fg="white",
            command=self.toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.video_record_btn = tk.Button(
            control_frame,
            text="Запись видео",
            font=("Arial", 11, "bold"),
            width=15,
            height=1,
            bg="#FF5722",
            fg="white",
            command=self.toggle_video_recording
        )
        self.video_record_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(
            control_frame,
            text="Показать графики",
            font=("Arial", 11),
            width=15,
            height=1,
            bg="#2196F3",
            fg="white",
            command=self.show_graphs,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        back_btn = tk.Button(
            control_frame,
            text="Назад в меню",
            font=("Arial", 11),
            width=15,
            height=1,
            bg="#FF9800",
            fg="white",
            command=self.go_back
        )
        back_btn.pack(side=tk.RIGHT, padx=5)

        self.video_frame = tk.Frame(
            self.window,
            bg="black",
            width=780,
            height=450
        )
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(self.window, bg="#f0f0f0", height=50)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(
            info_frame,
            text="Левый глаз: X: 0, Y: 0 | Диаметр: 0 px | Скорость: 0 px/s\n"
                 "Правый глаз: X: 0, Y: 0 | Диаметр: 0 px | Скорость: 0 px/s",
            bg="#f0f0f0",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        self.info_label.pack()

    def download_model(self):
        def update_status(msg):
            self.window.update()

        if not self.tracker.download_model(update_status):
            messagebox.showerror("Ошибка", "Не удалось скачать модель")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def toggle_video_recording(self):
        if not self.is_video_recording:
            self.start_video_recording()
        else:
            self.stop_video_recording()

    def start_video_recording(self):
        if not self.tracker.cap or not self.tracker.cap.isOpened():
            messagebox.showerror("Ошибка", "Камера не запущена. Сначала начните запись данных.")
            return

        if not os.path.exists("videos"):
            os.makedirs("videos")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename_raw = f"videos/recording_{timestamp}_raw.avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30.0
        frame_size = (780, 450)

        self.video_writer_raw = cv2.VideoWriter(self.video_filename_raw, fourcc, fps, frame_size)

        if not self.video_writer_raw.isOpened():
            messagebox.showerror("Ошибка", "Не удалось создать файл для записи видео")
            return

        self.is_video_recording = True
        self.video_start_time = time.time()

        self.video_record_btn.config(
            text="Остановить видео",
            bg="#f44336"
        )

        messagebox.showinfo("Информация", f"Запись чистого видео начата\nФайл: {self.video_filename_raw}")

    def stop_video_recording(self):
        self.is_video_recording = False

        if self.video_writer_raw:
            self.video_writer_raw.release()
            self.video_writer_raw = None

            if os.path.exists(self.video_filename_raw):
                file_size = os.path.getsize(self.video_filename_raw) / (1024*1024)
                messagebox.showinfo("Информация",
                                  f"Чистое видео сохранено\nФайл: {self.video_filename_raw}\n"
                                  f"Размер: {file_size:.2f} МБ")

        self.video_record_btn.config(
            text="Запись видео",
            bg="#FF5722"
        )

    def start_recording(self):
        if not self.tracker.start():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return

        self.tracker.reset_state()  # <-- Добавлен сброс состояния

        self.is_recording = True
        self.record_btn.config(
            text="Остановить запись данных",
            bg="#f44336"
        )
        self.save_btn.config(state=tk.NORMAL)

        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False

        if self.is_video_recording:
            self.stop_video_recording()

        self.tracker.stop()

        self.record_btn.config(
            text="Начать запись данных",
            bg="#4CAF50"
        )

        black_img = np.zeros((450, 780, 3), dtype=np.uint8)
        self.show_frame(black_img)

        data_count = len(self.tracker.history['x'])
        if data_count > 0 and len(self.tracker.history['timestamps']) > 0:
            duration = self.tracker.history['timestamps'][-1]
            messagebox.showinfo("Информация",
                              f"Запись завершена\nСобрано точек данных: {data_count}\n"
                              f"Длительность: {duration:.2f} сек")

    def recording_loop(self):
        while self.is_recording and self.tracker.cap and self.tracker.cap.isOpened():
            success, frame = self.tracker.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            raw_frame = frame.copy()

            tracked_frame = self.tracker.process_frame(frame)

            if self.tracker.left_eye['x'] != 0:
                self.db.save_eye_data(
                    time.time() - self.tracker.start_time,
                    self.tracker.left_eye,
                    self.tracker.right_eye
                )

            display_frame = cv2.resize(tracked_frame, (780, 450))

            if self.is_video_recording and self.video_writer_raw:
                raw_display = cv2.resize(raw_frame, (780, 450))
                cv2.circle(raw_display, (50, 50), 15, (0, 0, 255), -1)
                self.video_writer_raw.write(raw_display)

            self.window.after(0, self.update_info)
            self.show_frame(display_frame)

        self.tracker.stop()
        if self.is_video_recording:
            self.stop_video_recording()

    def show_frame(self, frame):
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())

        self.video_label.config(image=img)
        self.video_label.image = img

    def update_info(self):
        left = self.tracker.left_eye
        right = self.tracker.right_eye

        self.info_label.config(
            text=f"Левый глаз: X: {left['x']}, Y: {left['y']} | "
                 f"Диаметр: {left['diameter']:.1f} px | "
                 f"Скорость: {left['speed']:.0f} px/s\n"
                 f"Правый глаз: X: {right['x']}, Y: {right['y']} | "
                 f"Диаметр: {right['diameter']:.1f} px | "
                 f"Скорость: {right['speed']:.0f} px/s"
        )

    def show_graphs(self):
        history = self.tracker.history

        if len(history['x']) < 2:
            messagebox.showwarning("Предупреждение", "Недостаточно данных для построения графиков")
            return

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history['x'], history['y'], 'b-', alpha=0.5, linewidth=0.5)
        plt.scatter(history['x'][0], history['y'][0], c='green', s=50, label='Старт')
        plt.scatter(history['x'][-1], history['y'][-1], c='red', s=50, label='Финиш')
        plt.title("Траектория взгляда (левый глаз)")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history['timestamps'], history['diameter'], 'g-', linewidth=1)
        plt.title("Диаметр зрачка (левый глаз, px)")
        plt.xlabel("Секунды")
        plt.ylabel("Диаметр (px)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(history['timestamps'], history['speed'], 'r-', linewidth=1)
        plt.title("Скорость движения (левый глаз, px/s)")
        plt.xlabel("Секунды")
        plt.ylabel("Скорость (px/s)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def go_back(self):
        if self.is_recording:
            self.stop_recording()

        if self.is_video_recording:
            self.stop_video_recording()

        self.db.close()
        self.window.destroy()
        self.main_window.deiconify()

    def on_closing(self):
        self.go_back()