import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from eye_tracker import EyeTracker
import threading
import json
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import math
from chart_utils import show_charts_window
from collections import deque

class VideoProcessor:
    """Класс для обработки видео с айтрекингом"""

    def __init__(self, video_path, db, user_id, video_id):
        self.video_path = video_path
        self.db = db
        self.user_id = user_id
        self.video_id = video_id
        self.tracker = EyeTracker()
        self.is_processing = False
        self.processing_thread = None
        self.results = {
            'left_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []},
            'right_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []}
        }
        self.start_time = 0
        # Создаем отдельное соединение для потока
        self.thread_db_conn = None

    def process_video(self, progress_callback=None, frame_callback=None):
        """Обработка видео с айтрекингом"""
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._process_video_thread,
            args=(progress_callback, frame_callback)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return self.processing_thread

    def _get_thread_connection(self):
        """Создает отдельное соединение с БД для потока"""
        if not self.thread_db_conn:
            self.thread_db_conn = sqlite3.connect(self.db.db_name)
            self.thread_db_conn.execute("PRAGMA foreign_keys = ON")
        return self.thread_db_conn

    def _update_video_data_in_thread(self, video_id, eye_tracking_data, charts_path):
        """Обновление данных видео в текущем потоке"""
        try:
            conn = self._get_thread_connection()
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(user_videos)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'eye_tracking_data' not in columns:
                if 'eye_tracking_data' not in columns:
                    cursor.execute("ALTER TABLE user_videos ADD COLUMN eye_tracking_data TEXT")
                if 'charts_path' not in columns:
                    cursor.execute("ALTER TABLE user_videos ADD COLUMN charts_path TEXT")
                if 'processed_at' not in columns:
                    cursor.execute("ALTER TABLE user_videos ADD COLUMN processed_at TIMESTAMP")

            cursor.execute('''
                UPDATE user_videos 
                SET eye_tracking_data = ?, 
                    charts_path = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (eye_tracking_data, charts_path, video_id))

            conn.commit()
            return True

        except sqlite3.Error as e:
            print(f"Ошибка при обновлении данных видео в потоке: {e}")
            return False

    def _process_video_thread(self, progress_callback=None, frame_callback=None):
        """Поток обработки видео"""
        # Скачиваем модель если нужно
        if not self.tracker.download_model():
            print("Ошибка загрузки модели")
            self.is_processing = False
            return

        # Открываем видео
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Ошибка открытия видео")
            self.is_processing = False
            return

        # Получаем параметры видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Настройка MediaPipe
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        base_options = python.BaseOptions(model_asset_path=self.tracker.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.tracker.min_detection_confidence,
            min_face_presence_confidence=self.tracker.min_presence_confidence,
            min_tracking_confidence=self.tracker.min_tracking_confidence
        )

        landmarker = vision.FaceLandmarker.create_from_options(options)

        # Очищаем результаты и сбрасываем состояние трекера
        self.results = {
            'left_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []},
            'right_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []}
        }
        self.tracker.reset_state()
        self.start_time = time.time()

        frame_count = 0

        while self.is_processing:
            success, frame = cap.read()
            if not success:
                break

            # Конвертация в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Получение временной метки
            timestamp_ms = int((frame_count / fps) * 1000)

            # Детекция лица
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]

                # ИЗМЕНЕНО: используем встроенный метод EyeTracker._get_eye_data для Starburst
                left_x, left_y, left_d = self.tracker._get_eye_data(face_landmarks, rgb_frame, 'left')
                right_x, right_y, right_d = self.tracker._get_eye_data(face_landmarks, rgb_frame, 'right')

                current_time = frame_count / fps

                # Применяем все фильтры через трекер
                self.tracker.filter_measurements(
                    left_x, left_y, left_d,
                    right_x, right_y, right_d,
                    current_time
                )

                # Сохраняем отфильтрованные результаты
                self.results['left_eye']['x'].append(self.tracker.left_eye['x'])
                self.results['left_eye']['y'].append(self.tracker.left_eye['y'])
                self.results['left_eye']['diameter'].append(self.tracker.left_eye['diameter'])
                self.results['left_eye']['speed'].append(self.tracker.left_eye['speed'])
                self.results['left_eye']['timestamps'].append(current_time)

                self.results['right_eye']['x'].append(self.tracker.right_eye['x'])
                self.results['right_eye']['y'].append(self.tracker.right_eye['y'])
                self.results['right_eye']['diameter'].append(self.tracker.right_eye['diameter'])
                self.results['right_eye']['speed'].append(self.tracker.right_eye['speed'])
                self.results['right_eye']['timestamps'].append(current_time)

            frame_count += 1

            # Обновление прогресса
            if progress_callback:
                progress = (frame_count / total_frames) * 100
                self._safe_callback(progress_callback, progress)

            # Отображение кадра
            if frame_callback:
                if detection_result and detection_result.face_landmarks:
                    frame = self._draw_eyes_on_frame(frame, detection_result.face_landmarks[0], width, height)
                self._safe_callback(frame_callback, frame)

        cap.release()
        landmarker.close()

        self.is_processing = False

    def _safe_callback(self, callback, *args, **kwargs):
        """Безопасный вызов callback из потока"""
        try:
            callback(*args, **kwargs)
        except Exception as e:
            print(f"Ошибка в callback: {e}")

    # ИЗМЕНЕНО: метод _get_eye_data удалён, т.к. используется из трекера

    def _draw_eyes_on_frame(self, frame, face_landmarks, width, height):
        """Рисует глаза на кадре, используя отфильтрованные координаты из трекера"""
        from utils import COLORS

        # Получаем отфильтрованные координаты
        left_x = self.tracker.left_eye['x']
        left_y = self.tracker.left_eye['y']
        left_d = self.tracker.left_eye['diameter']
        right_x = self.tracker.right_eye['x']
        right_y = self.tracker.right_eye['y']
        right_d = self.tracker.right_eye['diameter']

        # Рисуем левый глаз, только если координаты валидны
        if left_x != 0 and left_y != 0:
            cv2.circle(frame, (left_x, left_y), 3, COLORS['left_eye'], -1)
            radius = int(left_d / 2)
            if radius > 0:
                cv2.circle(frame, (left_x, left_y), radius, COLORS['left_eye'], 2)
            cv2.putText(frame, "L", (left_x - 20, left_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['left_eye'], 2)

        # Рисуем правый глаз
        if right_x != 0 and right_y != 0:
            cv2.circle(frame, (right_x, right_y), 3, COLORS['right_eye'], -1)
            radius = int(right_d / 2)
            if radius > 0:
                cv2.circle(frame, (right_x, right_y), radius, COLORS['right_eye'], 2)
            cv2.putText(frame, "R", (right_x + 10, right_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['right_eye'], 2)

        return frame

    def save_results_to_db(self):
        """Сохраняет результаты в базу данных (вызывается вручную)"""
        try:
            results_json = json.dumps(self.results)
            chart_paths = self._save_charts()

            if self._update_video_data_in_thread(self.video_id, results_json, json.dumps(chart_paths)):
                print(f"Результаты успешно сохранены для видео {self.video_id}")
                return True
            else:
                print(f"Ошибка при сохранении результатов для видео {self.video_id}")
                return False

        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")
            return False

    def _save_charts(self):
        """Создает и сохраняет графики"""
        charts = {}
        charts_dir = os.path.join('user_videos', f'user_{self.user_id}', 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        for eye in ['left_eye', 'right_eye']:
            data = self.results[eye]
            if len(data['timestamps']) < 2:
                continue

            try:
                fig, axes = plt.subplots(3, 1, figsize=(10, 8))

                axes[0].plot(data['x'], data['y'], 'b-', alpha=0.5)
                axes[0].set_title(f'{eye} - Траектория взгляда')
                axes[0].invert_yaxis()
                axes[0].grid(True)

                axes[1].plot(data['timestamps'], data['diameter'], 'g-')
                axes[1].set_title(f'{eye} - Диаметр зрачка')
                axes[1].set_xlabel('Секунды')
                axes[1].grid(True)

                axes[2].plot(data['timestamps'], data['speed'], 'r-')
                axes[2].set_title(f'{eye} - Скорость движения')
                axes[2].set_xlabel('Секунды')
                axes[2].grid(True)

                plt.tight_layout()

                chart_filename = f'{eye}_chart_{int(time.time())}.png'
                chart_path = os.path.join(charts_dir, chart_filename)
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close()

                charts[eye] = chart_path

            except Exception as e:
                print(f"Ошибка при создании графика для {eye}: {e}")

        return charts

    def stop_processing(self):
        """Остановка обработки"""
        self.is_processing = False


class VideoPlayerWindow:
    """Окно для воспроизведения видео с айтрекингом"""

    def __init__(self, parent, db, user_id, video_id, video_path, video_title):
        self.parent = parent
        self.db = db
        self.user_id = user_id
        self.video_id = video_id
        self.video_path = video_path
        self.video_title = video_title
        self.processor = None
        self.is_playing = False
        self.tracker_enabled = True
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.playing_video = False
        self.landmarker = None

        self.create_window()

    def create_window(self):
        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Айтрекинг видео: {self.video_title}")
        self.window.geometry("1200x800")
        self.window.grab_set()

        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.play_btn = tk.Button(
            control_frame,
            text="Запуск видео",
            font=("Arial", 11),
            bg="#4CAF50",
            fg="white",
            width=15,
            command=self.toggle_playback
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.toggle_tracker_btn = tk.Button(
            control_frame,
            text="Трекер вкл",
            font=("Arial", 11),
            bg="#2196F3",
            fg="white",
            width=15,
            command=self.toggle_tracker
        )
        self.toggle_tracker_btn.pack(side=tk.LEFT, padx=5)

        self.preview_charts_btn = tk.Button(
            control_frame,
            text="Предпросмотр",
            font=("Arial", 11),
            bg="#FF9800",
            fg="white",
            width=15,
            command=self.preview_charts,
            state=tk.DISABLED
        )
        self.preview_charts_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(
            control_frame,
            text="Сохранить в базу данных",
            font=("Arial", 11),
            bg="#4CAF50",
            fg="white",
            width=20,
            command=self.save_to_database,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.time_label = tk.Label(control_frame, text="00:00 / 00:00", font=("Arial", 11), width=15)
        self.time_label.pack(side=tk.RIGHT, padx=10)

        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        video_container = tk.Frame(content_frame)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_frame = tk.Label(video_container, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        timeline_frame = tk.Frame(video_container)
        timeline_frame.pack(fill=tk.X, pady=5)

        self.timeline_var = tk.DoubleVar()
        self.timeline = ttk.Scale(
            timeline_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            variable=self.timeline_var,
            command=self.on_timeline_change
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True)

        settings_frame = tk.LabelFrame(content_frame, text="Настройки",
                                       font=("Arial", 11, "bold"), width=300)
        settings_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        settings_frame.pack_propagate(False)

        # СГЛАЖИВАНИЕ ТРАЕКТОРИИ
        tk.Label(settings_frame, text="Сглаживание траектории:", font=("Arial", 10, "bold")).pack(pady=(10, 5))

        smoothing_frame = tk.Frame(settings_frame)
        smoothing_frame.pack(fill=tk.X, pady=2)

        self.smoothing_var = tk.DoubleVar(value=0.6)
        smoothing_scale = tk.Scale(
            smoothing_frame,
            from_=0.1, to=0.9,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=200,
            variable=self.smoothing_var,
            command=self.on_smoothing_change
        )
        smoothing_scale.pack(side=tk.LEFT)

        self.smoothing_value = tk.Label(smoothing_frame, text=f"{self.smoothing_var.get():.1f}", width=4)
        self.smoothing_value.pack(side=tk.LEFT, padx=5)

        # ФИЛЬТР РЕЗКИХ СКАЧКОВ
        jump_frame = tk.LabelFrame(settings_frame, text="Фильтр резких скачков",
                                   font=("Arial", 10, "bold"))
        jump_frame.pack(fill=tk.X, pady=10, padx=5)

        self.use_jump_filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            jump_frame,
            text="Включить фильтр",
            variable=self.use_jump_filter_var,
            font=("Arial", 9),
            command=self.on_jump_filter_change
        ).pack(anchor="w", pady=2)

        tk.Label(jump_frame, text="Макс. перемещение (пиксели):", font=("Arial", 9)).pack(anchor="w", pady=(5,0))

        jump_scale_frame = tk.Frame(jump_frame)
        jump_scale_frame.pack(fill=tk.X, pady=2)

        self.jump_limit_var = tk.DoubleVar(value=100)
        jump_scale = tk.Scale(
            jump_scale_frame,
            from_=20, to=300,
            resolution=10,
            orient=tk.HORIZONTAL,
            length=180,
            variable=self.jump_limit_var,
            command=self.on_jump_limit_change
        )
        jump_scale.pack(side=tk.LEFT)

        self.jump_limit_value = tk.Label(jump_scale_frame, text=f"{self.jump_limit_var.get():.0f}", width=4)
        self.jump_limit_value.pack(side=tk.LEFT, padx=5)

        # ФИЛЬТР ДРОЖАНИЯ
        shake_frame = tk.LabelFrame(settings_frame, text="Фильтр дрожания",
                                    font=("Arial", 10, "bold"))
        shake_frame.pack(fill=tk.X, pady=10, padx=5)

        self.use_shake_filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            shake_frame,
            text="Включить фильтр",
            variable=self.use_shake_filter_var,
            font=("Arial", 9),
            command=self.on_shake_filter_change
        ).pack(anchor="w", pady=2)

        tk.Label(shake_frame, text="Сила сглаживания:", font=("Arial", 9)).pack(anchor="w", pady=(5,0))

        shake_scale_frame = tk.Frame(shake_frame)
        shake_scale_frame.pack(fill=tk.X, pady=2)

        self.shake_strength_var = tk.IntVar(value=3)
        shake_scale = tk.Scale(
            shake_scale_frame,
            from_=3, to=9,
            resolution=2,
            orient=tk.HORIZONTAL,
            length=180,
            variable=self.shake_strength_var,
            command=self.on_shake_strength_change
        )
        shake_scale.pack(side=tk.LEFT)

        self.shake_strength_value = tk.Label(shake_scale_frame, text=f"{self.shake_strength_var.get()}", width=4)
        self.shake_strength_value.pack(side=tk.LEFT, padx=5)

        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)

        self.info_label = tk.Label(
            info_frame,
            font=("Arial", 10),
            justify=tk.LEFT
        )
        self.info_label.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_jump_limit_change(self, value):
        self.jump_limit_value.config(text=f"{float(value):.0f}")
        if self.processor and self.processor.tracker:
            self.processor.tracker.max_jump_distance = float(value)

    def on_jump_filter_change(self):
        if self.processor and self.processor.tracker:
            self.processor.tracker.use_outlier_filter = self.use_jump_filter_var.get()

    def on_shake_strength_change(self, value):
        val = int(float(value))
        self.shake_strength_value.config(text=f"{val}")
        if self.processor and self.processor.tracker:
            self.processor.tracker.median_filter_size = val
            if hasattr(self.processor.tracker, 'left_x_buffer'):
                self.processor.tracker.left_x_buffer = deque(maxlen=val)
                self.processor.tracker.left_y_buffer = deque(maxlen=val)
                self.processor.tracker.right_x_buffer = deque(maxlen=val)
                self.processor.tracker.right_y_buffer = deque(maxlen=val)
                self.processor.tracker.left_d_buffer = deque(maxlen=val)
                self.processor.tracker.right_d_buffer = deque(maxlen=val)

    def on_shake_filter_change(self):
        if self.processor and self.processor.tracker:
            self.processor.tracker.use_median_filter = self.use_shake_filter_var.get()

    def on_smoothing_change(self, value):
        self.smoothing_value.config(text=f"{float(value):.1f}")
        if self.processor and self.processor.tracker:
            self.processor.tracker.smoothing_factor = float(value)

    def toggle_tracker(self):
        self.tracker_enabled = not self.tracker_enabled
        if self.tracker_enabled:
            self.toggle_tracker_btn.config(text="Трекер вкл", bg="#2196F3")
        else:
            self.toggle_tracker_btn.config(text="Трекер выкл", bg="#9E9E9E")

    def toggle_playback(self):
        if not self.playing_video:
            self.load_video()
        else:
            self.stop_playback()

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.timeline.config(to=self.total_frames)

        self.processor = VideoProcessor(self.video_path, self.db, self.user_id, self.video_id)

        # Устанавливаем параметры фильтров из интерфейса
        if hasattr(self, 'use_jump_filter_var'):
            self.processor.tracker.use_outlier_filter = self.use_jump_filter_var.get()
            self.processor.tracker.max_jump_distance = self.jump_limit_var.get()

        if hasattr(self, 'use_shake_filter_var'):
            self.processor.tracker.use_median_filter = self.use_shake_filter_var.get()
            self.processor.tracker.median_filter_size = self.shake_strength_var.get()

        self.processor.tracker.smoothing_factor = self.smoothing_var.get()

        if not self.processor.tracker.download_model():
            messagebox.showerror("Ошибка", "Не удалось загрузить модель")
            return

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=self.processor.tracker.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.playing_video = True
        self.play_btn.config(text="Остановить", bg="#f44336")
        self.preview_charts_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        self.current_frame = 0
        self.processor.tracker.reset_state()  # сброс состояния перед началом

        self.processor.results = {
            'left_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []},
            'right_eye': {'x': [], 'y': [], 'diameter': [], 'speed': [], 'timestamps': []}
        }

        self.play_video()

    def stop_playback(self):
        self.playing_video = False
        self.play_btn.config(text="Запуск видео", bg="#4CAF50")

        if self.processor and self.processor.results:
            if self.processor.results['left_eye']['x'] or self.processor.results['right_eye']['x']:
                self.preview_charts_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.NORMAL)
                self.info_label.config(text=f"Обработано данных: {len(self.processor.results['left_eye']['x'])} кадров.")

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def play_video(self):
        if not self.playing_video or not self.cap:
            return

        import mediapipe as mp

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        success, frame = self.cap.read()
        if success and self.current_frame < self.total_frames:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps

            time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}"
            self.time_label.config(text=time_str)

            if self.tracker_enabled and self.landmarker:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((self.current_frame / self.fps) * 1000)
                detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection_result and detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]

                    # ИЗМЕНЕНО: используем встроенный метод EyeTracker._get_eye_data для Starburst
                    left_x, left_y, left_d = self.processor.tracker._get_eye_data(face_landmarks, rgb_frame, 'left')
                    right_x, right_y, right_d = self.processor.tracker._get_eye_data(face_landmarks, rgb_frame, 'right')

                    # Применяем фильтры через трекер
                    self.processor.tracker.filter_measurements(
                        left_x, left_y, left_d,
                        right_x, right_y, right_d,
                        current_time
                    )

                    frame = self._draw_eyes_on_frame(frame, detection_result.face_landmarks[0],
                                                     frame.shape[1], frame.shape[0])

                    # Сохраняем результаты
                    self.processor.results['left_eye']['x'].append(self.processor.tracker.left_eye['x'])
                    self.processor.results['left_eye']['y'].append(self.processor.tracker.left_eye['y'])
                    self.processor.results['left_eye']['diameter'].append(self.processor.tracker.left_eye['diameter'])
                    self.processor.results['left_eye']['speed'].append(self.processor.tracker.left_eye['speed'])
                    self.processor.results['left_eye']['timestamps'].append(current_time)

                    self.processor.results['right_eye']['x'].append(self.processor.tracker.right_eye['x'])
                    self.processor.results['right_eye']['y'].append(self.processor.tracker.right_eye['y'])
                    self.processor.results['right_eye']['diameter'].append(self.processor.tracker.right_eye['diameter'])
                    self.processor.results['right_eye']['speed'].append(self.processor.tracker.right_eye['speed'])
                    self.processor.results['right_eye']['timestamps'].append(current_time)

            self._display_frame(frame)
            self.timeline_var.set(self.current_frame)
            self.current_frame += 1
            delay = int(1000 / self.fps)
            self.window.after(delay, self.play_video)
        else:
            self.stop_playback()

    # ИЗМЕНЕНО: метод _get_eye_data удалён, т.к. используется из трекера

    def _draw_eyes_on_frame(self, frame, face_landmarks, width, height):
        """Рисует глаза на кадре, используя отфильтрованные координаты из трекера"""
        from utils import COLORS

        left_x = self.processor.tracker.left_eye['x']
        left_y = self.processor.tracker.left_eye['y']
        left_d = self.processor.tracker.left_eye['diameter']
        right_x = self.processor.tracker.right_eye['x']
        right_y = self.processor.tracker.right_eye['y']
        right_d = self.processor.tracker.right_eye['diameter']

        if left_x != 0 and left_y != 0:
            cv2.circle(frame, (left_x, left_y), 3, COLORS['left_eye'], -1)
            radius = int(left_d / 2)
            if radius > 0:
                cv2.circle(frame, (left_x, left_y), radius, COLORS['left_eye'], 2)
            cv2.putText(frame, "L", (left_x - 20, left_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['left_eye'], 2)

        if right_x != 0 and right_y != 0:
            cv2.circle(frame, (right_x, right_y), 3, COLORS['right_eye'], -1)
            radius = int(right_d / 2)
            if radius > 0:
                cv2.circle(frame, (right_x, right_y), radius, COLORS['right_eye'], 2)
            cv2.putText(frame, "R", (right_x + 10, right_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['right_eye'], 2)

        return frame

    def _display_frame(self, frame):
        from PIL import Image, ImageTk

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        display_width = 800
        display_height = 500
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))

        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_frame.config(image=imgtk)
        self.video_frame.image = imgtk

        if self.processor and self.processor.results['left_eye']['x']:
            left = self.processor.results['left_eye']
            if left['x']:
                info = f"Обработано кадров: {len(left['x'])}\n"
                info += f"Сведения:\n"
                info += f"  X: {left['x'][-1]}, Y: {left['y'][-1]}\n"
                info += f"  Диаметр: {left['diameter'][-1]:.1f}px\n"
                info += f"  Скорость: {left['speed'][-1]:.0f}px/s"
                self.info_label.config(text=info)

    def on_timeline_change(self, value):
        if self.cap and self.playing_video:
            self.current_frame = int(float(value))

    def preview_charts(self):
        if not self.processor or not self.processor.results:
            messagebox.showinfo("Информация", "Нет данных для отображения")
            return

        if not self.processor.results['left_eye']['x'] and not self.processor.results['right_eye']['x']:
            messagebox.showinfo("Информация", "Нет данных для отображения")
            return

        show_charts_window(self.window, self.video_title, self.processor.results, "Предпросмотр")

    def save_to_database(self):
        if not self.processor or not self.processor.results:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        if not self.processor.results['left_eye']['x'] and not self.processor.results['right_eye']['x']:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        if self.processor.save_results_to_db():
            messagebox.showinfo("Успех", "Данные успешно сохранены в базу данных")
            self.save_btn.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить данные")

    def on_closing(self):
        if self.playing_video:
            self.stop_playback()
        self.window.destroy()