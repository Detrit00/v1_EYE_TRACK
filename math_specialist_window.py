# math_specialist_window.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import json
import os
import cv2
import numpy as np
import threading
import time
from utils import center_window
from eye_tracker import EyeTracker

class MathSpecialistWindow:
    """Окно для настройки алгоритмов трекинга с тестовой камерой"""

    def __init__(self, parent, tracker=None):
        self.parent = parent
        self.tracker = tracker if tracker else EyeTracker()
        self.window = tk.Toplevel(parent)
        self.window.title("Настройка алгоритмов трекинга")
        self.window.geometry("1200x800")
        center_window(self.window, 1200, 800)
        self.window.grab_set()
        self.window.focus_set()

        # Флаги состояния
        self.camera_running = False
        self.camera_thread = None
        self.test_camera = None

        # Загружаем сохраненные настройки
        self.settings_file = "tracker_settings.json"
        self.settings = self.load_settings()

        # Применяем настройки к трекеру
        if hasattr(self.tracker, 'update_all_settings'):
            self.tracker.update_all_settings(self.settings)

        self.create_widgets()

        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_settings(self):
        """Загрузка настроек из файла с дополнением отсутствующих ключей"""
        defaults = self.get_default_settings()
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Дополняем загруженные настройки значениями по умолчанию для отсутствующих ключей
                    for key, value in defaults.items():
                        if key not in loaded:
                            loaded[key] = value
                    return loaded
            except:
                return defaults
        return defaults

    def get_default_settings(self):
        """Настройки по умолчанию (полный список)"""
        return {
            # Детекция
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'min_presence_confidence': 0.5,
            'margin': 40,

            # Starburst
            'starburst_num_rays': 12,
            'starburst_ray_length': 30,
            'starburst_gradient_threshold': 15,
            'starburst_min_ellipse_points': 6,
            'use_gradient_direction': True,

            # Фильтрация
            'smoothing_factor': 0.7,
            'use_outlier_filter': True,
            'max_jump_distance': 100,
            'use_median_filter': True,
            'median_filter_size': 3,
            'use_kalman': True,
            'kalman_process_noise': 0.05,
            'kalman_measurement_noise': 0.2,

            # Классификация
            'velocity_threshold': 100,
            'min_fixation_duration': 0.04,
            'min_saccade_duration': 0.01,
        }

    def save_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")
            return False

    def create_widgets(self):
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая панель - видео
        left_panel = tk.Frame(main_frame, width=640, relief=tk.SUNKEN, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        left_panel.pack_propagate(False)

        video_title = tk.Label(
            left_panel,
            text="Тестовая камера",
            font=("Arial", 11, "bold"),
            bg="#2c3e50",
            fg="white",
            height=1
        )
        video_title.pack(fill=tk.X)

        video_container = tk.Frame(left_panel, bg="black", height=480)
        video_container.pack(fill=tk.BOTH, expand=True)
        video_container.pack_propagate(False)

        self.video_frame = tk.Label(video_container, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        camera_control_frame = tk.Frame(left_panel, height=40)
        camera_control_frame.pack(fill=tk.X, pady=2)
        camera_control_frame.pack_propagate(False)

        self.start_camera_btn = tk.Button(
            camera_control_frame,
            text="Запустить камеру",
            font=("Arial", 10),
            bg="#27ae60",
            fg="white",
            width=15,
            command=self.toggle_camera
        )
        self.start_camera_btn.pack(side=tk.LEFT, padx=2)

        # Правая панель - настройки (без фиксации размера)
        right_panel = tk.Frame(main_frame, width=500)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        # right_panel.pack_propagate(False)  # убираем фиксацию

        settings_title = tk.Label(
            right_panel,
            text="Параметры трекинга",
            font=("Arial", 11, "bold"),
            bg="#2c3e50",
            fg="white",
            height=1
        )
        settings_title.pack(fill=tk.X)

        # Контейнер с прокруткой
        canvas = tk.Canvas(right_panel, highlightthickness=0, bg="#f8f9fa")
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ===== Детекция =====
        detect_frame = tk.LabelFrame(scrollable_frame, text="Детекция глаз", font=("Arial", 10, "bold"), bg="#f8f9fa", padx=5, pady=5)
        detect_frame.pack(fill=tk.X, pady=2)

        self.create_slider(detect_frame, "Надёжность обнаружения лица:", "min_detection_confidence", 0.1, 1.0, 0.05)
        self.create_slider(detect_frame, "Надёжность отслеживания:", "min_tracking_confidence", 0.1, 1.0, 0.05)
        self.create_slider(detect_frame, "Надёжность присутствия:", "min_presence_confidence", 0.1, 1.0, 0.05)
        self.create_slider(detect_frame, "Отступ вокруг глаза (пикс):", "margin", 20, 100, 5)

        # ===== Starburst =====
        starburst_frame = tk.LabelFrame(scrollable_frame, text="Локализация зрачка (Starburst)", font=("Arial", 10, "bold"), bg="#f8f9fa", padx=5, pady=5)
        starburst_frame.pack(fill=tk.X, pady=2)

        self.create_slider(starburst_frame, "Количество лучей:", "starburst_num_rays", 6, 24, 1)
        self.create_slider(starburst_frame, "Длина луча (пикс):", "starburst_ray_length", 20, 60, 5)
        self.create_slider(starburst_frame, "Порог градиента:", "starburst_gradient_threshold", 10, 50, 1)
        self.create_slider(starburst_frame, "Мин. точек для эллипса:", "starburst_min_ellipse_points", 4, 12, 1)

        grad_dir_frame = tk.Frame(starburst_frame, bg="#f8f9fa")
        grad_dir_frame.pack(fill=tk.X, pady=2)
        self.use_grad_dir_var = tk.BooleanVar(value=self.settings['use_gradient_direction'])
        tk.Checkbutton(
            grad_dir_frame,
            text="Учитывать направление градиента",
            variable=self.use_grad_dir_var,
            bg="#f8f9fa",
            command=self.update_grad_dir
        ).pack(anchor="w")

        # ===== Фильтрация =====
        filt_frame = tk.LabelFrame(scrollable_frame, text="Фильтрация", font=("Arial", 10, "bold"), bg="#f8f9fa", padx=5, pady=5)
        filt_frame.pack(fill=tk.X, pady=2)

        self.create_slider(filt_frame, "Сглаживание траектории:", "smoothing_factor", 0.1, 0.95, 0.05)

        outlier_frame = tk.Frame(filt_frame, bg="#f8f9fa")
        outlier_frame.pack(fill=tk.X, pady=2)
        self.use_outlier_var = tk.BooleanVar(value=self.settings['use_outlier_filter'])
        tk.Checkbutton(
            outlier_frame,
            text="Фильтр резких скачков",
            variable=self.use_outlier_var,
            bg="#f8f9fa",
            command=self.update_outlier_filter
        ).pack(anchor="w")
        self.create_slider(filt_frame, "Макс. допустимый скачок (пикс):", "max_jump_distance", 20, 300, 10)

        median_frame = tk.Frame(filt_frame, bg="#f8f9fa")
        median_frame.pack(fill=tk.X, pady=2)
        self.use_median_var = tk.BooleanVar(value=self.settings['use_median_filter'])
        tk.Checkbutton(
            median_frame,
            text="Медианный фильтр",
            variable=self.use_median_var,
            bg="#f8f9fa",
            command=self.update_median_filter
        ).pack(anchor="w")
        self.create_slider(filt_frame, "Размер окна медианного фильтра (нечётное):", "median_filter_size", 3, 9, 2)

        kalman_frame = tk.Frame(filt_frame, bg="#f8f9fa")
        kalman_frame.pack(fill=tk.X, pady=2)
        self.use_kalman_var = tk.BooleanVar(value=self.settings['use_kalman'])
        tk.Checkbutton(
            kalman_frame,
            text="Фильтр Калмана",
            variable=self.use_kalman_var,
            bg="#f8f9fa",
            command=self.update_kalman
        ).pack(anchor="w")
        self.create_slider(filt_frame, "Шум процесса (Q):", "kalman_process_noise", 0.001, 0.2, 0.001)
        self.create_slider(filt_frame, "Шум измерения (R):", "kalman_measurement_noise", 0.01, 1.0, 0.01)

        # ===== Классификация =====
        class_frame = tk.LabelFrame(scrollable_frame, text="Классификация движений (I-VT)", font=("Arial", 10, "bold"), bg="#f8f9fa", padx=5, pady=5)
        class_frame.pack(fill=tk.X, pady=2)

        self.create_slider(class_frame, "Порог скорости (пикс/сек):", "velocity_threshold", 50, 500, 10)
        self.create_slider(class_frame, "Мин. длительность фиксации (сек):", "min_fixation_duration", 0.02, 0.5, 0.01)
        self.create_slider(class_frame, "Мин. длительность саккады (сек):", "min_saccade_duration", 0.005, 0.1, 0.005)

        # ===== Кнопки =====
        button_frame = tk.Frame(scrollable_frame, bg="#f8f9fa")
        button_frame.pack(fill=tk.X, pady=5)

        top_row = tk.Frame(button_frame, bg="#f8f9fa")
        top_row.pack(fill=tk.X, pady=2)

        tk.Button(
            top_row,
            text="Применить",
            font=("Arial", 9, "bold"),
            bg="#27ae60",
            fg="white",
            width=10,
            command=self.apply_settings
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            top_row,
            text="Сохранить профиль",
            font=("Arial", 9),
            bg="#3498db",
            fg="white",
            width=12,
            command=self.save_profile
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            top_row,
            text="Загрузить профиль",
            font=("Arial", 9),
            bg="#9b59b6",
            fg="white",
            width=12,
            command=self.load_profile
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            top_row,
            text="Сброс",
            font=("Arial", 9),
            bg="#e67e22",
            fg="white",
            width=6,
            command=self.reset_to_defaults
        ).pack(side=tk.LEFT, padx=2)

        bottom_row = tk.Frame(button_frame, bg="#f8f9fa")
        bottom_row.pack(fill=tk.X, pady=5)

        tk.Button(
            bottom_row,
            text="Вернуться в меню",
            font=("Arial", 10, "bold"),
            bg="#FF9800",
            fg="white",
            width=20,
            command=self.go_back
        ).pack(side=tk.LEFT, padx=2)

        # Принудительное обновление прокрутки после отрисовки
        self.window.update_idletasks()
        self.window.after(50, lambda: canvas.configure(scrollregion=canvas.bbox("all")))

    def create_slider(self, parent, label, key, from_, to_, resolution):
        frame = tk.Frame(parent, bg="#f8f9fa")
        frame.pack(fill=tk.X, pady=2)

        tk.Label(frame, text=label, font=("Arial", 9), bg="#f8f9fa", width=30, anchor="w").pack(side=tk.LEFT)

        # Если ключа нет в settings, добавим его со значением по умолчанию
        if key not in self.settings:
            defaults = self.get_default_settings()
            self.settings[key] = defaults.get(key, 0)

        value_var = tk.DoubleVar(value=self.settings[key])
        value_label = tk.Label(frame, text=f"{self.settings[key]:.3f}", font=("Arial", 9, "bold"), bg="#f8f9fa", width=6)
        value_label.pack(side=tk.LEFT)

        slider = tk.Scale(
            frame,
            from_=from_, to=to_, resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=value_var,
            length=150,
            bg="#f8f9fa",
            highlightthickness=0
        )
        slider.pack(side=tk.LEFT, padx=2)

        def update_value(*args):
            value_label.config(text=f"{value_var.get():.3f}")
            self.settings[key] = value_var.get()

        value_var.trace_add("write", update_value)

    def update_grad_dir(self):
        self.settings['use_gradient_direction'] = self.use_grad_dir_var.get()

    def update_outlier_filter(self):
        self.settings['use_outlier_filter'] = self.use_outlier_var.get()

    def update_median_filter(self):
        self.settings['use_median_filter'] = self.use_median_var.get()

    def update_kalman(self):
        self.settings['use_kalman'] = self.use_kalman_var.get()

    def toggle_camera(self):
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            self.test_camera = cv2.VideoCapture(0)
            if not self.test_camera.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру")
                return

            self.camera_running = True
            self.start_camera_btn.config(text="Остановить камеру", bg="#e74c3c")

            if not self.tracker.is_running:
                self.tracker.start()

            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить камеру: {e}")

    def stop_camera(self):
        self.camera_running = False
        if self.test_camera:
            self.test_camera.release()
            self.test_camera = None
        self.start_camera_btn.config(text="Запустить камеру", bg="#27ae60")
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.display_frame(black_img)

    def camera_loop(self):
        while self.camera_running and self.test_camera and self.test_camera.isOpened():
            success, frame = self.test_camera.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            frame = self.tracker.process_frame(frame)
            display_frame = cv2.resize(frame, (640, 480))
            self.display_frame(display_frame)
            time.sleep(0.03)

    def display_frame(self, frame):
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        self.video_frame.config(image=img)
        self.video_frame.image = img

    def apply_settings(self):
        if self.settings['median_filter_size'] % 2 == 0:
            self.settings['median_filter_size'] += 1
        if hasattr(self.tracker, 'update_all_settings'):
            if self.tracker.update_all_settings(self.settings):
                self.save_settings()
                messagebox.showinfo("Успех", "Настройки применены")
            else:
                messagebox.showerror("Ошибка", "Не удалось применить настройки")
        else:
            self.save_settings()
            messagebox.showinfo("Успех", "Настройки применены")

    def save_profile(self):
        profile_name = simpledialog.askstring("Сохранение профиля", "Введите имя профиля:")
        if profile_name:
            filename = f"tracker_profile_{profile_name}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Успех", f"Профиль сохранён как {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить профиль: {e}")

    def load_profile(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл профиля",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    for key in self.settings:
                        if key in loaded_settings:
                            self.settings[key] = loaded_settings[key]
                    self.use_grad_dir_var.set(self.settings['use_gradient_direction'])
                    self.use_outlier_var.set(self.settings['use_outlier_filter'])
                    self.use_median_var.set(self.settings['use_median_filter'])
                    self.use_kalman_var.set(self.settings['use_kalman'])
                    self.refresh_ui()
                messagebox.showinfo("Успех", "Профиль загружен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить профиль: {e}")

    def reset_to_defaults(self):
        if messagebox.askyesno("Подтверждение", "Сбросить все настройки?"):
            self.settings = self.get_default_settings()
            self.use_grad_dir_var.set(self.settings['use_gradient_direction'])
            self.use_outlier_var.set(self.settings['use_outlier_filter'])
            self.use_median_var.set(self.settings['use_median_filter'])
            self.use_kalman_var.set(self.settings['use_kalman'])
            self.refresh_ui()
            messagebox.showinfo("Успех", "Настройки сброшены")

    def refresh_ui(self):
        if self.camera_running:
            self.stop_camera()
        self.window.destroy()
        MathSpecialistWindow(self.parent, self.tracker)

    def go_back(self):
        if self.camera_running:
            self.stop_camera()
        if self.tracker.is_running:
            self.tracker.stop()
        self.window.destroy()
        if self.parent:
            self.parent.deiconify()

    def on_closing(self):
        self.go_back()