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
        self.window.geometry("1100x750")  # Немного увеличил высоту
        center_window(self.window, 1100, 750)
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
        """Загрузка настроек из файла"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self.get_default_settings()
        return self.get_default_settings()
    
    def get_default_settings(self):
        """Настройки по умолчанию"""
        return {
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'min_presence_confidence': 0.5,
            'smoothing_factor': 0.7,
            'use_kalman': True,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1,
            'velocity_threshold': 100,
            'fixation_threshold': 20,
            'max_prediction_time': 0.3,
            'min_face_size': 100,
        }
    
    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {e}")
            return False
    
    def create_widgets(self):
        """Создание интерфейса"""
        
        # Основной контейнер
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель - видео
        left_panel = tk.Frame(main_frame, width=640, relief=tk.SUNKEN, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Заголовок видео
        video_title = tk.Label(
            left_panel,
            text="Тестовая камера",
            font=("Arial", 11, "bold"),
            bg="#2c3e50",
            fg="white",
            height=1
        )
        video_title.pack(fill=tk.X)

        # Контейнер для видео
        video_container = tk.Frame(left_panel, bg="black", height=420)
        video_container.pack(fill=tk.BOTH, expand=True)
        video_container.pack_propagate(False)

        # Область видео
        self.video_frame = tk.Label(
            video_container,
            bg="black"
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Кнопка управления камерой
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
        
        # Правая панель - настройки
        right_panel = tk.Frame(main_frame, width=430)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Заголовок настроек
        settings_title = tk.Label(
            right_panel,
            text="Параметры трекинга",
            font=("Arial", 11, "bold"),
            bg="#2c3e50",
            fg="white",
            height=1
        )
        settings_title.pack(fill=tk.X)
        
        # Контейнер с прокруткой для настроек
        canvas = tk.Canvas(right_panel, highlightthickness=0, bg="#f8f9fa")
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ======== ПАРАМЕТРЫ ДЕТЕКЦИИ ========
        detection_frame = tk.LabelFrame(
            scrollable_frame,
            text="Параметры детекции",
            font=("Arial", 10, "bold"),
            bg="#f8f9fa",
            padx=5,
            pady=5
        )
        detection_frame.pack(fill=tk.X, pady=2)
        
        self.create_slider_compact(
            detection_frame,
            "Detection confidence:",
            "min_detection_confidence",
            0.1, 1.0, 0.05
        )
        
        self.create_slider_compact(
            detection_frame,
            "Tracking confidence:",
            "min_tracking_confidence",
            0.1, 1.0, 0.05
        )
        
        self.create_slider_compact(
            detection_frame,
            "Presence confidence:",
            "min_presence_confidence",
            0.1, 1.0, 0.05
        )
        
        # ======== ПАРАМЕТРЫ СГЛАЖИВАНИЯ ========
        smoothing_frame = tk.LabelFrame(
            scrollable_frame,
            text="Сглаживание",
            font=("Arial", 10, "bold"),
            bg="#f8f9fa",
            padx=5,
            pady=5
        )
        smoothing_frame.pack(fill=tk.X, pady=2)
        
        self.create_slider_compact(
            smoothing_frame,
            "Smoothing factor:",
            "smoothing_factor",
            0.1, 0.95, 0.05
        )
        
        kalman_frame = tk.Frame(smoothing_frame, bg="#f8f9fa")
        kalman_frame.pack(fill=tk.X, pady=2)
        
        self.kalman_var = tk.BooleanVar(value=self.settings['use_kalman'])
        tk.Checkbutton(
            kalman_frame,
            text="Использовать фильтр Калмана",
            variable=self.kalman_var,
            font=("Arial", 9),
            bg="#f8f9fa",
            command=self.update_kalman_setting
        ).pack(anchor="w")
        
        # ======== ПАРАМЕТРЫ ФИЛЬТРА КАЛМАНА ========
        kalman_params_frame = tk.LabelFrame(
            scrollable_frame,
            text="Параметры фильтра Калмана",
            font=("Arial", 10, "bold"),
            bg="#f8f9fa",
            padx=5,
            pady=5
        )
        kalman_params_frame.pack(fill=tk.X, pady=2)
        
        self.create_slider_compact(
            kalman_params_frame,
            "Process noise:",
            "kalman_process_noise",
            0.001, 0.1, 0.001
        )
        
        self.create_slider_compact(
            kalman_params_frame,
            "Measurement noise:",
            "kalman_measurement_noise",
            0.01, 1.0, 0.01
        )
        
        self.create_slider_compact(
            kalman_params_frame,
            "Max prediction time:",
            "max_prediction_time",
            0.1, 1.0, 0.05
        )
        
        # ======== ПАРАМЕТРЫ АНАЛИЗА ========
        analysis_frame = tk.LabelFrame(
            scrollable_frame,
            text="Анализ движений",
            font=("Arial", 10, "bold"),
            bg="#f8f9fa",
            padx=5,
            pady=5
        )
        analysis_frame.pack(fill=tk.X, pady=2)
        
        self.create_slider_compact(
            analysis_frame,
            "Velocity threshold:",
            "velocity_threshold",
            50, 1000, 10
        )
        
        self.create_slider_compact(
            analysis_frame,
            "Fixation threshold:",
            "fixation_threshold",
            5, 100, 5
        )
        
        self.create_slider_compact(
            analysis_frame,
            "Min face size:",
            "min_face_size",
            50, 300, 10
        )
        
        # ======== КНОПКИ УПРАВЛЕНИЯ ========
        button_frame = tk.Frame(scrollable_frame, bg="#f0f0f0", height=80)
        button_frame.pack(fill=tk.X, pady=5)
        button_frame.pack_propagate(False)
        
        # Верхний ряд кнопок
        top_row = tk.Frame(button_frame, bg="#f0f0f0")
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
            text="Сохранить",
            font=("Arial", 9),
            bg="#3498db",
            fg="white",
            width=8,
            command=self.save_profile
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            top_row,
            text="Загрузить",
            font=("Arial", 9),
            bg="#9b59b6",
            fg="white",
            width=8,
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
        
        # Нижний ряд с кнопками навигации
        bottom_row = tk.Frame(button_frame, bg="#f0f0f0")
        bottom_row.pack(fill=tk.X, pady=5)
        
        # Кнопка "Вернуться в меню"
        tk.Button(
            bottom_row,
            text="Вернуться в меню",
            font=("Arial", 10, "bold"),
            bg="#FF9800",
            fg="white",
            width=20,
            height=1,
            command=self.go_back
        ).pack(side=tk.LEFT, padx=2)
        
        # # Увеличенная кнопка "Закрыть окно"
        # tk.Button(
        #     bottom_row,
        #     text="Закрыть окно",
        #     font=("Arial", 10, "bold"),
        #     bg="#e74c3c",
        #     fg="white",
        #     width=15,
        #     height=1,
        #     command=self.on_closing
        # ).pack(side=tk.RIGHT, padx=2)
    
    def create_slider_compact(self, parent, label, key, from_, to_, resolution):
        """Компактный ползунок для экономии места"""
        frame = tk.Frame(parent, bg="#f8f9fa")
        frame.pack(fill=tk.X, pady=1)
        
        # Метка
        tk.Label(
            frame,
            text=label,
            font=("Arial", 9),
            bg="#f8f9fa",
            width=18,
            anchor="w"
        ).pack(side=tk.LEFT)
        
        # Значение
        value_var = tk.DoubleVar(value=self.settings[key])
        value_label = tk.Label(
            frame,
            text=f"{self.settings[key]:.3f}",
            font=("Arial", 9, "bold"),
            bg="#f8f9fa",
            width=6
        )
        value_label.pack(side=tk.LEFT)
        
        # Ползунок
        slider = tk.Scale(
            frame,
            from_=from_,
            to=to_,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=value_var,
            length=120,
            bg="#f8f9fa",
            highlightthickness=0
        )
        slider.pack(side=tk.LEFT, padx=2)
        
        def update_value(*args):
            value_label.config(text=f"{value_var.get():.3f}")
            self.settings[key] = value_var.get()
        
        value_var.trace_add("write", update_value)
    
    def update_kalman_setting(self):
        """Обновление настройки фильтра Калмана"""
        self.settings['use_kalman'] = self.kalman_var.get()
    
    def toggle_camera(self):
        """Включение/выключение тестовой камеры"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Запуск тестовой камеры"""
        try:
            self.test_camera = cv2.VideoCapture(0)
            if not self.test_camera.isOpened():
                messagebox.showerror("Ошибка", "Не удалось открыть камеру")
                return
            
            self.camera_running = True
            self.start_camera_btn.config(
                text="Остановить камеру",
                bg="#e74c3c"
            )
            
            # Запускаем трекер если он еще не запущен
            if not self.tracker.is_running:
                self.tracker.start()
            
            # Запускаем поток камеры
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить камеру: {e}")
    
    def stop_camera(self):
        """Остановка тестовой камеры"""
        self.camera_running = False
        
        if self.test_camera:
            self.test_camera.release()
            self.test_camera = None
        
        self.start_camera_btn.config(
            text="Запустить камеру",
            bg="#27ae60"
        )
        
        # Показываем черный экран
        black_img = np.zeros((420, 640, 3), dtype=np.uint8)
        self.display_frame(black_img)
    
    def camera_loop(self):
        """Основной цикл камеры"""
        while self.camera_running and self.test_camera and self.test_camera.isOpened():
            success, frame = self.test_camera.read()
            if not success:
                continue
            
            # Зеркальное отображение
            frame = cv2.flip(frame, 1)
            
            # Обработка трекером
            frame = self.tracker.process_frame(frame)
            
            # Масштабируем для отображения
            display_frame = cv2.resize(frame, (640, 420))
            
            # Отображаем
            self.display_frame(display_frame)
            
            # Небольшая задержка
            time.sleep(0.03)
    
    def display_frame(self, frame):
        """Отображение кадра в интерфейсе"""
        if frame is None:
            return
        
        # Конвертация для tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        
        self.video_frame.config(image=img)
        self.video_frame.image = img
    
    def apply_settings(self):
        """Применение настроек"""
        if hasattr(self.tracker, 'update_all_settings'):
            if self.tracker.update_all_settings(self.settings):
                self.save_settings()
                messagebox.showinfo("Успех", "Настройки применены")
            else:
                messagebox.showerror("Ошибка", "Не удалось применить настройки")
        else:
            # Для простой версии трекера
            self.tracker.min_detection_confidence = self.settings['min_detection_confidence']
            self.tracker.smoothing_factor = self.settings['smoothing_factor']
            self.save_settings()
            messagebox.showinfo("Успех", "Настройки применены")
    
    def save_profile(self):
        """Сохранение профиля настроек"""
        profile_name = simpledialog.askstring("Сохранение профиля", 
                                             "Введите имя профиля:")
        if profile_name:
            filename = f"tracker_profile_{profile_name}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Успех", f"Профиль сохранен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить профиль: {e}")
    
    def load_profile(self):
        """Загрузка профиля настроек"""
        filename = filedialog.askopenfilename(
            title="Выберите файл профиля",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    self.settings.update(loaded_settings)
                    
                    # Обновляем UI
                    self.kalman_var.set(self.settings['use_kalman'])
                    self.refresh_ui()
                    
                messagebox.showinfo("Успех", "Профиль загружен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить профиль: {e}")
    
    def reset_to_defaults(self):
        """Сброс к настройкам по умолчанию"""
        if messagebox.askyesno("Подтверждение", "Сбросить все настройки?"):
            self.settings = self.get_default_settings()
            self.kalman_var.set(self.settings['use_kalman'])
            self.refresh_ui()
            messagebox.showinfo("Успех", "Настройки сброшены")
    
    def refresh_ui(self):
        """Обновление интерфейса после загрузки настроек"""
        if self.camera_running:
            self.stop_camera()
        
        self.window.destroy()
        MathSpecialistWindow(self.parent, self.tracker)
    
    def go_back(self):
        """Возврат в главное меню"""
        if self.camera_running:
            self.stop_camera()
        if self.tracker.is_running:
            self.tracker.stop()
        self.window.destroy()
        # Показываем родительское окно (главное меню)
        if self.parent:
            self.parent.deiconify()
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        self.go_back()