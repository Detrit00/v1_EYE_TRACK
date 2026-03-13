import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3
import threading
from collections import deque
from utils import center_window

class MathematicalEyeTracker:
    """
    Математическая модель для количественной оценки параметров движения зрачка
    С поддержкой отслеживания обоих глаз
    """
    
    def __init__(self):
        # Параметры для каскадов Хаара
        self.eye_cascade = None
        self.haar_thresholds = {
            'edge': 50,      # Порог для краевых признаков θ_k
            'line': 40,      # Порог для линейных признаков
            'center': 45     # Порог для центральных признаков
        }
        
        # Параметры для градиентного анализа
        self.gaussian_sigma = 1.5
        self.gaussian_kernel_size = 5
        self.pupil_radius_range = (5, 30)
        
        # Параметры фильтра Калмана для каждого глаза
        self.kalman_filters = {}  # Словарь для хранения фильтров по ID глаза
        self.delta_t = 1/30
        
        # Матрицы фильтра Калмана
        self.init_kalman_matrices()
        
        # История для стабильности
        self.eye_positions = {}  # Хранение последних позиций глаз
        self.position_history = {}  # История для сглаживания
        self.max_history = 5  # Максимальная длина истории
        
        # Идентификаторы глаз
        self.next_eye_id = 0
        self.eye_id_map = {}  # Сопоставление позиций с ID
        
        # Инициализация OpenCV детектора
        self.init_haar_cascade()
    
    def init_kalman_matrices(self):
        """Инициализация матриц для фильтра Калмана"""
        # Матрица перехода состояний A
        self.A = np.array([
            [1, 0, self.delta_t, 0],
            [0, 1, 0, self.delta_t],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Матрица наблюдений H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Ковариация шума модели
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        
        # Ковариация шума измерений
        self.R = np.eye(2, dtype=np.float32) * 0.1
    
    def init_haar_cascade(self):
        """Инициализация каскадов Хаара для обнаружения глаз"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.eye_cascade.empty():
            print("Ошибка загрузки каскада Хаара")
    
    def get_eye_id(self, x, y, w, h):
        """
        Присваивает ID глазу на основе его позиции и размера
        Используется для стабильного отслеживания каждого глаза
        """
        eye_center = (x + w//2, y + h//2)
        eye_size = w * h
        
        # Ищем существующий глаз поблизости
        min_distance = 50  # Максимальное расстояние для соответствия
        matched_id = None
        
        for eye_id, prev_data in self.eye_positions.items():
            prev_center = prev_data['center']
            prev_size = prev_data['size']
            
            # Вычисляем расстояние между центрами
            distance = math.hypot(eye_center[0] - prev_center[0], 
                                 eye_center[1] - prev_center[1])
            
            # Проверяем размер (глаза не должны сильно меняться)
            size_diff = abs(eye_size - prev_size) / prev_size
            
            if distance < min_distance and size_diff < 0.3:
                matched_id = eye_id
                break
        
        if matched_id is None:
            # Новый глаз
            matched_id = self.next_eye_id
            self.next_eye_id += 1
            
            # Создаем новый фильтр Калмана для этого глаза
            self.kalman_filters[matched_id] = {
                'state': np.zeros((4, 1), dtype=np.float32),
                'covariance': np.eye(4, dtype=np.float32) * 100
            }
            
            # Создаем историю позиций
            self.position_history[matched_id] = deque(maxlen=self.max_history)
        
        # Обновляем позицию глаза
        self.eye_positions[matched_id] = {
            'center': eye_center,
            'size': eye_size,
            'bbox': (x, y, w, h),
            'last_seen': time.time()
        }
        
        return matched_id
    
    def detect_eyes(self, frame):
        """
        Обнаружение областей с глазами с присвоением ID
        Возвращает список глаз с ID и координатами
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение глаз
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Очищаем старые глаза (не видели больше 1 секунды)
        current_time = time.time()
        eyes_to_remove = []
        for eye_id, data in self.eye_positions.items():
            if current_time - data['last_seen'] > 1.0:
                eyes_to_remove.append(eye_id)
        
        for eye_id in eyes_to_remove:
            del self.eye_positions[eye_id]
            if eye_id in self.kalman_filters:
                del self.kalman_filters[eye_id]
            if eye_id in self.position_history:
                del self.position_history[eye_id]
        
        # Присваиваем ID каждому обнаруженному глазу
        detected_eyes = []
        for (x, y, w, h) in eyes:
            eye_id = self.get_eye_id(x, y, w, h)
            detected_eyes.append({
                'id': eye_id,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2)
            })
        
        return detected_eyes
    
    def gaussian_filter(self, image):
        """Фильтр Гаусса для уменьшения шума"""
        kernel_size = self.gaussian_kernel_size
        sigma = self.gaussian_sigma
        
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        filtered = cv2.filter2D(image, -1, kernel)
        return filtered
    
    def compute_gradient(self, image):
        """Вычисление градиента с помощью операторов Собеля"""
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return sobel_x, sobel_y
    
    def compute_integral_gradient(self, gradient_magnitude):
        """Интегральное представление градиентов"""
        integral_grad = cv2.integral(gradient_magnitude.astype(np.float32))
        return integral_grad
    
    def accumulate_gradients(self, integral_grad, x, y, radius):
        """Накопление суммы градиентов в окрестности точки"""
        half_size = radius
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(integral_grad.shape[1] - 1, x + half_size)
        y2 = min(integral_grad.shape[0] - 1, y + half_size)
        
        sum_gradients = (integral_grad[y2, x2] + integral_grad[y1, x1] - 
                         integral_grad[y1, x2] - integral_grad[y2, x1])
        
        return sum_gradients
    
    def find_pupil_center(self, eye_roi, roi_position, eye_id):
        """
        Поиск центра зрачка с учетом предыдущих положений для стабильности
        """
        # Предварительная обработка
        filtered = self.gaussian_filter(eye_roi)
        
        # Градиенты
        gx, gy = self.compute_gradient(filtered)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Интегральное представление
        integral_grad = self.compute_integral_gradient(magnitude)
        
        # Поиск максимума с учетом предыдущего положения
        height, width = eye_roi.shape
        best_score = -1
        best_x, best_y = 0, 0
        best_r = self.pupil_radius_range[0]
        
        # Если есть история, начинаем поиск с последней известной позиции
        search_radius = 20  # Радиус поиска от предыдущей позиции
        start_x = 0
        start_y = 0
        end_x = width
        end_y = height
        
        if eye_id in self.position_history and len(self.position_history[eye_id]) > 0:
            last_pos = self.position_history[eye_id][-1]
            # Ограничиваем область поиска
            start_x = max(0, last_pos[0] - search_radius)
            start_y = max(0, last_pos[1] - search_radius)
            end_x = min(width, last_pos[0] + search_radius)
            end_y = min(height, last_pos[1] + search_radius)
        
        # Поиск в ограниченной области
        for radius in range(self.pupil_radius_range[0], self.pupil_radius_range[1], 2):
            for y in range(start_y + radius, end_y - radius, 2):
                for x in range(start_x + radius, end_x - radius, 2):
                    score = self.accumulate_gradients(integral_grad, x, y, radius)
                    
                    if score > best_score:
                        best_score = score
                        best_x, best_y = x, y
                        best_r = radius
        
        # Сохраняем в историю
        if eye_id in self.position_history:
            self.position_history[eye_id].append((best_x, best_y))
        
        # Глобальные координаты
        global_x = roi_position[0] + best_x
        global_y = roi_position[1] + best_y
        
        return global_x, global_y, best_r * 2
    
    def kalman_predict(self, eye_id):
        """Прогнозирование для конкретного глаза"""
        if eye_id not in self.kalman_filters:
            return None, None
        
        kalman = self.kalman_filters[eye_id]
        
        # x̂ₖ₊₁⁻ = A x̂ₖ
        predicted_state = self.A @ kalman['state']
        
        # Pₖ₊₁⁻ = A Pₖ Aᵀ + Q
        predicted_covariance = self.A @ kalman['covariance'] @ self.A.T + self.Q
        
        return predicted_state, predicted_covariance
    
    def kalman_update(self, eye_id, measurement):
        """Коррекция для конкретного глаза"""
        if eye_id not in self.kalman_filters:
            # Создаем новый фильтр
            self.kalman_filters[eye_id] = {
                'state': np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32),
                'covariance': np.eye(4, dtype=np.float32) * 100
            }
            return self.kalman_filters[eye_id]['state']
        
        kalman = self.kalman_filters[eye_id]
        
        # Прогнозирование
        pred_state, pred_cov = self.kalman_predict(eye_id)
        
        if pred_state is None:
            return kalman['state']
        
        # Инновация
        innovation = measurement - self.H @ pred_state
        
        # Ковариация инновации
        innovation_cov = self.H @ pred_cov @ self.H.T + self.R
        
        # Коэффициент Калмана
        kalman_gain = pred_cov @ self.H.T @ np.linalg.inv(innovation_cov)
        
        # Обновление состояния
        kalman['state'] = pred_state + kalman_gain @ innovation
        
        # Обновление ковариации
        kalman['covariance'] = (np.eye(4) - kalman_gain @ self.H) @ pred_cov
        
        return kalman['state']


class RecordWindow:
    def __init__(self, window, main_window):
        self.window = window
        self.main_window = main_window
        self.window.title("Запись видео - Математический айтрекинг (оба глаза)")
        
        # Центрируем окно
        center_window(self.window, 800, 600)
        
        # Переменные для записи
        self.is_recording = False
        self.recording_thread = None
        self.cap = None
        
        # Инициализация математической модели
        self.tracker = MathematicalEyeTracker()
        
        # Данные для каждого глаза
        self.eyes_data = {}  # Словарь с данными по каждому глазу
        
        # История для графиков
        self.HISTORY_TIMESTAMPS = []
        self.HISTORY_EYES = {}  # Структура: {eye_id: {'x':[], 'y':[], 'fx':[], 'fy':[], 'd':[], 's':[]}}
        
        # Для расчета скорости
        self.PREV_DATA = {}  # {eye_id: {'x':, 'y':, 'time':}}
        self.START_TIME = 0
        
        # База данных
        self.DB_FILE = "eye_data_both_eyes.db"
        self.DB_CONNECTION = None
        self.DB_CURSOR = None
        
        # Цвета для разных глаз
        self.EYE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 165, 255), (255, 0, 255)]  # Синий, зеленый, оранжевый, розовый
        
        # Создаем интерфейс
        self.setup_ui()
        
        # Инициализируем БД
        self.init_db()
        
        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Создание интерфейса"""
        # Верхняя панель с кнопками
        control_frame = tk.Frame(self.window, bg="#f0f0f0", height=60)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Кнопка записи
        self.record_btn = tk.Button(
            control_frame,
            text="Начать запись",
            font=("Arial", 11, "bold"),
            width=15,
            height=1,
            bg="#4CAF50",
            fg="white",
            command=self.toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка показа графиков
        self.graph_btn = tk.Button(
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
        self.graph_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка "Назад"
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
        
        # Статус бар
        status_frame = tk.Frame(self.window, bg="#e0e0e0", height=30)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="Статус: Готов к записи",
            bg="#e0e0e0",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.eyes_count_label = tk.Label(
            status_frame,
            text="Глаз: 0",
            bg="#e0e0e0",
            font=("Arial", 10)
        )
        self.eyes_count_label.pack(side=tk.RIGHT, padx=10)
        
        self.data_label = tk.Label(
            status_frame,
            text="Данных: 0",
            bg="#e0e0e0",
            font=("Arial", 10)
        )
        self.data_label.pack(side=tk.RIGHT, padx=10)
        
        # Основная область для видео
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
        
        # Информационная панель для каждого глаза
        self.info_frame = tk.Frame(self.window, bg="#f0f0f0", height=80)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_labels = {}  # Словарь для меток информации по глазам
        
        # Пояснение
        explain_label = tk.Label(
            self.info_frame,
            text="Каждый глаз отслеживается отдельно со своим ID и фильтром Калмана",
            bg="#f0f0f0",
            font=("Arial", 9),
            fg="gray"
        )
        explain_label.pack()
    
    def update_info_display(self):
        """Обновление отображения информации по глазам"""
        # Очищаем старые метки
        for label in self.info_labels.values():
            label.destroy()
        self.info_labels.clear()
        
        # Создаем метки для каждого глаза
        for eye_id, data in self.eyes_data.items():
            color = self.EYE_COLORS[eye_id % len(self.EYE_COLORS)]
            color_hex = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'  # BGR to RGB hex
            
            frame = tk.Frame(self.info_frame, bg="#ffffff", relief=tk.RIDGE, bd=1)
            frame.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)
            
            # Заголовок с ID глаза
            title = tk.Label(
                frame,
                text=f"Глаз {eye_id}",
                bg=color_hex,
                fg="white",
                font=("Arial", 9, "bold")
            )
            title.pack(fill=tk.X)
            
            # Данные
            info = tk.Label(
                frame,
                text=f"Raw: ({data['x']}, {data['y']})\n"
                     f"Flt: ({data['fx']:.0f}, {data['fy']:.0f})\n"
                     f"D: {data['d']:.0f}px | S: {data['s']:.0f}px/s",
                bg="#ffffff",
                font=("Arial", 8),
                justify=tk.LEFT
            )
            info.pack(padx=2, pady=2)
            
            self.info_labels[eye_id] = frame
        
        self.eyes_count_label.config(text=f"Глаз: {len(self.eyes_data)}")
        self.data_label.config(text=f"Данных: {len(self.HISTORY_TIMESTAMPS)}")
    
    def init_db(self):
        """Инициализация базы данных для обоих глаз"""
        try:
            self.DB_CONNECTION = sqlite3.connect(self.DB_FILE, check_same_thread=False)
            self.DB_CURSOR = self.DB_CONNECTION.cursor()
            
            query = """
            CREATE TABLE IF NOT EXISTS pupils_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                eye_id INTEGER,
                pos_x INTEGER,
                pos_y INTEGER,
                filtered_x REAL,
                filtered_y REAL,
                diameter_px REAL,
                speed_px_sec REAL,
                velocity_x REAL,
                velocity_y REAL
            )
            """
            self.DB_CURSOR.execute(query)
            self.DB_CONNECTION.commit()
            print(f"База данных '{self.DB_FILE}' подключена")
        except sqlite3.Error as e:
            print(f"Ошибка базы данных: {e}")
    
    def save_to_db(self, t, eye_id, x, y, fx, fy, d, s, vx, vy):
        """Сохранение в базу данных"""
        if self.DB_CONNECTION:
            try:
                query = """INSERT INTO pupils_log 
                          (timestamp, eye_id, pos_x, pos_y, filtered_x, filtered_y, 
                           diameter_px, speed_px_sec, velocity_x, velocity_y) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                self.DB_CURSOR.execute(query, (t, eye_id, x, y, fx, fy, d, s, vx, vy))
                self.DB_CONNECTION.commit()
            except sqlite3.Error as e:
                print(f"Ошибка записи: {e}")
    
    def toggle_recording(self):
        """Запуск/остановка записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Запуск айтрекинга"""
        self.is_recording = True
        self.record_btn.config(
            text="⏸ Остановить запись",
            bg="#f44336"
        )
        self.status_label.config(text="Статус: Запись...")
        self.graph_btn.config(state=tk.NORMAL)
        
        # Очистка данных
        self.eyes_data.clear()
        self.HISTORY_TIMESTAMPS.clear()
        self.HISTORY_EYES.clear()
        self.PREV_DATA.clear()
        
        # Запуск потока
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Остановка записи"""
        self.is_recording = False
        self.record_btn.config(
            text="Начать запись",
            bg="#4CAF50"
        )
        self.status_label.config(text="Статус: Запись остановлена")
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def recording_loop(self):
        """Основной цикл записи с отслеживанием обоих глаз"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.START_TIME = time.time()
        
        while self.is_recording and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            rel_time = current_time - self.START_TIME
            
            # Обнаружение глаз с присвоением ID
            detected_eyes = self.tracker.detect_eyes(frame)
            
            # Обновляем данные по каждому глазу
            current_eyes = {}
            
            for eye in detected_eyes:
                eye_id = eye['id']
                x, y, w, h = eye['bbox']
                
                # Выделяем область глаза
                eye_roi = frame[y:y+h, x:x+w]
                if eye_roi.size == 0:
                    continue
                    
                gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                
                # Поиск центра зрачка
                pupil_x, pupil_y, diameter = self.tracker.find_pupil_center(
                    gray_eye, (x, y), eye_id
                )
                
                # Фильтрация Калмана
                measurement = np.array([[pupil_x], [pupil_y]], dtype=np.float32)
                filtered_state = self.tracker.kalman_update(eye_id, measurement)
                
                filtered_x = filtered_state[0, 0]
                filtered_y = filtered_state[1, 0]
                vel_x = filtered_state[2, 0]
                vel_y = filtered_state[3, 0]
                
                # Расчет скорости
                speed = 0
                if eye_id in self.PREV_DATA:
                    prev = self.PREV_DATA[eye_id]
                    dist = math.hypot(pupil_x - prev['x'], pupil_y - prev['y'])
                    time_diff = rel_time - prev['time']
                    if time_diff > 0:
                        speed = dist / time_diff
                
                # Сохраняем для следующего кадра
                self.PREV_DATA[eye_id] = {
                    'x': pupil_x,
                    'y': pupil_y,
                    'time': rel_time
                }
                
                # Сохраняем текущие данные
                current_eyes[eye_id] = {
                    'x': pupil_x,
                    'y': pupil_y,
                    'fx': filtered_x,
                    'fy': filtered_y,
                    'd': diameter,
                    's': speed,
                    'vx': vel_x,
                    'vy': vel_y,
                    'bbox': (x, y, w, h)
                }
                
                # Сохраняем в историю для графиков
                if eye_id not in self.HISTORY_EYES:
                    self.HISTORY_EYES[eye_id] = {
                        'x': [], 'y': [], 'fx': [], 'fy': [],
                        'd': [], 's': [], 'time': []
                    }
                
                self.HISTORY_EYES[eye_id]['x'].append(pupil_x)
                self.HISTORY_EYES[eye_id]['y'].append(pupil_y)
                self.HISTORY_EYES[eye_id]['fx'].append(filtered_x)
                self.HISTORY_EYES[eye_id]['fy'].append(filtered_y)
                self.HISTORY_EYES[eye_id]['d'].append(diameter)
                self.HISTORY_EYES[eye_id]['s'].append(speed)
                self.HISTORY_EYES[eye_id]['time'].append(rel_time)
                
                # Сохраняем в БД
                self.save_to_db(rel_time, eye_id, pupil_x, pupil_y, 
                               filtered_x, filtered_y, diameter, speed, vel_x, vel_y)
                
                # Визуализация
                color = self.EYE_COLORS[eye_id % len(self.EYE_COLORS)]
                
                # Прямоугольник вокруг глаза
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Измеренное положение (круг)
                cv2.circle(frame, (pupil_x, pupil_y), 3, color, -1)
                cv2.circle(frame, (pupil_x, pupil_y), int(diameter/2), color, 1)
                
                # Отфильтрованное положение (крест)
                cv2.drawMarker(frame, (int(filtered_x), int(filtered_y)), 
                              color, cv2.MARKER_CROSS, 10, 2)
                
                # ID глаза
                cv2.putText(frame, f"Eye {eye_id}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Скорость
                cv2.putText(frame, f"{speed:.0f} px/s", (x, y+h+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Обновляем данные для отображения
            self.eyes_data = current_eyes
            
            if rel_time % 1 < 0.1:  # Примерно раз в секунду
                self.HISTORY_TIMESTAMPS.append(rel_time)
            
            # Обновление интерфейса
            self.window.after(0, self.update_info_display)
            self.show_frame(frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        if self.cap:
            self.cap.release()
    
    def show_frame(self, frame):
        """Отображение кадра в интерфейсе"""
        if frame is None:
            return
        
        frame = cv2.resize(frame, (780, 450))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        
        self.video_label.config(image=img)
        self.video_label.image = img
    
    def show_graphs(self):
        """Показ графиков для всех глаз"""
        if not self.HISTORY_EYES:
            messagebox.showwarning("Предупреждение", "Нет данных для отображения")
            return
        
        num_eyes = len(self.HISTORY_EYES)
        plt.figure(figsize=(15, 10))
        
        plot_idx = 1
        for eye_id, data in self.HISTORY_EYES.items():
            if len(data['x']) < 2:
                continue
                
            color = self.EYE_COLORS[eye_id % len(self.EYE_COLORS)]
            color_rgb = (color[2]/255, color[1]/255, color[0]/255)  # BGR to RGB для matplotlib
            
            # Траектория для каждого глаза
            plt.subplot(2, num_eyes, plot_idx)
            plt.plot(data['x'], data['y'], 'o-', color=color_rgb, alpha=0.5, markersize=2)
            plt.plot(data['fx'], data['fy'], '-', color=color_rgb, linewidth=2, alpha=0.8)
            plt.title(f"Глаз {eye_id} - траектория")
            plt.gca().invert_yaxis()
            plt.grid(True)
            
            # Диаметр и скорость
            plt.subplot(2, num_eyes, plot_idx + num_eyes)
            plt.plot(data['time'], data['d'], '-', color=color_rgb, label='Диаметр')
            plt.plot(data['time'], data['s'], '--', color=color_rgb, label='Скорость')
            plt.title(f"Глаз {eye_id} - параметры")
            plt.xlabel("Секунды")
            plt.grid(True)
            plt.legend()
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    def go_back(self):
        """Возврат в главное меню"""
        if self.is_recording:
            self.stop_recording()
        
        if self.DB_CONNECTION:
            self.DB_CONNECTION.close()
        
        self.window.destroy()
        self.main_window.deiconify()
    
    def on_closing(self):
        """Обработчик закрытия"""
        self.go_back()