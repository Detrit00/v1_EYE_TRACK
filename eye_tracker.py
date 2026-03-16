# eye_tracker.py
import cv2
import time
import math
import mediapipe as mp
import numpy as np
import urllib.request
import os
from utils import LEFT_EYE, RIGHT_EYE, calculate_distance, COLORS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
from collections import deque


class KalmanFilter2D:
    """2D фильтр Калмана для сглаживания координат"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
    def update(self, x, y):
        if not self.initialized:
            self.kf.statePost = np.array([x, y, 0, 0], np.float32)
            self.initialized = True
            return x, y
        
        self.kf.predict()
        measurement = np.array([x, y], np.float32)
        self.kf.correct(measurement)
        estimated = self.kf.statePost
        return int(estimated[0]), int(estimated[1])
    
    def reset(self):
        self.initialized = False
    
    def update_params(self, process_noise, measurement_noise):
        """Обновление параметров фильтра"""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise


class EyeTracker:
    def __init__(self, model_path="face_landmarker.task", 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 min_presence_confidence=0.5,
                 smoothing_factor=0.7,
                 use_kalman=True,
                 kalman_process_noise=0.01,
                 kalman_measurement_noise=0.1,
                 velocity_threshold=100,
                 fixation_threshold=20,
                 max_prediction_time=0.3,
                 min_face_size=100,
                 max_jump_distance=100,
                 median_filter_size=3,
                 use_outlier_filter=True,
                 use_median_filter=True,
                 # Параметры для метода контуров (новый метод)
                 use_contour_method=True,
                 contour_gaussian_blur=21,
                 contour_threshold_offset=6,
                 contour_morph_kernel=3,
                 contour_min_area_ratio=0.01,
                 contour_max_area_ratio=0.3,
                 contour_use_morph=True,
                 pupil_fallback_ratio=0.3):
        """
        Полная версия трекера со всеми настройками
        """
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_presence_confidence = min_presence_confidence
        self.smoothing_factor = smoothing_factor
        self.use_kalman = use_kalman
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.velocity_threshold = velocity_threshold
        self.fixation_threshold = fixation_threshold
        self.max_prediction_time = max_prediction_time
        self.min_face_size = min_face_size
        
        # Параметры фильтрации
        self.max_jump_distance = max_jump_distance
        self.median_filter_size = median_filter_size if median_filter_size % 2 == 1 else 3
        self.use_outlier_filter = use_outlier_filter
        self.use_median_filter = use_median_filter
        
        # Параметры для метода контуров
        self.use_contour_method = use_contour_method
        self.contour_gaussian_blur = contour_gaussian_blur if contour_gaussian_blur % 2 == 1 else 21
        self.contour_threshold_offset = contour_threshold_offset
        self.contour_morph_kernel = contour_morph_kernel
        self.contour_min_area_ratio = contour_min_area_ratio
        self.contour_max_area_ratio = contour_max_area_ratio
        self.contour_use_morph = contour_use_morph
        self.pupil_fallback_ratio = pupil_fallback_ratio
        
        self.landmarker = None
        self.cap = None
        self.is_running = False
        self.tracker_enabled = True
        
        # Данные глаз
        self.left_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0}
        self.right_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0}
        
        # Для сглаживания
        self.smoothed_left = {'x': 0, 'y': 0, 'diameter': 0}
        self.smoothed_right = {'x': 0, 'y': 0, 'diameter': 0}
        
        # Буферы для медианного фильтра
        if self.use_median_filter:
            self.left_x_buffer = deque(maxlen=self.median_filter_size)
            self.left_y_buffer = deque(maxlen=self.median_filter_size)
            self.right_x_buffer = deque(maxlen=self.median_filter_size)
            self.right_y_buffer = deque(maxlen=self.median_filter_size)
            self.left_d_buffer = deque(maxlen=self.median_filter_size)
            self.right_d_buffer = deque(maxlen=self.median_filter_size)
        
        # Фильтры Калмана
        if self.use_kalman:
            self.kalman_left = KalmanFilter2D(kalman_process_noise, kalman_measurement_noise)
            self.kalman_right = KalmanFilter2D(kalman_process_noise, kalman_measurement_noise)
            self.last_detection_time = {'left': 0, 'right': 0}
        
        # Для расчета скорости
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None
        self.start_time = 0
        
        # История для графиков
        self.history = {
            'x': [], 'y': [], 'speed': [], 'diameter': [], 'timestamps': []
        }
        
        # Callback для обработки кадров
        self.frame_callback = None

        self.load_settings_from_file()

    def load_settings_from_file(self):
        """Загрузка настроек из файла tracker_settings.json"""
        settings_file = "tracker_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.update_all_settings(settings)
                    print("Настройки загружены из tracker_settings.json")
            except Exception as e:
                print(f"Ошибка загрузки настроек: {e}")

    def update_all_settings(self, settings):
        """Обновление всех настроек трекера"""
        try:
            # Параметры детекции
            if 'min_detection_confidence' in settings:
                self.min_detection_confidence = settings['min_detection_confidence']
            if 'min_tracking_confidence' in settings:
                self.min_tracking_confidence = settings['min_tracking_confidence']
            if 'min_presence_confidence' in settings:
                self.min_presence_confidence = settings['min_presence_confidence']
            
            # Параметры сглаживания
            if 'smoothing_factor' in settings:
                self.smoothing_factor = settings['smoothing_factor']
            if 'use_kalman' in settings:
                self.use_kalman = settings['use_kalman']
            
            # Параметры фильтра Калмана
            if 'kalman_process_noise' in settings:
                self.kalman_process_noise = settings['kalman_process_noise']
            if 'kalman_measurement_noise' in settings:
                self.kalman_measurement_noise = settings['kalman_measurement_noise']
            
            # Параметры анализа
            if 'velocity_threshold' in settings:
                self.velocity_threshold = settings['velocity_threshold']
            if 'fixation_threshold' in settings:
                self.fixation_threshold = settings['fixation_threshold']
            if 'max_prediction_time' in settings:
                self.max_prediction_time = settings['max_prediction_time']
            if 'min_face_size' in settings:
                self.min_face_size = settings['min_face_size']
            
            # Параметры фильтрации
            if 'max_jump_distance' in settings:
                self.max_jump_distance = settings['max_jump_distance']
            if 'median_filter_size' in settings:
                self.median_filter_size = settings['median_filter_size']
                if self.use_median_filter:
                    self.left_x_buffer = deque(maxlen=self.median_filter_size)
                    self.left_y_buffer = deque(maxlen=self.median_filter_size)
                    self.right_x_buffer = deque(maxlen=self.median_filter_size)
                    self.right_y_buffer = deque(maxlen=self.median_filter_size)
                    self.left_d_buffer = deque(maxlen=self.median_filter_size)
                    self.right_d_buffer = deque(maxlen=self.median_filter_size)
            if 'use_outlier_filter' in settings:
                self.use_outlier_filter = settings['use_outlier_filter']
            if 'use_median_filter' in settings:
                self.use_median_filter = settings['use_median_filter']
                if self.use_median_filter:
                    self.left_x_buffer = deque(maxlen=self.median_filter_size)
                    self.left_y_buffer = deque(maxlen=self.median_filter_size)
                    self.right_x_buffer = deque(maxlen=self.median_filter_size)
                    self.right_y_buffer = deque(maxlen=self.median_filter_size)
                    self.left_d_buffer = deque(maxlen=self.median_filter_size)
                    self.right_d_buffer = deque(maxlen=self.median_filter_size)
            
            # Параметры для метода контуров
            if 'use_contour_method' in settings:
                self.use_contour_method = settings['use_contour_method']
            if 'contour_gaussian_blur' in settings:
                self.contour_gaussian_blur = settings['contour_gaussian_blur']
                if self.contour_gaussian_blur % 2 == 0:
                    self.contour_gaussian_blur += 1
            if 'contour_threshold_offset' in settings:
                self.contour_threshold_offset = settings['contour_threshold_offset']
            if 'contour_morph_kernel' in settings:
                self.contour_morph_kernel = settings['contour_morph_kernel']
            if 'contour_min_area_ratio' in settings:
                self.contour_min_area_ratio = settings['contour_min_area_ratio']
            if 'contour_max_area_ratio' in settings:
                self.contour_max_area_ratio = settings['contour_max_area_ratio']
            if 'contour_use_morph' in settings:
                self.contour_use_morph = settings['contour_use_morph']
            if 'pupil_fallback_ratio' in settings:
                self.pupil_fallback_ratio = settings['pupil_fallback_ratio']
            
            # Обновляем параметры фильтров Калмана
            if self.use_kalman and hasattr(self, 'kalman_left'):
                self.kalman_left.update_params(self.kalman_process_noise, self.kalman_measurement_noise)
                self.kalman_right.update_params(self.kalman_process_noise, self.kalman_measurement_noise)
            
            # Перезапускаем если запущен
            if self.is_running:
                self.restart()
                
            return True
        except Exception as e:
            print(f"Ошибка обновления настроек: {e}")
            return False
        
    def update_settings(self, min_detection_confidence=None, smoothing_factor=None):
        """Для совместимости"""
        if min_detection_confidence is not None:
            self.min_detection_confidence = min_detection_confidence
        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor
        
        if self.is_running:
            self.restart()
    
    def toggle_tracker(self):
        """Включение/выключение трекера"""
        self.tracker_enabled = not self.tracker_enabled
        return self.tracker_enabled
    
    def restart(self):
        """Перезапуск трекера с новыми настройками"""
        was_running = self.is_running
        if was_running:
            self.stop()
        if was_running:
            self.start(self.frame_callback)
    
    def reset_state(self):
        """Сброс состояния трекера (буферы, фильтры) для начала новой сессии"""
        self.history = {'x': [], 'y': [], 'speed': [], 'diameter': [], 'timestamps': []}
        self.prev_x = self.prev_y = self.prev_time = None
        self.left_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0}
        self.right_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0}
        self.smoothed_left = {'x': 0, 'y': 0, 'diameter': 0}
        self.smoothed_right = {'x': 0, 'y': 0, 'diameter': 0}
        
        # Сброс фильтров Калмана
        if self.use_kalman:
            self.kalman_left = KalmanFilter2D(self.kalman_process_noise, self.kalman_measurement_noise)
            self.kalman_right = KalmanFilter2D(self.kalman_process_noise, self.kalman_measurement_noise)
            self.last_detection_time = {'left': 0, 'right': 0}
        
        # Очистка буферов медианного фильтра
        if self.use_median_filter:
            self.left_x_buffer.clear()
            self.left_y_buffer.clear()
            self.right_x_buffer.clear()
            self.right_y_buffer.clear()
            self.left_d_buffer.clear()
            self.right_d_buffer.clear()
    
    def download_model(self, status_callback=None):
        """Скачивает модель если её нет"""
        if os.path.exists(self.model_path):
            return True
        
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        
        if status_callback:
            status_callback("Скачивание модели. Ожидание.")
        
        try:
            urllib.request.urlretrieve(url, self.model_path)
            if status_callback:
                status_callback("Модель загружена")
            return True
        except Exception as e:
            print(f"Ошибка скачивания модели: {e}")
            return False
    
    def start(self, frame_callback=None):
        """Запуск трекинга (режим реального времени)"""
        self.frame_callback = frame_callback
        self.is_running = True
        self.tracker_enabled = True
        
        # Сброс состояния
        self.reset_state()
        
        # Настройка MediaPipe
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            result_callback=self._process_frame
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Запуск камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        return self.cap.isOpened()
    
    def stop(self):
        """Остановка трекинга"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
    
    def _apply_smoothing(self, new_value, smoothed_value):
        """Применяет экспоненциальное сглаживание"""
        if smoothed_value == 0:
            return new_value
        return self.smoothing_factor * smoothed_value + (1 - self.smoothing_factor) * new_value
    
    def _is_outlier(self, new_x, new_y, prev_x, prev_y):
        """Проверка на выброс по максимальному перемещению"""
        if prev_x is None or prev_y is None:
            return False
        distance = math.hypot(new_x - prev_x, new_y - prev_y)
        return distance > self.max_jump_distance
    
    def _apply_median_filter(self, buffer, new_value):
        """Применение медианного фильтра"""
        buffer.append(new_value)
        if len(buffer) == buffer.maxlen:
            return sorted(buffer)[len(buffer)//2]
        return new_value
    
    def _find_pupil_by_contour(self, gray_roi, roi_x_offset, roi_y_offset):
        """
        Находит зрачок методом контуров (бинаризация + поиск контура)
        Возвращает (x, y, diameter) в координатах исходного изображения
        """
        # Сохраняем исходное изображение для отладки (если нужно)
        rows, cols = gray_roi.shape
        
        # 1. Гауссово размытие
        blur_size = (self.contour_gaussian_blur, self.contour_gaussian_blur)
        blurred = cv2.GaussianBlur(gray_roi, blur_size, 0)
        
        # 2. Бинаризация
        min_val = np.min(blurred)
        thresh_val = min_val + self.contour_threshold_offset
        _, threshold = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Морфологические операции (опционально)
        if self.contour_use_morph:
            kernel = np.ones((self.contour_morph_kernel, self.contour_morph_kernel), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        # 4. Поиск контуров
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, threshold  # Возвращаем threshold для отладки
        
        # 5. Сортируем по убыванию площади
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        # 6. Вычисляем ожидаемую площадь зрачка (относительно ROI)
        roi_area = rows * cols
        min_area = roi_area * self.contour_min_area_ratio
        max_area = roi_area * self.contour_max_area_ratio
        
        # 7. Ищем подходящий контур
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Проверяем площадь
            if area < min_area or area > max_area:
                continue
            
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Проверяем, что прямоугольник не слишком вытянутый
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Вычисляем центр и диаметр
            center_x = x + w // 2
            center_y = y + h // 2
            diameter = (w + h) // 2  # Среднее между шириной и высотой
            
            # Переводим в координаты исходного изображения
            abs_x = center_x + roi_x_offset
            abs_y = center_y + roi_y_offset
            
            return abs_x, abs_y, diameter, threshold
        
        return None, None, None, threshold
    
    def _get_eye_data(self, face_landmarks, image, eye_type):
        """
        Получает данные для глаза: центр зрачка и его диаметр (пиксели).
        Использует метод контуров (бинаризация + поиск контура).
        """
        if eye_type == 'left':
            indices = LEFT_EYE
        else:
            indices = RIGHT_EYE

        # Координаты углов глаза (нормированные)
        left_corner = face_landmarks[indices['left']]
        right_corner = face_landmarks[indices['right']]
        
        # Также можно использовать верхнюю и нижнюю точки века, если они есть
        # Для расширения словаря LEFT_EYE и RIGHT_EYE можно добавить ключи 'upper' и 'lower'
        # Но пока используем углы с запасом

        h, w, _ = image.shape

        # Абсолютные координаты углов
        x_left = int(left_corner.x * w)
        y_left = int(left_corner.y * h)
        x_right = int(right_corner.x * w)
        y_right = int(right_corner.y * h)

        # Определяем ROI с запасом
        margin = 30  # Увеличим отступ для метода контуров
        x_min = max(0, min(x_left, x_right) - margin)
        x_max = min(w, max(x_left, x_right) + margin)
        y_min = max(0, min(y_left, y_right) - margin)
        y_max = min(h, max(y_left, y_right) + margin)

        # Если ROI слишком мал, возвращаем fallback
        if x_max - x_min < 20 or y_max - y_min < 20:
            center_x = (x_left + x_right) // 2
            center_y = (y_left + y_right) // 2
            eye_width = math.hypot(x_right - x_left, y_right - y_left)
            diameter = eye_width * self.pupil_fallback_ratio
            return center_x, center_y, diameter

        # Вырезаем ROI
        roi = image[y_min:y_max, x_min:x_max]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Если включён метод контуров
        if self.use_contour_method:
            pupil_x, pupil_y, diameter, threshold = self._find_pupil_by_contour(
                gray_roi, x_min, y_min
            )
            
            # Для отладки можно сохранять threshold (но в реальном времени не рекомендуется)
            # Если нужно визуализировать threshold, можно добавить отдельный режим
            
            if pupil_x is not None:
                return pupil_x, pupil_y, diameter
            else:
                # Fallback если зрачок не найден
                center_x = (x_left + x_right) // 2
                center_y = (y_left + y_right) // 2
                eye_width = math.hypot(x_right - x_left, y_right - y_left)
                diameter = eye_width * self.pupil_fallback_ratio
                return center_x, center_y, diameter
        else:
            # Если метод контуров отключён, возвращаем старые данные (центр глаза и его ширину)
            center_x = (x_left + x_right) // 2
            center_y = (y_left + y_right) // 2
            eye_width = math.hypot(x_right - x_left, y_right - y_left)
            diameter = eye_width
            return center_x, center_y, diameter
    
    def _calculate_speed(self, x, y, current_time):
        """Расчёт скорости движения"""
        if self.prev_x is not None and self.prev_time is not None:
            dist = math.hypot(x - self.prev_x, y - self.prev_y)
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                return dist / time_diff
        return 0.0
    
    def filter_measurements(self, left_x, left_y, left_d, right_x, right_y, right_d, current_time):
        """
        Применяет все включённые фильтры к сырым измерениям глаз.
        Обновляет состояние трекера (self.left_eye, self.right_eye, буферы, скорость).
        """
        # 1. Фильтр резких скачков (выбросы)
        if self.use_outlier_filter and self.prev_x is not None:
            if self._is_outlier(left_x, left_y, self.prev_x, self.prev_y):
                if self.use_kalman:
                    left_x, left_y = self.kalman_left.update(0, 0)
                    right_x, right_y = self.kalman_right.update(0, 0)
                else:
                    left_x, left_y = self.prev_x, self.prev_y
                    right_x, right_y = self.prev_x, self.prev_y

        # 2. Медианный фильтр (дрожание)
        if self.use_median_filter:
            left_x = self._apply_median_filter(self.left_x_buffer, left_x)
            left_y = self._apply_median_filter(self.left_y_buffer, left_y)
            right_x = self._apply_median_filter(self.right_x_buffer, right_x)
            right_y = self._apply_median_filter(self.right_y_buffer, right_y)
            left_d = self._apply_median_filter(self.left_d_buffer, left_d)
            right_d = self._apply_median_filter(self.right_d_buffer, right_d)

        # 3. Фильтр Калмана
        if self.use_kalman:
            left_x, left_y = self.kalman_left.update(left_x, left_y)
            right_x, right_y = self.kalman_right.update(right_x, right_y)

        # 4. Экспоненциальное сглаживание
        self.smoothed_left['x'] = self._apply_smoothing(left_x, self.smoothed_left['x'])
        self.smoothed_left['y'] = self._apply_smoothing(left_y, self.smoothed_left['y'])
        self.smoothed_left['diameter'] = self._apply_smoothing(left_d, self.smoothed_left['diameter'])

        self.smoothed_right['x'] = self._apply_smoothing(right_x, self.smoothed_right['x'])
        self.smoothed_right['y'] = self._apply_smoothing(right_y, self.smoothed_right['y'])
        self.smoothed_right['diameter'] = self._apply_smoothing(right_d, self.smoothed_right['diameter'])

        # Итоговые значения после всех фильтров
        left_x = int(self.smoothed_left['x'])
        left_y = int(self.smoothed_left['y'])
        left_d = self.smoothed_left['diameter']
        right_x = int(self.smoothed_right['x'])
        right_y = int(self.smoothed_right['y'])
        right_d = self.smoothed_right['diameter']

        # 5. Расчёт скорости
        speed = self._calculate_speed(left_x, left_y, current_time)
        self.prev_x, self.prev_y, self.prev_time = left_x, left_y, current_time

        # 6. Обновление данных глаз
        self.left_eye.update({
            'x': left_x,
            'y': left_y,
            'diameter': left_d,
            'speed': speed
        })
        self.right_eye.update({
            'x': right_x,
            'y': right_y,
            'diameter': right_d,
            'speed': speed
        })

        # 7. Сохранение в историю
        rel_time = current_time - self.start_time if self.start_time else current_time
        self.history['x'].append(left_x)
        self.history['y'].append(left_y)
        self.history['diameter'].append(left_d)
        self.history['speed'].append(speed)
        self.history['timestamps'].append(rel_time)
    
    def _process_frame(self, result, output_image, timestamp_ms):
        """Обработка кадра (callback для LIVE_STREAM)"""
        current_time = timestamp_ms / 1000.0

        if not result.face_landmarks or not self.tracker_enabled:
            # При потере трекинга используем предсказание Калмана
            if self.use_kalman and hasattr(self, 'last_detection_time'):
                time_since_detection = current_time - max(
                    self.last_detection_time.get('left', 0),
                    self.last_detection_time.get('right', 0)
                )

                if time_since_detection < self.max_prediction_time:
                    left_x, left_y = self.kalman_left.update(0, 0)
                    right_x, right_y = self.kalman_right.update(0, 0)

                    self.left_eye['x'] = left_x
                    self.left_eye['y'] = left_y
                    self.right_eye['x'] = right_x
                    self.right_eye['y'] = right_y
            return

        face_landmarks = result.face_landmarks[0]
        # Получаем изображение в формате numpy (RGB)
        frame_rgb = output_image.numpy_view()

        # Получаем данные для глаз
        left_x, left_y, left_d = self._get_eye_data(face_landmarks, frame_rgb, 'left')
        right_x, right_y, right_d = self._get_eye_data(face_landmarks, frame_rgb, 'right')

        # Обновляем время последней успешной детекции для Калмана
        if self.use_kalman:
            self.last_detection_time['left'] = current_time
            self.last_detection_time['right'] = current_time

        # Применяем все фильтры
        self.filter_measurements(left_x, left_y, left_d,
                                 right_x, right_y, right_d,
                                 current_time)
    
    def draw_eyes(self, frame):
        """Отрисовка глаз на кадре"""
        if self.tracker_enabled:
            # Левый глаз
            if self.left_eye['x'] != 0 and self.left_eye['y'] != 0:
                color = COLORS['left_eye']
                cv2.circle(frame, (self.left_eye['x'], self.left_eye['y']), 3, color, -1)
                radius = int(self.left_eye['diameter'] / 2)
                if radius > 0:
                    cv2.circle(frame, (self.left_eye['x'], self.left_eye['y']), radius, color, 2)
                cv2.putText(frame, "L", (self.left_eye['x'] - 20, self.left_eye['y'] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(frame, "L: not found", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Правый глаз
            if self.right_eye['x'] != 0 and self.right_eye['y'] != 0:
                color = COLORS['right_eye']
                cv2.circle(frame, (self.right_eye['x'], self.right_eye['y']), 3, color, -1)
                radius = int(self.right_eye['diameter'] / 2)
                if radius > 0:
                    cv2.circle(frame, (self.right_eye['x'], self.right_eye['y']), radius, color, 2)
                cv2.putText(frame, "R", (self.right_eye['x'] + 10, self.right_eye['y'] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(frame, "R: not found", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Статус трекера
        status = "TRACKING ON" if self.tracker_enabled else "TRACKING OFF"
        color = (0, 255, 0) if self.tracker_enabled else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
    
    def process_frame(self, frame):
        """Обработка одного кадра для реального времени"""
        if not self.is_running or not self.cap:
            return frame

        if self.tracker_enabled:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms = int(time.time() * 1000)
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)

        frame = self.draw_eyes(frame)

        return frame