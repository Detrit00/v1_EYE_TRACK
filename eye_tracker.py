# eye_tracker.py
import cv2
import time
import math
import mediapipe as mp
import numpy as np
import urllib.request
import os
from utils import LEFT_EYE, RIGHT_EYE, COLORS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
from collections import deque
from typing import Optional, Tuple, List


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
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise


class StarburstEllipseFitter:
    """
    Реализация алгоритма Starburst для поиска граничных точек зрачка
    и последующей аппроксимации эллипсом.
    """

    def __init__(self, num_rays=12, ray_length=30, gradient_threshold=30,
                 min_ellipse_points=6, use_gradient_direction=True):
        """
        :param num_rays: количество лучей, испускаемых из предполагаемого центра
        :param ray_length: максимальная длина луча (пикселей)
        :param gradient_threshold: минимальный модуль градиента для фиксации границы
        :param min_ellipse_points: минимальное количество точек для подгонки эллипса
        :param use_gradient_direction: учитывать ли направление градиента (от тёмного к светлому)
        """
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.gradient_threshold = gradient_threshold
        self.min_ellipse_points = min_ellipse_points
        self.use_gradient_direction = use_gradient_direction

    def find_pupil(self, gray_roi: np.ndarray, roi_offset_x: int, roi_offset_y: int,
                   initial_center: Optional[Tuple[int, int]] = None) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """
        Основной метод: ищет зрачок в ROI и возвращает (x, y, диаметр) в координатах исходного изображения.

        :param gray_roi: одноканальное изображение ROI глаза (grayscale)
        :param roi_offset_x: смещение ROI по x в исходном кадре
        :param roi_offset_y: смещение ROI по y в исходном кадре
        :param initial_center: начальное предположение центра зрачка (в координатах ROI)
        :return: (x_abs, y_abs, diameter) или (None, None, None) если не удалось
        """
        if gray_roi.size == 0:
            return None, None, None

        h, w = gray_roi.shape

        # 1. Если нет начального центра, используем центр ROI
        if initial_center is None:
            init_cx = w // 2
            init_cy = h // 2
        else:
            init_cx, init_cy = initial_center
            # Проверяем, что центр внутри ROI
            init_cx = max(0, min(init_cx, w - 1))
            init_cy = max(0, min(init_cy, h - 1))

        # 2. Вычисляем градиенты (Sobel)
        grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = np.arctan2(grad_y, grad_x)  # угол градиента

        # 3. Испускаем лучи и собираем граничные точки
        edge_points = []
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        for angle in angles:
            # Направление луча от центра
            dx = math.cos(angle)
            dy = math.sin(angle)

            for step in range(1, self.ray_length + 1):
                px = int(init_cx + dx * step)
                py = int(init_cy + dy * step)

                if px < 0 or px >= w or py < 0 or py >= h:
                    break

                mag = magnitude[py, px]

                if mag >= self.gradient_threshold:
                    # Проверяем направление градиента (зрачок тёмный, радужка светлая)
                    if self.use_gradient_direction:
                        grad_angle = direction[py, px]
                        # Вектор от центра к точке
                        radial_angle = np.arctan2(py - init_cy, px - init_cx)
                        angle_diff = np.abs(grad_angle - radial_angle)
                        # Нормализуем в [0, pi]
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                        # Для тёмного зрачка градиент должен быть направлен наружу (противоположно радиальному)
                        # Упрощённо: разрешаем только градиент, направленный от центра
                        if angle_diff > np.pi / 2:
                            continue
                    edge_points.append((px, py))
                    break  # нашли границу по этому лучу

        # 4. Если точек слишком мало, возвращаем None
        if len(edge_points) < self.min_ellipse_points:
            return None, None, None

        # 5. Подгоняем эллипс
        pts = np.array(edge_points, dtype=np.int32)
        ellipse = cv2.fitEllipse(pts)
        (center_x_roi, center_y_roi), (axes_w, axes_h), angle = ellipse

        # 6. Диаметр – среднее между осями (или можно использовать большую ось)
        diameter = (axes_w + axes_h) / 2.0

        # Переводим в абсолютные координаты
        abs_x = int(center_x_roi + roi_offset_x)
        abs_y = int(center_y_roi + roi_offset_y)

        return abs_x, abs_y, diameter


class EyeTracker:
    def __init__(self, model_path="face_landmarker.task",
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 min_presence_confidence=0.5,
                 smoothing_factor=0.7,
                 use_kalman=True,
                 kalman_process_noise=0.05, # было 0.01 чуть больше, чтобы быстрее реагировать на движение
                 kalman_measurement_noise=0.2, # 0.1 больше, так как измерения шумнее
                 # I-VT параметры
                 velocity_threshold=100,      # порог скорости (пикс/сек)
                 min_fixation_duration=0.04,  # мин. длительность фиксации (сек)
                 min_saccade_duration=0.01,   # мин. длительность саккады (сек)
                 # Параметры фильтрации
                 max_jump_distance=100,
                 median_filter_size=3,
                 use_outlier_filter=True,
                 use_median_filter=True,
                 # Параметры Starburst
                 starburst_num_rays=12,
                 starburst_ray_length=30,
                 starburst_gradient_threshold=15, # 30 ниже для RGB
                 starburst_min_ellipse_points=6,
                 use_gradient_direction=True):
        """
        Инициализация трекера с выбранными методами.
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
        self.min_fixation_duration = min_fixation_duration
        self.min_saccade_duration = min_saccade_duration
        self.max_jump_distance = max_jump_distance
        self.median_filter_size = median_filter_size if median_filter_size % 2 == 1 else 3
        self.use_outlier_filter = use_outlier_filter
        self.use_median_filter = use_median_filter

        # Параметры Starburst
        self.starburst_num_rays = starburst_num_rays
        self.starburst_ray_length = starburst_ray_length
        self.starburst_gradient_threshold = starburst_gradient_threshold
        self.starburst_min_ellipse_points = starburst_min_ellipse_points
        self.use_gradient_direction = use_gradient_direction

        self.landmarker = None
        self.cap = None
        self.is_running = False
        self.tracker_enabled = True

        # Данные глаз (после всех фильтров и классификации)
        self.left_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0, 'movement': 'unknown'}
        self.right_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0, 'movement': 'unknown'}

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

        # Для расчёта скорости
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None
        self.start_time = 0

        # История для графиков (добавим тип движения)
        self.history = {
            'x': [], 'y': [], 'speed': [], 'diameter': [], 'timestamps': [], 'movement_type': []
        }

        # I-VT состояние
        self.current_movement = 'fixation'   # 'fixation' или 'saccade'
        self.movement_start_time = 0
        self.movement_end_time = 0

        # Callback
        self.frame_callback = None

        # Инициализируем Starburst
        self.starburst = StarburstEllipseFitter(
            num_rays=self.starburst_num_rays,
            ray_length=self.starburst_ray_length,
            gradient_threshold=self.starburst_gradient_threshold,
            min_ellipse_points=self.starburst_min_ellipse_points,
            use_gradient_direction=self.use_gradient_direction
        )

        self.load_settings_from_file()

    # --- Методы загрузки/сохранения настроек (оставлены для совместимости) ---
    def load_settings_from_file(self):
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
            if 'min_detection_confidence' in settings:
                self.min_detection_confidence = settings['min_detection_confidence']
            if 'min_tracking_confidence' in settings:
                self.min_tracking_confidence = settings['min_tracking_confidence']
            if 'min_presence_confidence' in settings:
                self.min_presence_confidence = settings['min_presence_confidence']
            if 'smoothing_factor' in settings:
                self.smoothing_factor = settings['smoothing_factor']
            if 'use_kalman' in settings:
                self.use_kalman = settings['use_kalman']
            if 'kalman_process_noise' in settings:
                self.kalman_process_noise = settings['kalman_process_noise']
            if 'kalman_measurement_noise' in settings:
                self.kalman_measurement_noise = settings['kalman_measurement_noise']
            if 'velocity_threshold' in settings:
                self.velocity_threshold = settings['velocity_threshold']
            if 'min_fixation_duration' in settings:
                self.min_fixation_duration = settings['min_fixation_duration']
            if 'min_saccade_duration' in settings:
                self.min_saccade_duration = settings['min_saccade_duration']
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
            # Параметры Starburst
            if 'starburst_num_rays' in settings:
                self.starburst_num_rays = settings['starburst_num_rays']
            if 'starburst_ray_length' in settings:
                self.starburst_ray_length = settings['starburst_ray_length']
            if 'starburst_gradient_threshold' in settings:
                self.starburst_gradient_threshold = settings['starburst_gradient_threshold']
            if 'starburst_min_ellipse_points' in settings:
                self.starburst_min_ellipse_points = settings['starburst_min_ellipse_points']
            if 'use_gradient_direction' in settings:
                self.use_gradient_direction = settings['use_gradient_direction']

            # Пересоздаём Starburst с новыми параметрами
            self.starburst = StarburstEllipseFitter(
                num_rays=self.starburst_num_rays,
                ray_length=self.starburst_ray_length,
                gradient_threshold=self.starburst_gradient_threshold,
                min_ellipse_points=self.starburst_min_ellipse_points,
                use_gradient_direction=self.use_gradient_direction
            )

            if self.use_kalman and hasattr(self, 'kalman_left'):
                self.kalman_left.update_params(self.kalman_process_noise, self.kalman_measurement_noise)
                self.kalman_right.update_params(self.kalman_process_noise, self.kalman_measurement_noise)

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
        self.tracker_enabled = not self.tracker_enabled
        return self.tracker_enabled

    def restart(self):
        was_running = self.is_running
        if was_running:
            self.stop()
        if was_running:
            self.start(self.frame_callback)

    def reset_state(self):
        """Сброс состояния трекера"""
        self.history = {'x': [], 'y': [], 'speed': [], 'diameter': [], 'timestamps': [], 'movement_type': []}
        self.prev_x = self.prev_y = self.prev_time = None
        self.left_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0, 'movement': 'unknown'}
        self.right_eye = {'x': 0, 'y': 0, 'diameter': 0, 'speed': 0, 'movement': 'unknown'}
        self.smoothed_left = {'x': 0, 'y': 0, 'diameter': 0}
        self.smoothed_right = {'x': 0, 'y': 0, 'diameter': 0}
        self.current_movement = 'fixation'
        self.movement_start_time = 0
        self.movement_end_time = 0

        if self.use_kalman:
            self.kalman_left = KalmanFilter2D(self.kalman_process_noise, self.kalman_measurement_noise)
            self.kalman_right = KalmanFilter2D(self.kalman_process_noise, self.kalman_measurement_noise)
            self.last_detection_time = {'left': 0, 'right': 0}

        if self.use_median_filter:
            self.left_x_buffer.clear()
            self.left_y_buffer.clear()
            self.right_x_buffer.clear()
            self.right_y_buffer.clear()
            self.left_d_buffer.clear()
            self.right_d_buffer.clear()

    def download_model(self, status_callback=None):
        if os.path.exists(self.model_path):
            return True
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        if status_callback:
            status_callback("Скачивание модели...")
        try:
            urllib.request.urlretrieve(url, self.model_path)
            if status_callback:
                status_callback("Модель загружена")
            return True
        except Exception as e:
            print(f"Ошибка скачивания модели: {e}")
            return False

    def start(self, frame_callback=None):
        self.frame_callback = frame_callback
        self.is_running = True
        self.tracker_enabled = True
        self.reset_state()

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

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.start_time = time.time()
        self.last_frame_time = time.time()
        return self.cap.isOpened()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    # --- Вспомогательные функции фильтрации ---
    def _apply_smoothing(self, new_value, smoothed_value):
        if smoothed_value == 0:
            return new_value
        return self.smoothing_factor * smoothed_value + (1 - self.smoothing_factor) * new_value

    def _is_outlier(self, new_x, new_y, prev_x, prev_y):
        if prev_x is None or prev_y is None:
            return False
        distance = math.hypot(new_x - prev_x, new_y - prev_y)
        return distance > self.max_jump_distance

    def _apply_median_filter(self, buffer, new_value):
        buffer.append(new_value)
        if len(buffer) == buffer.maxlen:
            return sorted(buffer)[len(buffer) // 2]
        return new_value

    # --- I-VT классификация ---
    def _classify_movement(self, speed, current_time):
        """
        Классифицирует движение по скорости (I-VT).
        Обновляет текущее состояние self.current_movement и отслеживает длительности.
        Возвращает строку 'fixation' или 'saccade'.
        """
        if speed > self.velocity_threshold:
            # Если скорость превышает порог, это саккада
            if self.current_movement != 'saccade':
                # Переход из фиксации в саккаду
                if self.current_movement == 'fixation':
                    # Проверяем, что фиксация была достаточно долгой
                    if self.movement_end_time - self.movement_start_time >= self.min_fixation_duration:
                        # сохраняем фиксацию как валидную (можно добавить в историю событий)
                        pass
                self.current_movement = 'saccade'
                self.movement_start_time = current_time
        else:
            # Скорость ниже порога – фиксация
            if self.current_movement != 'fixation':
                # Переход из саккады в фиксацию
                if self.current_movement == 'saccade':
                    if current_time - self.movement_start_time >= self.min_saccade_duration:
                        # саккада валидна
                        pass
                self.current_movement = 'fixation'
                self.movement_start_time = current_time
        self.movement_end_time = current_time
        return self.current_movement

    # --- Основная логика обработки глаза ---
    def _get_eye_data(self, face_landmarks, image, eye_type):
        """
        Получает данные для глаза с использованием Starburst + эллиптической аппроксимации.
        """
        if eye_type == 'left':
            indices = LEFT_EYE
        else:
            indices = RIGHT_EYE

        left_corner = face_landmarks[indices['left']]
        right_corner = face_landmarks[indices['right']]

        h, w, _ = image.shape
        x_left = int(left_corner.x * w)
        y_left = int(left_corner.y * h)
        x_right = int(right_corner.x * w)
        y_right = int(right_corner.y * h)

        # ROI с запасом
        margin = 40
        x_min = max(0, min(x_left, x_right) - margin)
        x_max = min(w, max(x_left, x_right) + margin)
        y_min = max(0, min(y_left, y_right) - margin)
        y_max = min(h, max(y_left, y_right) + margin)

        if x_max - x_min < 20 or y_max - y_min < 20:
            # Слишком маленькая область – возвращаем fallback
            center_x = (x_left + x_right) // 2
            center_y = (y_left + y_right) // 2
            eye_width = math.hypot(x_right - x_left, y_right - y_left)
            diameter = eye_width * 0.3   # примерный диаметр зрачка
            return center_x, center_y, diameter

        roi = image[y_min:y_max, x_min:x_max]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Начальная оценка центра зрачка – середина между углами глаза (в ROI)
        init_center_roi_x = ((x_left + x_right) // 2) - x_min
        init_center_roi_y = ((y_left + y_right) // 2) - y_min
        init_center = (init_center_roi_x, init_center_roi_y)

        # Запускаем Starburst
        abs_x, abs_y, diameter = self.starburst.find_pupil(
            gray_roi, x_min, y_min, initial_center=init_center
        )

        if abs_x is None:
            # Fallback: центр между углами глаза
            center_x = (x_left + x_right) // 2
            center_y = (y_left + y_right) // 2
            eye_width = math.hypot(x_right - x_left, y_right - y_left)
            diameter = eye_width * 0.3
            return center_x, center_y, diameter

        return abs_x, abs_y, diameter

    def _calculate_speed(self, x, y, current_time):
        if self.prev_x is not None and self.prev_time is not None:
            dist = math.hypot(x - self.prev_x, y - self.prev_y)
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                return dist / time_diff
        return 0.0

    def filter_measurements(self, left_x, left_y, left_d, right_x, right_y, right_d, current_time):
        """
        Применяет фильтры и I-VT классификацию.
        """
        # 1. Фильтр резких скачков
        if self.use_outlier_filter and self.prev_x is not None:
            if self._is_outlier(left_x, left_y, self.prev_x, self.prev_y):
                if self.use_kalman:
                    left_x, left_y = self.kalman_left.update(0, 0)
                    right_x, right_y = self.kalman_right.update(0, 0)
                else:
                    left_x, left_y = self.prev_x, self.prev_y
                    right_x, right_y = self.prev_x, self.prev_y

        # 2. Медианный фильтр
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

        left_x = int(self.smoothed_left['x'])
        left_y = int(self.smoothed_left['y'])
        left_d = self.smoothed_left['diameter']
        right_x = int(self.smoothed_right['x'])
        right_y = int(self.smoothed_right['y'])
        right_d = self.smoothed_right['diameter']

        # 5. Расчёт скорости
        speed = self._calculate_speed(left_x, left_y, current_time)
        self.prev_x, self.prev_y, self.prev_time = left_x, left_y, current_time

        # 6. I-VT классификация
        movement_type = self._classify_movement(speed, current_time)
        movement_code = 0 if movement_type == 'fixation' else 1

        # 7. Обновление данных глаз
        self.left_eye.update({
            'x': left_x,
            'y': left_y,
            'diameter': left_d,
            'speed': speed,
            'movement': movement_type
        })
        self.right_eye.update({
            'x': right_x,
            'y': right_y,
            'diameter': right_d,
            'speed': speed,
            'movement': movement_type
        })

        # 8. Сохранение в историю
        rel_time = current_time - self.start_time if self.start_time else current_time
        self.history['x'].append(left_x)
        self.history['y'].append(left_y)
        self.history['diameter'].append(left_d)
        self.history['speed'].append(speed)
        self.history['timestamps'].append(rel_time)
        self.history['movement_type'].append(movement_code)

    def _process_frame(self, result, output_image, timestamp_ms):
        """Callback для обработки кадра"""
        current_time = timestamp_ms / 1000.0

        if not result.face_landmarks or not self.tracker_enabled:
            if self.use_kalman and hasattr(self, 'last_detection_time'):
                time_since_detection = current_time - max(
                    self.last_detection_time.get('left', 0),
                    self.last_detection_time.get('right', 0)
                )
                if time_since_detection < 0.3:  # макс. время предсказания
                    left_x, left_y = self.kalman_left.update(0, 0)
                    right_x, right_y = self.kalman_right.update(0, 0)
                    self.left_eye['x'] = left_x
                    self.left_eye['y'] = left_y
                    self.right_eye['x'] = right_x
                    self.right_eye['y'] = right_y
            return

        face_landmarks = result.face_landmarks[0]
        frame_rgb = output_image.numpy_view()

        # Получаем данные для глаз (Starburst)
        left_x, left_y, left_d = self._get_eye_data(face_landmarks, frame_rgb, 'left')
        right_x, right_y, right_d = self._get_eye_data(face_landmarks, frame_rgb, 'right')

        if self.use_kalman:
            self.last_detection_time['left'] = current_time
            self.last_detection_time['right'] = current_time

        # Применяем фильтры и классификацию
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

            # Отображаем текущее движение и скорость
            if self.left_eye['speed'] > 0:
                cv2.putText(frame, f"Speed: {self.left_eye['speed']:.1f} px/s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Movement: {self.left_eye['movement']}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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