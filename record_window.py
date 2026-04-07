# # record_window.py
# import tkinter as tk
# from tkinter import messagebox
# import cv2
# import time
# import numpy as np
# import threading
# import matplotlib.pyplot as plt
# from utils import center_window
# from database import EyeDatabase
# from eye_tracker import EyeTracker
# import os
# from datetime import datetime

# class RecordWindow:
#     def __init__(self, window, main_window):
#         self.window = window
#         self.main_window = main_window
#         self.window.title("Запись видео - Айтрекинг")
        
#         # Центрируем окно
#         center_window(self.window, 800, 600)
        
#         # Компоненты
#         self.db = EyeDatabase()
#         self.tracker = EyeTracker()
#         self.is_recording = False
#         self.is_video_recording = False
#         self.recording_thread = None
#         self.video_writer_raw = None  # Для чистого видео (без обработки)
#         self.video_filename_raw = None
#         self.video_start_time = 0
        
#         # Создаем интерфейс
#         self.setup_ui()
        
#         # Скачиваем модель
#         self.download_model()
        
#         # Обработчик закрытия окна
#         self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
#     def setup_ui(self):
#         """Создание интерфейса"""
#         # Верхняя панель с кнопками
#         control_frame = tk.Frame(self.window, bg="#f0f0f0", height=60)
#         control_frame.pack(fill=tk.X, padx=10, pady=10)
        
#         # Кнопка записи данных (трекер)
#         self.record_btn = tk.Button(
#             control_frame,
#             text="Начать запись данных",
#             font=("Arial", 11, "bold"),
#             width=20,
#             height=1,
#             bg="#4CAF50",
#             fg="white",
#             command=self.toggle_recording
#         )
#         self.record_btn.pack(side=tk.LEFT, padx=5)
        
#         # Кнопка записи видео (чистое видео)
#         self.video_record_btn = tk.Button(
#             control_frame,
#             text="Запись видео",
#             font=("Arial", 11, "bold"),
#             width=15,
#             height=1,
#             bg="#FF5722",
#             fg="white",
#             command=self.toggle_video_recording
#         )
#         self.video_record_btn.pack(side=tk.LEFT, padx=5)
        
#         # Кнопка сохранения графиков
#         self.save_btn = tk.Button(
#             control_frame,
#             text="Показать графики",
#             font=("Arial", 11),
#             width=15,
#             height=1,
#             bg="#2196F3",
#             fg="white",
#             command=self.show_graphs,
#             state=tk.DISABLED
#         )
#         self.save_btn.pack(side=tk.LEFT, padx=5)
        
#         # Кнопка "Назад"
#         back_btn = tk.Button(
#             control_frame,
#             text="Назад в меню",
#             font=("Arial", 11),
#             width=15,
#             height=1,
#             bg="#FF9800",
#             fg="white",
#             command=self.go_back
#         )
#         back_btn.pack(side=tk.RIGHT, padx=5)
        
#         # Основная область для видео
#         self.video_frame = tk.Frame(
#             self.window,
#             bg="black",
#             width=780,
#             height=450
#         )
#         self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
#         self.video_frame.pack_propagate(False)
        
#         # Label для отображения видео
#         self.video_label = tk.Label(self.video_frame, bg="black")
#         self.video_label.pack(fill=tk.BOTH, expand=True)
        
#         # Информационная панель
#         info_frame = tk.Frame(self.window, bg="#f0f0f0", height=50)
#         info_frame.pack(fill=tk.X, padx=10, pady=5)
        
#         self.info_label = tk.Label(
#             info_frame,
#             text="Левый глаз: X: 0, Y: 0 | Диаметр: 0 px | Скорость: 0 px/s\n"
#                  "Правый глаз: X: 0, Y: 0 | Диаметр: 0 px | Скорость: 0 px/s",
#             bg="#f0f0f0",
#             font=("Arial", 10),
#             justify=tk.LEFT
#         )
#         self.info_label.pack()
    
#     def download_model(self):
#         """Скачивание модели"""
#         def update_status(msg):
#             # Просто обновляем окно без статуса
#             self.window.update()
        
#         if not self.tracker.download_model(update_status):
#             messagebox.showerror("Ошибка", "Не удалось скачать модель")
    
#     def toggle_recording(self):
#         """Запуск/остановка записи данных"""
#         if not self.is_recording:
#             self.start_recording()
#         else:
#             self.stop_recording()
    
#     def toggle_video_recording(self):
#         """Запуск/остановка записи чистого видео (без обработки)"""
#         if not self.is_video_recording:
#             self.start_video_recording()
#         else:
#             self.stop_video_recording()
    
#     def start_video_recording(self):
#         """Запуск записи чистого видео (без обработки)"""
#         if not self.tracker.cap or not self.tracker.cap.isOpened():
#             messagebox.showerror("Ошибка", "Камера не запущена. Сначала начните запись данных.")
#             return
        
#         # Создаем папку для видео если её нет
#         if not os.path.exists("videos"):
#             os.makedirs("videos")
        
#         # Генерируем имя файла с временной меткой
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.video_filename_raw = f"videos/recording_{timestamp}_raw.avi"
        
#         # Настройки кодека
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         fps = 30.0
#         frame_size = (780, 450)
        
#         self.video_writer_raw = cv2.VideoWriter(self.video_filename_raw, fourcc, fps, frame_size)
        
#         if not self.video_writer_raw.isOpened():
#             messagebox.showerror("Ошибка", "Не удалось создать файл для записи видео")
#             return
        
#         self.is_video_recording = True
#         self.video_start_time = time.time()
        
#         self.video_record_btn.config(
#             text="Остановить видео",
#             bg="#f44336"
#         )
        
#         messagebox.showinfo("Информация", f"Запись чистого видео начата\nФайл: {self.video_filename_raw}")
    
#     def stop_video_recording(self):
#         """Остановка записи видео"""
#         self.is_video_recording = False
        
#         if self.video_writer_raw:
#             self.video_writer_raw.release()
#             self.video_writer_raw = None
            
#             # Показываем сообщение о сохранении
#             if os.path.exists(self.video_filename_raw):
#                 file_size = os.path.getsize(self.video_filename_raw) / (1024*1024)
#                 messagebox.showinfo("Информация", 
#                                   f"Чистое видео сохранено\nФайл: {self.video_filename_raw}\n"
#                                   f"Размер: {file_size:.2f} МБ")
        
#         self.video_record_btn.config(
#             text="Запись видео",
#             bg="#FF5722"
#         )
    
#     def start_recording(self):
#         """Запуск записи данных"""
#         if not self.tracker.start():
#             messagebox.showerror("Ошибка", "Не удалось открыть камеру")
#             return
        
#         self.is_recording = True
#         self.record_btn.config(
#             text="Остановить запись данных",
#             bg="#f44336"
#         )
#         self.save_btn.config(state=tk.NORMAL)
        
#         # Запускаем поток с записью
#         self.recording_thread = threading.Thread(target=self.recording_loop)
#         self.recording_thread.daemon = True
#         self.recording_thread.start()
    
#     def stop_recording(self):
#         """Остановка записи данных"""
#         self.is_recording = False
        
#         # Останавливаем запись видео если она активна
#         if self.is_video_recording:
#             self.stop_video_recording()
        
#         self.tracker.stop()
        
#         self.record_btn.config(
#             text="Начать запись данных",
#             bg="#4CAF50"
#         )
        
#         # Показываем черный экран
#         black_img = np.zeros((450, 780, 3), dtype=np.uint8)
#         self.show_frame(black_img)
        
#         # Показываем статистику
#         data_count = len(self.tracker.history['x'])
#         if data_count > 0 and len(self.tracker.history['timestamps']) > 0:
#             duration = self.tracker.history['timestamps'][-1]
#             messagebox.showinfo("Информация", 
#                               f"Запись завершена\nСобрано точек данных: {data_count}\n"
#                               f"Длительность: {duration:.2f} сек")
    
#     def recording_loop(self):
#         """Основной цикл записи"""
#         while self.is_recording and self.tracker.cap and self.tracker.cap.isOpened():
#             success, frame = self.tracker.cap.read()
#             if not success:
#                 continue
            
#             # Зеркальное отображение
#             frame = cv2.flip(frame, 1)
            
#             # Сохраняем копию чистого кадра (без обработки) для записи в видео
#             raw_frame = frame.copy()
            
#             # Обработка кадра трекером (для отображения на экране и сбора данных)
#             tracked_frame = self.tracker.process_frame(frame)
            
#             # Сохранение данных в БД
#             if self.tracker.left_eye['x'] != 0:
#                 self.db.save_eye_data(
#                     time.time() - self.tracker.start_time,
#                     self.tracker.left_eye,
#                     self.tracker.right_eye
#                 )
            
#             # Подготовка кадра для отображения на экране (с трекером)
#             display_frame = cv2.resize(tracked_frame, (780, 450))
            
#             # Если идет запись видео, сохраняем чистый кадр (без трекера)
#             if self.is_video_recording and self.video_writer_raw:
#                 # Подготавливаем чистый кадр для сохранения
#                 raw_display = cv2.resize(raw_frame, (780, 450))
                
#                 # Добавляем индикатор записи (красная точка) на чистое видео
#                 cv2.circle(raw_display, (50, 50), 15, (0, 0, 255), -1)
                
#                 # Сохраняем чистый кадр в видео
#                 self.video_writer_raw.write(raw_display)
            
#             # Обновление интерфейса (показываем кадр с трекером)
#             self.window.after(0, self.update_info)
#             self.show_frame(display_frame)
        
#         # Очистка
#         self.tracker.stop()
#         if self.is_video_recording:
#             self.stop_video_recording()
    
#     def show_frame(self, frame):
#         """Отображает кадр в интерфейсе"""
#         if frame is None:
#             return
        
#         # Конвертируем в формат для tkinter
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        
#         self.video_label.config(image=img)
#         self.video_label.image = img
    
#     def update_info(self):
#         """Обновление информационной метки"""
#         left = self.tracker.left_eye
#         right = self.tracker.right_eye
        
#         self.info_label.config(
#             text=f"Левый глаз: X: {left['x']}, Y: {left['y']} | "
#                  f"Диаметр: {left['diameter']:.1f} px | "
#                  f"Скорость: {left['speed']:.0f} px/s\n"
#                  f"Правый глаз: X: {right['x']}, Y: {right['y']} | "
#                  f"Диаметр: {right['diameter']:.1f} px | "
#                  f"Скорость: {right['speed']:.0f} px/s"
#         )
    
#     def show_graphs(self):
#         """Показывает графики"""
#         history = self.tracker.history
        
#         if len(history['x']) < 2:
#             messagebox.showwarning("Предупреждение", "Недостаточно данных для построения графиков")
#             return
        
#         plt.figure(figsize=(15, 5))
        
#         # График траектории
#         plt.subplot(1, 3, 1)
#         plt.plot(history['x'], history['y'], 'b-', alpha=0.5, linewidth=0.5)
#         plt.scatter(history['x'][0], history['y'][0], c='green', s=50, label='Старт')
#         plt.scatter(history['x'][-1], history['y'][-1], c='red', s=50, label='Финиш')
#         plt.title("Траектория взгляда (левый глаз)")
#         plt.gca().invert_yaxis()
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # График диаметра
#         plt.subplot(1, 3, 2)
#         plt.plot(history['timestamps'], history['diameter'], 'g-', linewidth=1)
#         plt.title("Диаметр зрачка (левый глаз, px)")
#         plt.xlabel("Секунды")
#         plt.ylabel("Диаметр (px)")
#         plt.grid(True, alpha=0.3)
        
#         # График скорости
#         plt.subplot(1, 3, 3)
#         plt.plot(history['timestamps'], history['speed'], 'r-', linewidth=1)
#         plt.title("Скорость движения (левый глаз, px/s)")
#         plt.xlabel("Секунды")
#         plt.ylabel("Скорость (px/s)")
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.show()
    
#     def go_back(self):
#         """Возврат в главное меню"""
#         if self.is_recording:
#             self.stop_recording()
        
#         if self.is_video_recording:
#             self.stop_video_recording()
        
#         self.db.close()
#         self.window.destroy()
#         self.main_window.deiconify()
    
#     def on_closing(self):
#         """Обработчик закрытия окна"""
#         self.go_back()

# record_window.py
import tkinter as tk
from tkinter import messagebox
import cv2
import time
import numpy as np
import threading
import os
from datetime import datetime


class RecordWindow:
    def __init__(self, window, main_window):
        self.window = window
        self.main_window = main_window
        self.window.title("Запись видео - Айтрекинг")

        # Компоненты
        self.is_recording = False
        self.is_video_recording = False
        self.recording_thread = None
        self.video_writer_raw = None  # Для чистого видео (без обработки)
        self.video_filename_raw = None
        self.video_start_time = 0

        # ИНИЦИАЛИЗИРУЕМ КАМЕРУ ДЛЯ ВИДЕОЗАПИСИ
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")

        # Создаем интерфейс
        self.setup_ui()

        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)


    def setup_ui(self):
        """Создание интерфейса"""
        # Верхняя панель с кнопками
        control_frame = tk.Frame(self.window, bg="#f0f0f0", height=60)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Кнопка записи видео (чистое видео)
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

        # Основная область для видео
        self.video_frame = tk.Frame(
            self.window,
            bg="black",
            width=780,
            height=450
        )
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)

        # Label для отображения видео
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Запускаем поток отображения видео
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()


    def toggle_video_recording(self):
        """Запуск/остановка записи чистого видео (без обработки)"""
        if not self.is_video_recording:
            self.start_video_recording()
        else:
            self.stop_video_recording()


    def start_video_recording(self):
        """Запуск записи чистого видео (без обработки)"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Камера не запущена.")
            return

        # Создаем папку для видео если её нет
        if not os.path.exists("videos"):
            os.makedirs("videos")

        # Читаем первый кадр, чтобы взять реальный размер
        ret, test_frame = self.cap.read()
        if not ret:
            messagebox.showerror("Ошибка", "Не удалось прочитать кадр с камеры")
            return

        # Возвращаем кадр в поток
        self.cap.read()

        height, width = test_frame.shape[:2]
        frame_size = (width, height)  # (ширина, высота)

        # Генерируем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename_raw = f"videos/recording_{timestamp}_raw.avi"

        # Настройки кодека
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30.0

        self.video_writer_raw = cv2.VideoWriter(
            self.video_filename_raw, fourcc, fps, frame_size
        )

        if not self.video_writer_raw.isOpened():
            messagebox.showerror("Ошибка", "Не удалось создать файл для записи видео")
            return

        self.is_video_recording = True
        self.video_start_time = time.time()

        self.video_record_btn.config(
            text="Остановить видео",
            bg="#f44336"
        )

        messagebox.showinfo(
            "Информация",
            f"Запись видео начата\nФайл: {self.video_filename_raw}"
        )


    def stop_video_recording(self):
        """Остановка записи видео"""
        self.is_video_recording = False

        if self.video_writer_raw:
            self.video_writer_raw.release()
            self.video_writer_raw = None

            # Показываем сообщение о сохранении
            if os.path.exists(self.video_filename_raw):
                file_size = os.path.getsize(self.video_filename_raw) / (1024 * 1024)
                messagebox.showinfo(
                    "Информация",
                    f"Чистое видео сохранено\nФайл: {self.video_filename_raw}\n"
                    f"Размер: {file_size:.2f} МБ"
                )

        self.video_record_btn.config(
            text="Запись видео",
            bg="#FF5722"
        )


    def video_loop(self):
        """Цикл отображения и записи видео"""
        while self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            # Зеркальное отображение
            frame = cv2.flip(frame, 1)

            # Если идет запись видео, пишем кадр в файл и добавляем индикатор
            if self.is_video_recording and self.video_writer_raw is not None:
                cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
                self.video_writer_raw.write(frame)

            # Подготовка кадра для отображения в tkinter
            display_frame = cv2.resize(frame, (780, 450))

            # Обновление интерфейса
            self.window.after(0, lambda: self.show_frame(display_frame))
            time.sleep(0.033)  # ~30 FPS


    def show_frame(self, frame):
        """Отображает кадр в интерфейсе"""
        if frame is None:
            return

        # Конвертируем в формат для tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())

        self.video_label.config(image=img)
        self.video_label.image = img


    def go_back(self):
        """Возврат в главное меню"""
        if self.is_video_recording:
            self.stop_video_recording()

        if self.cap:
            self.cap.release()

        self.window.destroy()
        self.main_window.deiconify()


    def on_closing(self):
        """Обработчик закрытия окна"""
        self.go_back()
