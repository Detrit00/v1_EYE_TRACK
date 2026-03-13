import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import subprocess
import json
from chart_utils import show_charts_window
import math
import numpy as np
from utils import center_window
from video_processor import VideoPlayerWindow
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageTk


class UserCardWindow:
    def __init__(self, parent, db, user_id, refresh_callback=None):
        self.parent = parent
        self.db = db
        self.user_id = user_id
        self.refresh_callback = refresh_callback
        self.user = None
        self.comment_text = None
        self.full_name_entry = None
        self.age_entry = None
        self.edit_btn = None
        self.save_btn = None
        self.delete_btn = None
        self.edit_mode = False
        self.full_name_var = None
        self.age_var = None
        self.videos_listbox = None
        self.videos = []  # Список видео пользователя
        
        self.load_user_data()
        if self.user:
            self.create_window()
            self.load_videos()
    
    def load_user_data(self):
        """Загрузка данных пользователя"""
        self.user = self.db.get_user_by_id(self.user_id)
        if not self.user:
            messagebox.showerror("Ошибка", "Пользователь не найден")
    
    def load_videos(self):
        """Загрузка списка видео пользователя"""
        self.videos = self.db.get_user_videos(self.user_id)
        self.update_videos_list()
    
    def update_videos_list(self):
        """Обновление отображения списка видео"""
        if self.videos_listbox:
            self.videos_listbox.delete(0, tk.END)
            for video_data in self.videos:
                # Распаковываем данные видео
                if len(video_data) == 6:  # Новый формат с eye_tracking_data
                    video_id, title, file_path, original_name, eye_data, processed_at = video_data
                else:  # Старый формат
                    video_id, title, file_path, original_name = video_data
                    eye_data, processed_at = None, None
                
                # Добавляем индикатор обработки
                status = "✅" if eye_data else "⏳"
                display_text = f"{status} {title}"
                self.videos_listbox.insert(tk.END, display_text)
            
            # # Обновляем счетчик видео
            # if hasattr(self, 'video_count_label'):
            #     count = len(self.videos)
            #     processed_count = sum(1 for v in self.videos if len(v) >= 5 and v[4])
            #     self.video_count_label.config(text=f"Всего видео: {count} (обработано: {processed_count})")
    
    def create_window(self):
        """Создание окна карточки пользователя"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Карточка пользователя: {self.user[1]}")
        self.window.grab_set()
        center_window(self.window, 1200, 700)
        
        # Режим просмотра/редактирования
        self.edit_mode = False
        
        # Переменные для хранения значений
        self.full_name_var = tk.StringVar(value=self.user[1])
        self.age_var = tk.StringVar(value=self.user[2] if self.user[2] else "")
        
        self.create_widgets()
        self.update_widgets_state()
    
    def create_widgets(self):
        """Создание виджетов"""
        # Основной контейнер с двумя колонками
        main_container = tk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая колонка - информация о пользователе
        left_frame = tk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Правая колонка - видео
        right_frame = tk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # === ЛЕВАЯ КОЛОНКА: ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ ===
        info_frame = tk.LabelFrame(left_frame, text="Информация о пользователе", font=("Arial", 12, "bold"))
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ID (только для чтения)
        tk.Label(info_frame, text="ID:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5, padx=10)
        tk.Label(info_frame, text=self.user[0], font=("Arial", 12)).grid(row=0, column=1, sticky="w", pady=5, padx=10)
        
        # ФИО (редактируемое)
        tk.Label(info_frame, text="ФИО:*", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5, padx=10)
        self.full_name_entry = tk.Entry(info_frame, font=("Arial", 12), width=25, textvariable=self.full_name_var)
        self.full_name_entry.grid(row=1, column=1, pady=5, padx=10, sticky="w")
        
        # Возраст (редактируемый)
        tk.Label(info_frame, text="Возраст:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=5, padx=10)
        self.age_entry = tk.Entry(info_frame, font=("Arial", 12), width=10, textvariable=self.age_var)
        self.age_entry.grid(row=2, column=1, pady=5, padx=10, sticky="w")
        
        # Комментарий (редактируемый)
        tk.Label(info_frame, text="Комментарий:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky="nw", pady=5, padx=10)
        
        # Создаем фрейм для комментария со скроллбаром
        comment_frame = tk.Frame(info_frame)
        comment_frame.grid(row=3, column=1, pady=5, padx=10, sticky="nsew")
        comment_frame.columnconfigure(0, weight=1)
        comment_frame.rowconfigure(0, weight=1)
        
        # Текстовое поле для комментария
        self.comment_text = tk.Text(comment_frame, font=("Arial", 12), width=25, height=4, wrap=tk.WORD)
        self.comment_text.grid(row=0, column=0, sticky="nsew")
        self.comment_text.insert("1.0", self.user[3] if self.user[3] else "")
        
        # Скроллбар для комментария
        comment_scrollbar = tk.Scrollbar(comment_frame, orient=tk.VERTICAL, command=self.comment_text.yview)
        comment_scrollbar.grid(row=0, column=1, sticky="ns")
        self.comment_text.config(yscrollcommand=comment_scrollbar.set)
        
        # Дата создания (только для чтения)
        tk.Label(info_frame, text="Дата создания:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky="w", pady=5, padx=10)
        tk.Label(info_frame, text=self.user[4], font=("Arial", 12)).grid(row=4, column=1, sticky="w", pady=5, padx=10)
        
        # Настройка весов для растягивания
        info_frame.columnconfigure(1, weight=1)
        info_frame.rowconfigure(3, weight=1)
        
        # === ПРАВАЯ КОЛОНКА: ВИДЕО ===
        video_frame = tk.LabelFrame(right_frame, text="Видео пользователя", font=("Arial", 12, "bold"))
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Кнопки управления видео
        video_btn_frame = tk.Frame(video_frame)
        video_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        add_video_btn = tk.Button(
            video_btn_frame,
            text="Добавить видео",
            font=("Arial", 10),
            bg="#4CAF50",
            fg="white",
            command=self.add_video
        )
        add_video_btn.pack(side=tk.LEFT, padx=2)
        
        analyze_video_btn = tk.Button(
            video_btn_frame,
            text="Анализировать видео",
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            command=self.analyze_video
        )
        analyze_video_btn.pack(side=tk.LEFT, padx=2)
        
        show_charts_btn = tk.Button(
            video_btn_frame,
            text="Показать графики",
            font=("Arial", 10),
            bg="#FF9800",
            fg="white",
            command=self.show_video_charts
        )
        show_charts_btn.pack(side=tk.LEFT, padx=2)
        
        delete_video_btn = tk.Button(
            video_btn_frame,
            text="Удалить видео",
            font=("Arial", 10),
            bg="#f44336",
            fg="white",
            command=self.delete_video
        )
        delete_video_btn.pack(side=tk.LEFT, padx=2)
        
        # Список видео
        list_frame = tk.Frame(video_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Создаем Listbox с прокруткой
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.videos_listbox = tk.Listbox(
            list_frame,
            font=("Arial", 10),
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            height=8
        )
        self.videos_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.videos_listbox.yview)
        
        # Привязываем двойной клик к анализу видео
        self.videos_listbox.bind("<Double-1>", lambda e: self.analyze_video())
        
        # # Информация о количестве видео
        # self.video_count_label = tk.Label(video_frame, text="Всего видео: 0", font=("Arial", 10, "italic"))
        # self.video_count_label.pack(pady=5)
        
        # === КНОПКИ ДЕЙСТВИЙ ВНИЗУ ===
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)
        
        # Кнопка редактирования/просмотра
        self.edit_btn = tk.Button(
            button_frame,
            text="Редактировать",
            font=("Arial", 11),
            width=15,
            bg="#2196F3",
            fg="white",
            command=self.toggle_edit_mode
        )
        self.edit_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка сохранения
        self.save_btn = tk.Button(
            button_frame,
            text="💾 Сохранить",
            font=("Arial", 11),
            width=15,
            bg="#4CAF50",
            fg="white",
            command=self.save_changes
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка удаления пользователя
        self.delete_btn = tk.Button(
            button_frame,
            text="🗑 Удалить пользователя",
            font=("Arial", 11),
            width=15,
            bg="#f44336",
            fg="white",
            command=self.delete_user
        )
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка закрытия
        close_btn = tk.Button(
            button_frame,
            text="✖ Закрыть",
            font=("Arial", 11),
            width=15,
            bg="#9E9E9E",
            fg="white",
            command=self.window.destroy
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def add_video(self):
        """Добавление видео пользователю"""
        # Выбор видеофайла
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Диалог для ввода названия видео
        dialog = tk.Toplevel(self.window)
        dialog.title("Название видео")
        dialog.geometry("400x150")
        dialog.grab_set()
        center_window(dialog, 400, 150)
        
        tk.Label(dialog, text=f"Файл: {os.path.basename(file_path)}", font=("Arial", 10)).pack(pady=10)
        tk.Label(dialog, text="Введите название видео:", font=("Arial", 11)).pack(pady=5)
        
        title_entry = tk.Entry(dialog, font=("Arial", 11), width=40)
        title_entry.pack(pady=5)
        # Предлагаем имя файла без расширения как название по умолчанию
        default_title = os.path.splitext(os.path.basename(file_path))[0]
        title_entry.insert(0, default_title)
        
        def save_video():
            title = title_entry.get().strip()
            if not title:
                messagebox.showwarning("Предупреждение", "Введите название видео")
                return
            
            if self.db.add_user_video(self.user_id, title, file_path):
                messagebox.showinfo("Успех", "Видео успешно добавлено!")
                self.load_videos()  # Обновляем список видео
                dialog.destroy()
            else:
                messagebox.showerror("Ошибка", "Не удалось добавить видео")
        
        tk.Button(dialog, text="Сохранить", command=save_video, bg="#4CAF50", fg="white", width=15).pack(pady=10)
    
    def analyze_video(self):
        """Анализ выбранного видео с айтрекингом"""
        selection = self.videos_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите видео для анализа")
            return
        
        index = selection[0]
        video_data = self.videos[index]
        
        # Распаковываем данные видео
        if len(video_data) >= 5:
            video_id, title, file_path, original_name, eye_data = video_data[:5]
        else:
            video_id, title, file_path, original_name = video_data
            eye_data = None
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            messagebox.showerror("Ошибка", "Файл видео не найден")
            return
        
        # Открываем окно плеера с айтрекингом
        from video_processor import VideoPlayerWindow
        VideoPlayerWindow(self.window, self.db, self.user_id, video_id, file_path, title)
    
    def show_video_charts(self):
        """Показать графики для выбранного видео"""
        selection = self.videos_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите видео для просмотра графиков")
            return
        
        index = selection[0]
        video_data = self.videos[index]
        
        video_id = video_data[0]
        
        # Получаем данные из БД
        video_info = self.db.get_video_data(video_id)
        if not video_info:
            messagebox.showerror("Ошибка", "Не удалось загрузить данные видео")
            return
        
        # # Отладочный вывод (можно удалить после исправления)
        # print(f"video_info: {video_info}")
        # print(f"type: {type(video_info)}")
        # print(f"length: {len(video_info) if video_info else 0}")
        
        # Инициализируем переменные
        title = "Видео"
        eye_data_json = None
        
        # Распаковываем данные в зависимости от длины кортежа
        if video_info and len(video_info) >= 6:
            # Формат: (id, user_id, title, file_path, eye_tracking_data, charts_path)
            _, _, title, _, eye_data_json, _ = video_info
        elif video_info and len(video_info) >= 5:
            # Формат: (id, user_id, title, file_path, eye_tracking_data)
            _, _, title, _, eye_data_json = video_info
        elif video_info and len(video_info) > 2:
            # Минимальный формат: (id, user_id, title, ...)
            title = video_info[2]
        
        # Проверяем, что eye_data_json не None и не пустая строка
        if not eye_data_json or eye_data_json == "null" or eye_data_json == "":
            messagebox.showinfo("Информация", 
                            "Видео еще не обработано или данные повреждены.\n"
                            "Сначала выполните анализ видео.")
            return
        
        # Преобразуем JSON-строку в словарь
        try:
            import json
            from chart_utils import show_charts_window
            
            # Убеждаемся, что eye_data_json - это строка
            if not isinstance(eye_data_json, str):
                eye_data_json = str(eye_data_json)
                
            data = json.loads(eye_data_json)
            
            # Проверяем, что данные содержат необходимые ключи
            if not data or ('left_eye' not in data and 'right_eye' not in data):
                messagebox.showwarning("Предупреждение", 
                                    "Данные имеют неверный формат")
                return
                
            show_charts_window(self.window, title, data, "Сохранённые данные")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {e}\n"
                                f"Получено: {eye_data_json[:100]}...")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неожиданная ошибка: {e}")
    
    def delete_video(self):
        """Удаление выбранного видео"""
        selection = self.videos_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите видео для удаления")
            return
        
        index = selection[0]
        video_data = self.videos[index]
        video_id, title, file_path, original_name = video_data[:4]
        
        if messagebox.askyesno("Подтверждение", f"Удалить видео '{title}'?"):
            if self.db.delete_user_video(video_id, file_path):
                messagebox.showinfo("Успех", "Видео удалено")
                self.load_videos()
            else:
                messagebox.showerror("Ошибка", "Не удалось удалить видео")
    
    def update_videos_list(self):
        """Обновление отображения списка видео"""
        if self.videos_listbox:
            self.videos_listbox.delete(0, tk.END)
            for video_data in self.videos:
                if len(video_data) >= 5:
                    video_id, title, file_path, original_name, eye_data = video_data[:5]
                    processed_at = video_data[5] if len(video_data) > 5 else None
                else:
                    video_id, title, file_path, original_name = video_data
                    eye_data, processed_at = None, None
                
                status = "✅" if eye_data else "⏳"
                display_text = f"{status} {title}"
                self.videos_listbox.insert(tk.END, display_text)
            
            # Обновляем счетчик
            if hasattr(self, 'video_count_label'):
                count = len(self.videos)
                processed = sum(1 for v in self.videos if len(v) >= 5 and v[4])
                self.video_count_label.config(text=f"Всего видео: {count} (обработано: {processed})")
    
    def toggle_edit_mode(self):
        """Переключение между режимами просмотра и редактирования"""
        if self.edit_mode:
            # Переход в режим просмотра (отмена редактирования)
            if messagebox.askyesno("Подтверждение", "Отменить изменения?"):
                self.edit_mode = False
                # Восстанавливаем исходные значения
                self.full_name_var.set(self.user[1])
                self.age_var.set(self.user[2] if self.user[2] else "")
                self.comment_text.delete("1.0", tk.END)
                self.comment_text.insert("1.0", self.user[3] if self.user[3] else "")
                self.update_widgets_state()
        else:
            # Переход в режим редактирования
            self.edit_mode = True
            self.update_widgets_state()
    
    def update_widgets_state(self):
        """Обновление состояния виджетов в зависимости от режима"""
        # Проверяем, что все виджеты существуют
        if not all([self.full_name_entry, self.age_entry, self.comment_text]):
            return
        
        # Обновление состояния полей ввода
        if self.edit_mode:
            # Режим редактирования - все поля доступны
            self.full_name_entry.config(state=tk.NORMAL)
            self.age_entry.config(state=tk.NORMAL)
            self.comment_text.config(state=tk.NORMAL)
            
            # Обновление кнопок
            self.edit_btn.config(text="👁 Просмотр", bg="#FF9800")
            self.save_btn.config(state=tk.NORMAL)
            self.delete_btn.config(state=tk.DISABLED)
        else:
            # Режим просмотра - все поля заблокированы
            self.full_name_entry.config(state=tk.DISABLED)
            self.age_entry.config(state=tk.DISABLED)
            self.comment_text.config(state=tk.DISABLED)
            
            # Обновление кнопок
            self.edit_btn.config(text="✏ Редактировать", bg="#2196F3")
            self.save_btn.config(state=tk.DISABLED)
            self.delete_btn.config(state=tk.NORMAL)
    
    def save_changes(self):
        """Сохранение изменений в базе данных"""
        if not self.comment_text:
            messagebox.showerror("Ошибка", "Ошибка инициализации поля комментария")
            return
        
        full_name = self.full_name_var.get().strip()
        age = self.age_var.get().strip()
        comment = self.comment_text.get("1.0", tk.END).strip()
        
        if not full_name:
            messagebox.showwarning("Предупреждение", "ФИО не может быть пустым")
            return
        
        if self.db.update_user(self.user_id, full_name, age, comment):
            messagebox.showinfo("Успех", "Данные успешно обновлены!")
            
            # Обновление заголовка окна
            self.window.title(f"Карточка пользователя: {full_name}")
            
            # Обновление сохраненных данных
            self.user = self.db.get_user_by_id(self.user_id)
            
            # Возврат в режим просмотра
            self.edit_mode = False
            self.update_widgets_state()
            
            # Обновление списка пользователей, если есть функция обновления
            if self.refresh_callback:
                self.refresh_callback()
    
    def delete_user(self):
        """Удаление пользователя с подтверждением"""
        if messagebox.askyesno("Подтверждение удаления", 
                              f"Вы уверены, что хотите удалить пользователя\n{self.user[1]}?\n\nВсе видео пользователя также будут удалены!\nЭто действие нельзя отменить!"):
            if self.db.delete_user(self.user_id):
                messagebox.showinfo("Успех", "Пользователь успешно удален!")
                self.window.destroy()
                
                # Обновление списка пользователей, если есть функция обновления
                if self.refresh_callback:
                    self.refresh_callback()