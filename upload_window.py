import tkinter as tk
from utils import center_window
from database_USER import Database
from create_user_window import CreateUserWindow
from users_list_window import UsersListWindow

class UploadWindow:
    def __init__(self, window, main_window):
        self.window = window
        self.main_window = main_window
        self.window.title("Пациенты")
        
        # Центрируем окно
        center_window(self.window, 500, 400)
        
        # Инициализация базы данных
        self.db = Database()
        
        # Создание виджетов
        self.create_widgets()
        
        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Создание виджетов главного окна"""
        # # Заголовок
        # title_label = tk.Label(
        #     self.window,
        #     text="Окно загрузки видео",
        #     font=("Arial", 16, "bold")
        # )
        # title_label.pack(pady=20)
        
        # Информация
        # info_label = tk.Label(
        #     self.window,
        #     text="Здесь будет функционал загрузки видео",
        #     font=("Arial", 12)
        # )
        # info_label.pack(pady=10)
        
        # Фрейм для кнопок управления пользователями
        user_frame = tk.Frame(self.window)
        user_frame.pack(pady=20)
        
        # Кнопка "Список пользователей"
        list_users_btn = tk.Button(
            user_frame,
            text="Список пациентов",
            font=("Arial", 12),
            width=20,
            height=2,
            bg="#2196F3",
            fg="white",
            command=self.show_users_list
        )
        list_users_btn.pack(side=tk.LEFT, padx=10)
        
        # Кнопка "Создать пользователя"
        create_user_btn = tk.Button(
            user_frame,
            text="Создать пациента",
            font=("Arial", 12),
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white",
            command=self.create_user
        )
        create_user_btn.pack(side=tk.LEFT, padx=10)
        
        # Кнопка "Назад"
        back_btn = tk.Button(
            self.window,
            text="Назад в меню",
            font=("Arial", 12),
            width=15,
            height=2,
            bg="#FF9800",
            fg="white",
            command=self.go_back
        )
        back_btn.pack(pady=20)
    
    def create_user(self):
        """Открытие окна создания нового пользователя"""
        CreateUserWindow(self.window, self.db, None)
    
    def show_users_list(self):
        """Отображение списка пользователей"""
        UsersListWindow(self.window, self.db)
    
    def go_back(self):
        """Возврат в главное меню"""
        self.db.close()
        self.window.destroy()
        self.main_window.deiconify()  # Показываем главное окно
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        self.go_back()