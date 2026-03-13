import tkinter as tk
from tkinter import messagebox
import sys
from record_window import RecordWindow
from upload_window import UploadWindow
from math_specialist_window import MathSpecialistWindow
from utils import center_window

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Главное меню")
        
        # Центрируем окно
        center_window(self.root, 400, 350)
        
        # Заголовок
        title_label = tk.Label(
            self.root, 
            text="Главное меню", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=20)
        
        # Кнопка "Начать запись видео"
        record_btn = tk.Button(
            self.root,
            text="Запись видео",
            font=("Arial", 12),
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white",
            command=self.open_record_window
        )
        record_btn.pack(pady=5)
        
        # Кнопка "Работа с пациентами"
        upload_btn = tk.Button(
            self.root,
            text="Работа с пациентами",
            font=("Arial", 12),
            width=20,
            height=2,
            bg="#2196F3",
            fg="white",
            command=self.open_upload_window
        )
        upload_btn.pack(pady=5)
        
        # Кнопка для специалиста по математическому обеспечению
        math_btn = tk.Button(
            self.root,
            text="Настройка алгоритмов",
            font=("Arial", 12),
            width=20,
            height=2,
            bg="#9C27B0",
            fg="white",
            command=self.open_math_specialist_window
        )
        math_btn.pack(pady=5)
        
        # Кнопка выхода
        exit_btn = tk.Button(
            self.root,
            text="Выход",
            font=("Arial", 10),
            width=10,
            bg="#f44336",
            fg="white",
            command=self.exit_app
        )
        exit_btn.pack(pady=20)
    
    def open_record_window(self):
        """Открывает окно записи видео"""
        self.root.withdraw()
        record_window = tk.Toplevel(self.root)
        RecordWindow(record_window, self.root)
    
    def open_upload_window(self):
        """Открывает окно загрузки видео"""
        self.root.withdraw()
        upload_window = tk.Toplevel(self.root)
        UploadWindow(upload_window, self.root)
    
    def open_math_specialist_window(self):
        """Открывает окно для специалиста по математическому обеспечению"""
        self.root.withdraw()
        # Убрал создание лишнего окна, просто передаем root
        MathSpecialistWindow(self.root, None)
    
    def exit_app(self):
        """Выход из приложения"""
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            self.root.quit()
            self.root.destroy()
            sys.exit()