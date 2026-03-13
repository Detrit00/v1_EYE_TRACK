import tkinter as tk
from tkinter import messagebox
from utils import center_window

class CreateUserWindow:
    def __init__(self, parent, db, refresh_callback=None):
        self.parent = parent
        self.db = db
        self.refresh_callback = refresh_callback
        self.create_window()
    
    def create_window(self):
        """Создание окна создания пользователя"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Создание пользователя")
        self.window.grab_set()
        center_window(self.window, 500, 500)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создание виджетов"""
        # Поля ввода
        tk.Label(self.window, text="ФИО:", font=("Arial", 12)).pack(pady=(20, 5))
        self.full_name_entry = tk.Entry(self.window, font=("Arial", 12), width=30)
        self.full_name_entry.pack()
        
        tk.Label(self.window, text="Возраст:", font=("Arial", 12)).pack(pady=(10, 5))
        self.age_entry = tk.Entry(self.window, font=("Arial", 12), width=30)
        self.age_entry.pack()
        
        tk.Label(self.window, text="Комментарий:", font=("Arial", 12)).pack(pady=(10, 5))
        self.comment_text = tk.Text(self.window, font=("Arial", 12), width=30, height=5)
        self.comment_text.pack()
        
        # Кнопка сохранения
        save_btn = tk.Button(
            self.window,
            text="Сохранить",
            font=("Arial", 12),
            width=15,
            bg="#4CAF50",
            fg="white",
            command=self.save_user
        )
        save_btn.pack(pady=20)
        
        # Кнопка отмены
        cancel_btn = tk.Button(
            self.window,
            text="Отмена",
            font=("Arial", 12),
            width=15,
            bg="#f44336",
            fg="white",
            command=self.window.destroy
        )
        cancel_btn.pack(pady=5)
    
    def save_user(self):
        """Сохранение нового пользователя"""
        full_name = self.full_name_entry.get().strip()
        age = self.age_entry.get().strip()
        comment = self.comment_text.get("1.0", tk.END).strip()
        
        # Валидация
        if not full_name:
            messagebox.showwarning("Предупреждение", "Введите ФИО")
            return
        
        if self.db.create_user(full_name, age, comment):
            messagebox.showinfo("Успех", "Пользователь успешно создан!")
            if self.refresh_callback:
                self.refresh_callback()
            self.window.destroy()