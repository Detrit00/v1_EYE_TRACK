import tkinter as tk
from tkinter import ttk, messagebox
from utils import center_window
from user_card import UserCardWindow

class UsersListWindow:
    def __init__(self, parent, db):
        self.parent = parent
        self.db = db
        self.create_window()
    
    def create_window(self):
        """Создание окна списка пользователей"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Список пациентов")
        self.window.grab_set()
        center_window(self.window, 1000, 500)
        
        self.create_widgets()
        self.load_users()
    
    def create_widgets(self):
        """Создание виджетов"""
        # Заголовок
        tk.Label(
            self.window,
            text="Список пациентов",
            font=("Arial", 16, "bold")
        ).pack(pady=10)
        
        # Поиск
        search_frame = tk.Frame(self.window)
        search_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(search_frame, text="Поиск:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        self.search_entry = tk.Entry(search_frame, font=("Arial", 11), width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<Return>', lambda e: self.on_search())
        
        # Кнопка поиска
        search_btn = tk.Button(
            search_frame,
            text="Найти",
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            command=self.on_search
        )
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка очистки поиска
        clear_btn = tk.Button(
            search_frame,
            text="Сбросить",
            font=("Arial", 10),
            bg="#FF9800",
            fg="white",
            command=self.clear_search
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Создаем Treeview для отображения пользователей
        self.create_treeview()
        
        # Кнопки действий
        self.create_action_buttons()
    
    def create_treeview(self):
        """Создание таблицы для отображения пользователей"""
        columns = ("ID", "ФИО", "Возраст", "Комментарий")
        self.tree = ttk.Treeview(self.window, columns=columns, show="headings", height=15)
        
        # Настройка колонок
        self.tree.heading("ID", text="ID")
        self.tree.heading("ФИО", text="ФИО")
        self.tree.heading("Возраст", text="Возраст")
        self.tree.heading("Комментарий", text="Комментарий")
        
        self.tree.column("ID", width=50, anchor="center")
        self.tree.column("ФИО", width=200)
        self.tree.column("Возраст", width=80, anchor="center")
        self.tree.column("Комментарий", width=220)
        
        # Добавляем скроллбар
        scrollbar = ttk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # Привязываем двойной клик к открытию карточки
        self.tree.bind("<Double-1>", lambda e: self.view_user())
    
    def create_action_buttons(self):
        """Создание кнопок действий"""
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        view_btn = tk.Button(
            btn_frame,
            text="Открыть карточку",
            font=("Arial", 11),
            width=15,
            bg="#2196F3",
            fg="white",
            command=self.view_user
        )
        view_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(
            btn_frame,
            text="Закрыть",
            font=("Arial", 11),
            width=15,
            bg="#f44336",
            fg="white",
            command=self.window.destroy
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def load_users(self, search_text=""):
        """Загрузка пользователей из базы данных"""
        # Очищаем текущие данные
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        users = self.db.get_all_users(search_text)
        for user in users:
            self.tree.insert("", tk.END, values=user)
    
    def on_search(self):
        """Обработчик поиска"""
        search_text = self.search_entry.get().strip()
        self.load_users(search_text)
    
    def clear_search(self):
        """Очистка поиска"""
        self.search_entry.delete(0, tk.END)
        self.load_users()
    
    def view_user(self):
        """Просмотр карточки выбранного пользователя"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя из списка")
            return
        
        user_id = self.tree.item(selected[0])['values'][0]
        UserCardWindow(self.window, self.db, user_id, self.load_users)