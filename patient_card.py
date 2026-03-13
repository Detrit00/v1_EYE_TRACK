import os
import sqlite3
import shutil
import uuid
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import subprocess
from upload_window import UploadWindow  # импортируем ваш класс

# Константы
DB_NAME = "clinic.db"
VIDEOS_DIR = "videos"

# Создание папки для видео, если её нет
os.makedirs(VIDEOS_DIR, exist_ok=True)


class DatabaseManager:
    """Класс для работы с базой данных SQLite"""

    def __init__(self, db_name=DB_NAME):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """Создание таблиц, если они не существуют"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                comment TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_filename TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
            )
        ''')
        self.conn.commit()

    def get_patients(self):
        """Возвращает список всех пациентов (id, full_name, comment)"""
        self.cursor.execute("SELECT id, full_name, comment FROM patients ORDER BY full_name")
        return self.cursor.fetchall()

    def add_patient(self, full_name, comment):
        """Добавляет нового пациента и возвращает его id"""
        self.cursor.execute(
            "INSERT INTO patients (full_name, comment) VALUES (?, ?)",
            (full_name, comment)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_patient(self, patient_id):
        """Возвращает данные пациента по id"""
        self.cursor.execute("SELECT full_name, comment FROM patients WHERE id=?", (patient_id,))
        return self.cursor.fetchone()

    def get_videos(self, patient_id):
        """Возвращает список видео для пациента (id, title, file_path)"""
        self.cursor.execute(
            "SELECT id, title, file_path FROM videos WHERE patient_id=? ORDER BY title",
            (patient_id,)
        )
        return self.cursor.fetchall()

    def add_video(self, patient_id, title, source_path):
        """Копирует видео в папку VIDEOS_DIR и добавляет запись в БД"""
        # Генерация уникального имени файла
        ext = os.path.splitext(source_path)[1]
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(VIDEOS_DIR, unique_filename)

        # Копирование файла
        shutil.copy2(source_path, dest_path)

        # Сохранение в БД
        self.cursor.execute(
            "INSERT INTO videos (patient_id, title, file_path, original_filename) VALUES (?, ?, ?, ?)",
            (patient_id, title, dest_path, os.path.basename(source_path))
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def delete_video(self, video_id, file_path):
        """Удаляет запись о видео и сам файл (опционально)"""
        self.cursor.execute("DELETE FROM videos WHERE id=?", (video_id,))
        self.conn.commit()
        if os.path.exists(file_path):
            os.remove(file_path)

    def close(self):
        self.conn.close()


class AddPatientWindow(Toplevel):
    """Окно добавления нового пациента"""

    def __init__(self, parent, db_manager, callback):
        super().__init__(parent)
        self.parent = parent
        self.db = db_manager
        self.callback = callback  # функция для обновления списка в главном окне

        self.title("Новый пациент")
        self.geometry("400x200")
        self.resizable(False, False)

        # Поля ввода
        ttk.Label(self, text="ФИО:").pack(pady=5)
        self.entry_name = ttk.Entry(self, width=50)
        self.entry_name.pack(pady=5)

        ttk.Label(self, text="Комментарий:").pack(pady=5)
        self.entry_comment = ttk.Entry(self, width=50)
        self.entry_comment.pack(pady=5)

        # Кнопки
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Сохранить", command=self.save_patient).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side=LEFT, padx=5)

    def save_patient(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "ФИО не может быть пустым")
            return
        comment = self.entry_comment.get().strip()
        patient_id = self.db.add_patient(name, comment)
        messagebox.showinfo("Успех", f"Пациент {name} добавлен с ID {patient_id}")
        self.callback()  # обновляем список в главном окне
        self.destroy()


class PatientCardWindow(Toplevel):
    """Окно карточки пациента (просмотр информации и видео)"""

    def __init__(self, parent, db_manager, patient_id):
        super().__init__(parent)
        self.parent = parent
        self.db = db_manager
        self.patient_id = patient_id

        self.title("Карточка пациента")
        self.geometry("500x400")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Загружаем данные пациента
        patient_data = self.db.get_patient(patient_id)
        if not patient_data:
            messagebox.showerror("Ошибка", "Пациент не найден")
            self.destroy()
            return
        self.full_name, self.comment = patient_data

        # Фрейм с информацией
        info_frame = ttk.LabelFrame(self, text="Информация о пациенте", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(info_frame, text=f"ФИО: {self.full_name}").pack(anchor="w")
        ttk.Label(info_frame, text=f"Комментарий: {self.comment}").pack(anchor="w")

        # Фрейм со списком видео
        video_frame = ttk.LabelFrame(self, text="Видео", padding=10)
        video_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Кнопки управления видео
        btn_frame = ttk.Frame(video_frame)
        btn_frame.pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Добавить видео", command=self.add_video).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Удалить видео", command=self.delete_video).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Открыть видео", command=self.open_video).pack(side=LEFT, padx=5)

        # Список видео
        self.video_listbox = Listbox(video_frame, height=10)
        self.video_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.video_listbox.bind("<Double-Button-1>", lambda e: self.open_video())

        # Загружаем видео
        self.load_videos()

    def load_videos(self):
        """Обновляет список видео из БД"""
        self.video_listbox.delete(0, END)
        self.videos = self.db.get_videos(self.patient_id)  # список (id, title, file_path)
        for vid_id, title, path in self.videos:
            self.video_listbox.insert(END, title)

    def add_video(self):
        """Открывает диалог выбора видео и добавления названия"""
        file_paths = filedialog.askopenfilenames(
            title="Выберите видеофайлы",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if not file_paths:
            return

        # Для каждого выбранного файла запрашиваем название
        for path in file_paths:
            # Диалог для названия
            title_dialog = Toplevel(self)
            title_dialog.title("Название видео")
            title_dialog.geometry("300x120")
            title_dialog.resizable(False, False)

            ttk.Label(title_dialog, text=f"Файл: {os.path.basename(path)}").pack(pady=5)
            ttk.Label(title_dialog, text="Введите название видео:").pack(pady=5)
            title_entry = ttk.Entry(title_dialog, width=40)
            title_entry.pack(pady=5)
            title_entry.insert(0, os.path.splitext(os.path.basename(path))[0])  # имя файла без расширения

            def save_video(title_entry=title_entry, path=path, dialog=title_dialog):
                title = title_entry.get().strip()
                if not title:
                    messagebox.showerror("Ошибка", "Название не может быть пустым")
                    return
                try:
                    self.db.add_video(self.patient_id, title, path)
                    self.load_videos()
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось добавить видео: {e}")

            ttk.Button(title_dialog, text="Сохранить", command=save_video).pack(pady=5)
            # Ждём закрытия диалога перед следующим файлом
            self.wait_window(title_dialog)

    def delete_video(self):
        """Удаляет выбранное видео из БД и с диска"""
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите видео для удаления")
            return
        index = selection[0]
        video_id, title, file_path = self.videos[index]

        # Подтверждение
        if messagebox.askyesno("Подтверждение", f"Удалить видео '{title}'?"):
            self.db.delete_video(video_id, file_path)
            self.load_videos()

    def open_video(self):
        """Открывает выбранное видео системным плеером"""
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите видео для открытия")
            return
        index = selection[0]
        _, _, file_path = self.videos[index]
        if os.path.exists(file_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # Linux или macOS
                    subprocess.run(['xdg-open', file_path])
                else:
                    messagebox.showerror("Ошибка", "Неизвестная операционная система")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")
        else:
            messagebox.showerror("Ошибка", "Файл видео не найден")

    def on_close(self):
        self.destroy()


class MainWindow:
    """Главное окно приложения"""

    def __init__(self, root):
        self.root = root
        self.root.title("Электронная картотека пациентов")
        self.root.geometry("600x400")

        self.db = DatabaseManager()

        # Панель с кнопками
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(btn_frame, text="Добавить пациента", command=self.open_add_patient).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="Обновить список", command=self.load_patients).pack(side=LEFT, padx=5)
        
        # Новая кнопка для открытия окна загрузки видео
        ttk.Button(btn_frame, text="Загрузить видео", command=self.open_upload_window).pack(side=LEFT, padx=5)

        # Список пациентов
        list_frame = ttk.Frame(root)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Дерево (Treeview) для отображения пациентов
        columns = ("id", "full_name", "comment")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.heading("full_name", text="ФИО")
        self.tree.heading("comment", text="Комментарий")
        self.tree.column("id", width=50, anchor="center")
        self.tree.column("full_name", width=250)
        self.tree.column("comment", width=250)

        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.tree.pack(side=LEFT, fill="both", expand=True)

        self.tree.bind("<Double-1>", self.open_patient_card)

        # Загружаем данные
        self.load_patients()

    def load_patients(self):
        """Загружает список пациентов из БД и обновляет Treeview"""
        # Очистка
        for row in self.tree.get_children():
            self.tree.delete(row)

        patients = self.db.get_patients()
        for patient in patients:
            self.tree.insert("", END, values=patient)

    def open_add_patient(self):
        """Открывает окно добавления пациента"""
        AddPatientWindow(self.root, self.db, self.load_patients)

    def open_upload_window(self):
        """Открывает окно загрузки видео"""
        # Скрываем главное окно
        self.root.withdraw()
        # Создаем новое окно для загрузки видео
        upload_window = Toplevel(self.root)
        # Передаем ссылку на главное окно для возможности вернуться
        UploadWindow(upload_window, self.root)

    def open_patient_card(self, event):
        """Открывает карточку выбранного пациента"""
        selected = self.tree.selection()
        if not selected:
            return
        item = self.tree.item(selected[0])
        patient_id = item['values'][0]
        PatientCardWindow(self.root, self.db, patient_id)

    def on_close(self):
        self.db.close()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = MainWindow(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()