# database_USER.py
import sqlite3
import os
import shutil
import uuid
from tkinter import messagebox

class Database:
    def __init__(self, db_name='users.db'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.videos_dir = "user_videos"
        os.makedirs(self.videos_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            
            # Создание таблицы пользователей, если её нет
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    age INTEGER,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Создание таблицы видео для пользователей
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    original_filename TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # Проверяем и добавляем недостающие колонки
            self._add_missing_columns()
            
            self.conn.commit()
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при инициализации базы данных: {e}")
    
    def _add_missing_columns(self):
        """Добавление недостающих колонок в таблицу user_videos"""
        try:
            # Получаем список существующих колонок
            self.cursor.execute("PRAGMA table_info(user_videos)")
            existing_columns = [column[1] for column in self.cursor.fetchall()]
            
            # Добавляем колонку eye_tracking_data если её нет
            if 'eye_tracking_data' not in existing_columns:
                self.cursor.execute("ALTER TABLE user_videos ADD COLUMN eye_tracking_data TEXT")
                print("Добавлена колонка eye_tracking_data")
            
            # Добавляем колонку charts_path если её нет
            if 'charts_path' not in existing_columns:
                self.cursor.execute("ALTER TABLE user_videos ADD COLUMN charts_path TEXT")
                print("Добавлена колонка charts_path")
            
            # Добавляем колонку processed_at если её нет
            if 'processed_at' not in existing_columns:
                self.cursor.execute("ALTER TABLE user_videos ADD COLUMN processed_at TIMESTAMP")
                print("Добавлена колонка processed_at")
                
        except sqlite3.Error as e:
            print(f"Ошибка при добавлении колонок: {e}")
    
    def get_connection(self):
        """Получение соединения с базой данных"""
        if not self.conn:
            self.init_database()
        return self.conn, self.cursor
    
    def create_user(self, full_name, age, comment):
        """Создание нового пользователя"""
        try:
            self.cursor.execute(
                "INSERT INTO users (full_name, age, comment) VALUES (?, ?, ?)",
                (full_name, age if age else None, comment)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {e}")
            return False
    
    def get_all_users(self, search_text=""):
        """Получение всех пользователей с возможностью поиска"""
        try:
            if search_text:
                self.cursor.execute(
                    "SELECT id, full_name, age, comment FROM users WHERE full_name LIKE ? ORDER BY full_name",
                    (f'%{search_text}%',)
                )
            else:
                self.cursor.execute("SELECT id, full_name, age, comment FROM users ORDER BY full_name")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке списка: {e}")
            return []
    
    def get_user_by_id(self, user_id):
        """Получение пользователя по ID"""
        try:
            self.cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {e}")
            return None
    
    def update_user(self, user_id, full_name, age, comment):
        """Обновление данных пользователя"""
        try:
            self.cursor.execute(
                "UPDATE users SET full_name = ?, age = ?, comment = ? WHERE id = ?",
                (full_name, age if age else None, comment, user_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении: {e}")
            return False
    
    def delete_user(self, user_id):
        """Удаление пользователя"""
        try:
            # Сначала удаляем все видео пользователя с диска
            videos = self.get_user_videos(user_id)
            for video in videos:
                if len(video) >= 3:  # (id, title, file_path, ...)
                    file_path = video[2]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Также удаляем папку с графиками
                    charts_dir = os.path.join(self.videos_dir, f'user_{user_id}', 'charts')
                    if os.path.exists(charts_dir):
                        shutil.rmtree(charts_dir, ignore_errors=True)
            
            # Затем удаляем записи из БД
            self.cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            self.conn.commit()
            
            # Удаляем папку пользователя
            user_dir = os.path.join(self.videos_dir, f'user_{user_id}')
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir, ignore_errors=True)
                
            return True
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при удалении: {e}")
            return False
    
    # Методы для работы с видео
    def get_user_videos(self, user_id):
        """Получение списка видео пользователя"""
        try:
            # Проверяем существование колонок перед запросом
            self.cursor.execute("PRAGMA table_info(user_videos)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            if 'eye_tracking_data' in columns and 'processed_at' in columns:
                self.cursor.execute(
                    "SELECT id, title, file_path, original_filename, eye_tracking_data, processed_at FROM user_videos WHERE user_id=? ORDER BY uploaded_at DESC",
                    (user_id,)
                )
            else:
                self.cursor.execute(
                    "SELECT id, title, file_path, original_filename FROM user_videos WHERE user_id=? ORDER BY uploaded_at DESC",
                    (user_id,)
                )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке видео: {e}")
            return []
    
    def add_user_video(self, user_id, title, source_path):
        """Добавление видео пользователю"""
        try:
            # Создаем папку для пользователя
            user_dir = os.path.join(self.videos_dir, f'user_{user_id}')
            os.makedirs(user_dir, exist_ok=True)
            
            # Генерация уникального имени файла
            ext = os.path.splitext(source_path)[1]
            unique_filename = f"{uuid.uuid4().hex}{ext}"
            dest_path = os.path.join(user_dir, unique_filename)
            
            # Копирование файла
            shutil.copy2(source_path, dest_path)
            
            # Сохранение в БД
            self.cursor.execute(
                "INSERT INTO user_videos (user_id, title, file_path, original_filename) VALUES (?, ?, ?, ?)",
                (user_id, title, dest_path, os.path.basename(source_path))
            )
            self.conn.commit()
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при добавлении видео: {e}")
            return False
    
    def get_video_data(self, video_id):
        """Получение данных видео по ID"""
        try:
            # Проверяем существование колонок
            self.cursor.execute("PRAGMA table_info(user_videos)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            if 'eye_tracking_data' in columns and 'charts_path' in columns:
                self.cursor.execute(
                    "SELECT id, user_id, title, file_path, eye_tracking_data, charts_path FROM user_videos WHERE id=?",
                    (video_id,)
                )
            else:
                self.cursor.execute(
                    "SELECT id, user_id, title, file_path FROM user_videos WHERE id=?",
                    (video_id,)
                )
            result = self.cursor.fetchone()
            
            # Если eye_tracking_data отсутствует, возвращаем None вместо пустой строки
            if result and len(result) >= 4:
                # Преобразуем результат в список, чтобы можно было изменять
                result_list = list(result)
                # Если eye_tracking_data нет в результате, добавляем None
                while len(result_list) < 6:
                    result_list.append(None)
                return tuple(result_list)
            
            return result
            
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке данных видео: {e}")
            return None
    
    def delete_user_video(self, video_id, file_path):
        """Удаление видео"""
        try:
            # Получаем информацию о видео для удаления графиков
            self.cursor.execute("SELECT user_id, charts_path FROM user_videos WHERE id=?", (video_id,))
            result = self.cursor.fetchone()
            
            if result and len(result) > 1 and result[1]:
                user_id, charts_path_json = result
                
                # Удаление файлов графиков
                if charts_path_json:
                    try:
                        import json
                        charts_paths = json.loads(charts_path_json)
                        for path in charts_paths.values():
                            if path and os.path.exists(path):
                                os.remove(path)
                    except:
                        pass
            
            # Удаление записи из БД
            self.cursor.execute("DELETE FROM user_videos WHERE id=?", (video_id,))
            self.conn.commit()
            
            # Удаление файла видео с диска
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при удалении видео: {e}")
            return False
    
    def update_video_processing_data(self, video_id, eye_tracking_data, charts_path):
        """Обновление данных обработки видео"""
        try:
            # Сначала проверяем, существует ли колонка
            self.cursor.execute("PRAGMA table_info(user_videos)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            if 'eye_tracking_data' not in columns:
                self._add_missing_columns()
            
            self.cursor.execute('''
                UPDATE user_videos 
                SET eye_tracking_data = ?, 
                    charts_path = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (eye_tracking_data, charts_path, video_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении данных видео: {e}")
            return False
    
    def get_processed_videos(self, user_id=None):
        """Получение списка обработанных видео"""
        try:
            # Проверяем существование колонки
            self.cursor.execute("PRAGMA table_info(user_videos)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            if 'eye_tracking_data' not in columns:
                return []
            
            if user_id:
                self.cursor.execute(
                    "SELECT id, title, file_path, processed_at FROM user_videos WHERE user_id=? AND eye_tracking_data IS NOT NULL ORDER BY processed_at DESC",
                    (user_id,)
                )
            else:
                self.cursor.execute(
                    "SELECT id, title, file_path, processed_at FROM user_videos WHERE eye_tracking_data IS NOT NULL ORDER BY processed_at DESC"
                )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке обработанных видео: {e}")
            return []
    
    def get_video_stats(self, video_id):
        """Получение статистики по видео"""
        try:
            self.cursor.execute(
                "SELECT eye_tracking_data FROM user_videos WHERE id=?",
                (video_id,)
            )
            result = self.cursor.fetchone()
            if result and result[0]:
                import json
                data = json.loads(result[0])
                
                stats = {}
                for eye in ['left_eye', 'right_eye']:
                    eye_data = data.get(eye, {})
                    if eye_data.get('timestamps'):
                        stats[eye] = {
                            'duration': eye_data['timestamps'][-1] - eye_data['timestamps'][0],
                            'frames': len(eye_data['timestamps']),
                            'avg_diameter': sum(eye_data.get('diameter', [0])) / len(eye_data.get('diameter', [1])),
                            'avg_speed': sum(eye_data.get('speed', [0])) / len(eye_data.get('speed', [1]))
                        }
                return stats
            return None
        except Exception as e:
            print(f"Ошибка при получении статистики: {e}")
            return None
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()