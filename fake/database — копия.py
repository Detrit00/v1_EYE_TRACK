# database.py
import sqlite3
import os

class EyeDatabase:
    def __init__(self, db_file="eye_data.db"):
        self.db_file = db_file
        self.connection = None
        self.cursor = None
        self.init_db()
    
    def init_db(self):
        """Инициализация базы данных"""
        try:
            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.cursor = self.connection.cursor()
            
            # Проверяем существование таблицы
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pupil_log'")
            table_exists = self.cursor.fetchone()
            
            if table_exists:
                # Проверяем структуру существующей таблицы
                self.cursor.execute("PRAGMA table_info(pupil_log)")
                columns = [column[1] for column in self.cursor.fetchall()]
                
                # Если таблица старая, пересоздаем её
                if 'left_x' not in columns:
                    print("Обновление структуры базы данных...")
                    self._upgrade_table()
            else:
                # Создаем новую таблицу
                self._create_table()
                print(f"Новая база данных '{self.db_file}' создана")
            
            self.connection.commit()
            print("База данных подключена")
            
        except sqlite3.Error as e:
            print(f"Ошибка базы данных: {e}")
            self._create_fresh_database()
    
    def _create_table(self):
        """Создание новой таблицы"""
        query = """
        CREATE TABLE pupil_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            left_x INTEGER,
            left_y INTEGER,
            left_diameter_px REAL,
            left_speed_px_sec REAL,
            right_x INTEGER,
            right_y INTEGER,
            right_diameter_px REAL,
            right_speed_px_sec REAL
        )
        """
        self.cursor.execute(query)
    
    def _upgrade_table(self):
        """Обновление структуры таблицы"""
        # Переименовываем старую таблицу
        self.cursor.execute("ALTER TABLE pupil_log RENAME TO pupil_log_old")
        
        # Создаем новую таблицу
        self._create_table()
        
        # Копируем данные из старой таблицы (если есть)
        try:
            self.cursor.execute("""
                INSERT INTO pupil_log (timestamp, left_x, left_y, left_diameter_px, left_speed_px_sec)
                SELECT timestamp, pos_x, pos_y, diameter_px, speed_px_sec FROM pupil_log_old
            """)
        except:
            print("Старые данные не удалось скопировать")
        
        # Удаляем старую таблицу
        self.cursor.execute("DROP TABLE IF EXISTS pupil_log_old")
        print("База данных успешно обновлена")
    
    def _create_fresh_database(self):
        """Создание новой базы данных с нуля"""
        try:
            if self.connection:
                self.connection.close()
            
            # Удаляем старый файл БД
            if os.path.exists(self.db_file):
                os.remove(self.db_file)
            
            # Создаем заново
            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.cursor = self.connection.cursor()
            self._create_table()
            self.connection.commit()
            print(f"Новая база данных создана после ошибки")
        except Exception as e2:
            print(f"Критическая ошибка базы данных: {e2}")
    
    def save_eye_data(self, t, left_data, right_data):
        """Сохраняет измерения в базу"""
        if self.connection:
            try:
                query = """INSERT INTO pupil_log 
                          (timestamp, left_x, left_y, left_diameter_px, left_speed_px_sec,
                           right_x, right_y, right_diameter_px, right_speed_px_sec) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                self.cursor.execute(query, (
                    t, 
                    left_data['x'], left_data['y'], left_data['diameter'], left_data['speed'],
                    right_data['x'], right_data['y'], right_data['diameter'], right_data['speed']
                ))
                self.connection.commit()
            except sqlite3.Error as e:
                print(f"Ошибка записи: {e}")
    
    def close(self):
        """Закрытие соединения с БД"""
        if self.connection:
            self.connection.close()