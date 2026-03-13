# chart_utils.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import math
import numpy as np


# speed_data — список значений скорости (пиксели/секунду) для каждого кадра
# timestamps — список временных меток (секунды) для каждого кадра.
# velocity_threshold — порог скорости, выше которого движение считается саккадой (по умолчанию 100)
def calculate_saccade_metrics(speed_data, timestamps, velocity_threshold=100):
    """Расчет метрик саккад"""
    # Саккада определяется как интервал времени, в течение которого скорость
    #  движения глаза устойчиво превышает заданный порог. Функция выделяет все
    # такие интервалы и вычисляет:
    #       - общее количество саккад;
    #       - среднюю пиковую скорость (максимальная скорость в каждой саккаде);
    #       - латентность первой саккады (время от начала записи до её начала).

    # Если данных меньше 10, возвращаются нулевые значения (недостаточно для статистики)
    if len(speed_data) < 10:
        return {
            'saccade_count': 0,
            'avg_peak_velocity': 0,
            'avg_latency': 0
        }

    is_saccade = False          # флаг, находимся ли мы внутри саккады
    saccade_start = 0            # индекс начала текущей саккады
    saccades = []                # список обнаруженных саккад

    # Проходим по всем значениям скорости
    for i, speed in enumerate(speed_data):

        # Если скорость превышает порог и мы ещё не в саккаде – начинаем саккаду
        if not is_saccade and speed > velocity_threshold:
            is_saccade = True
            saccade_start = i  

        # Если скорость упала ниже порога и мы были в саккаде – завершаем саккаду
        elif is_saccade and speed <= velocity_threshold:
            is_saccade = False

            # Минимальная длительность саккады – 3 кадра (чтобы отсечь шум)
            if i - saccade_start > 2:

                # Берём скорости на интервале саккады
                saccade_speeds = speed_data[saccade_start:i]
                # Вычисляем пиковую скорость (максимум)
                peak_vel = max(saccade_speeds)
                # Длительность в секундах (разница временных меток последнего и первого кадра)
                duration = timestamps[i-1] - timestamps[saccade_start] if timestamps else 0
                
                saccades.append({
                    'start_idx': saccade_start,
                    'end_idx': i,
                    'peak_velocity': peak_vel,
                    'duration': duration
                })

    if saccades:
        # Средняя пиковая скорость по всем саккадам
        avg_peak = sum(s['peak_velocity'] for s in saccades) / len(saccades)
        
        # Латентность первой саккады – время начала самой первой саккады
        first_saccade_latency = timestamps[saccades[0]['start_idx']] if saccades and timestamps else 0
        
        return {
            'saccade_count': len(saccades),
            'avg_peak_velocity': avg_peak,
            'avg_latency': first_saccade_latency
        }

    return {
        'saccade_count': 0,
        'avg_peak_velocity': 0,
        'avg_latency': 0
    }



# x_data (list[float]): координаты X для каждого кадра
# y_data (list[float]): координаты Y для каждого кадра
# timestamps (list[float]): временные метки
# max_deviation (float): максимально допустимый разброс (дисперсия) для фиксации
# min_duration (float): минимальная длительность фиксации в секундах
def calculate_fixation_metrics(x_data, y_data, timestamps,
                                max_deviation=20, min_duration=0.1):
    """Расчет метрик фиксаций"""
    # Фиксация определяется как интервал, в течение которого координаты взгляда
    # имеют малый разброс (дисперсию). Используется скользящее окно размером 5 кадров.
    # Для каждого окна вычисляется дисперсия по X и Y, затем общая дисперсия
    # (евклидова норма). Если дисперсия ниже порога и держится несколько окон,
    # фиксируется фиксация.

    # Если данных меньше 5, возвращаются нулевые значения (недостаточно для статистики)
    if len(x_data) < 5:
        return {
            'fixation_count': 0,
            'avg_fixation_duration': 0
        }

    fixations = []            # список обнаруженных фиксаций
    in_fixation = False       # флаг, находимся ли мы внутри фиксации
    fixation_start = 0         # индекс начала текущей фиксации
    window_size = 5            # размер окна для скользящего анализа

    # Проходим по данным с окном размером window_size
    for i in range(len(x_data) - window_size):
        # Берём окно данных по X и Y
        window_x = x_data[i:i+window_size]
        window_y = y_data[i:i+window_size]

        # Вычисляем дисперсию в окне (мера разброса)
        var_x = np.var(window_x) if len(window_x) > 1 else 0
        var_y = np.var(window_y) if len(window_y) > 1 else 0

        # Объединяем дисперсии: корень из суммы квадратов (евклидова норма)
        dispersion = math.sqrt(var_x**2 + var_y**2)

        # Если дисперсия меньше порога и мы не в фиксации – начинаем фиксацию
        if dispersion < max_deviation and not in_fixation:
            in_fixation = True
            fixation_start = i

        # Если дисперсия превысила порог и мы были в фиксации – завершаем фиксацию
        elif dispersion >= max_deviation and in_fixation:
            in_fixation = False

            # Проверяем, что фиксация длилась хотя бы 2 кадра
            if i - fixation_start > 1:
                # Вычисляем длительность в секундах (последний кадр фиксации – i-1)
                duration = timestamps[i-1] - timestamps[fixation_start]

                # Если длительность не меньше минимальной – сохраняем фиксацию
                if duration >= min_duration:
                    fixations.append({
                        'start': fixation_start,
                        'end': i-1,
                        'duration': duration
                    })

    if fixations:
        # Средняя длительность всех фиксаций
        avg_duration = sum(f['duration'] for f in fixations) / len(fixations)
        return {
            'fixation_count': len(fixations),
            'avg_fixation_duration': avg_duration
        }
    
    # Если фиксации не обнаружены, возвращаем нули
    return {
        'fixation_count': 0,
        'avg_fixation_duration': 0
    }



# x_data (list[float]): координаты X для каждого кадра (можно использовать любую ось)
# timestamps (list[float]): временные метки
# target_speed (float): предполагаемая скорость движения цели (пиксели/секунду)
def calculate_smooth_pursuit_gain(x_data, timestamps, target_speed=100):
    """Расчет усиления следящих движений"""

    # Усиление определяется как отношение средней скорости движения глаза
    # к скорости движения цели. В реальных экспериментах скорость цели известна,
    # здесь используется приблизительное значение target_speed.

    if len(x_data) < 10:
        return 0

    eye_speeds = [] # список мгновенных скоростей глаза по оси X

    # Для каждого интервала между кадрами вычисляем скорость
    for i in range(1, len(x_data)):
        dx = x_data[i] - x_data[i-1]                     # перемещение по X
        dt = timestamps[i] - timestamps[i-1]             # временной интервал
        # Защита от нулевого или отрицательного dt (если метки одинаковы, берём 0.033 с)
        if dt <= 0:
            dt = 0.033
        speed = abs(dx) / dt                              # модуль скорости
        eye_speeds.append(speed)

    if not eye_speeds:
        return 0

    # Средняя скорость глаза за всю запись
    avg_eye_speed = sum(eye_speeds) / len(eye_speeds)

    # Вычисляем gain = скорость глаза / скорость цели
    if target_speed > 0:
        gain = avg_eye_speed / target_speed
    else:
        gain = 0

    # Ограничиваем сверху значением 2.0 (чтобы избежать выбросов)
    return min(gain, 2.0)


def show_charts_window(parent, video_title, data, source_message=""):
    """
    Отображает окно с графиками и статистикой.
    parent - родительское окно Tkinter
    video_title - название видео
    data - словарь с данными айтрекинга вида:
           {'left_eye': {'x': [...], 'y': [...], 'diameter': [...], 'speed': [...], 'timestamps': [...]},
            'right_eye': {...}}
    source_message - доп. информация (например, "Предпросмотр")
    """
    charts_window = tk.Toplevel(parent)
    charts_window.title(f"Графики и статистика - {video_title} {source_message}")
    charts_window.geometry("1400x900")

    notebook = ttk.Notebook(charts_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Извлекаем данные для левого и правого глаза с защитой от отсутствия ключей
    left_eye = data.get('left_eye', {})
    right_eye = data.get('right_eye', {})

    left_x = left_eye.get('x', [])
    left_y = left_eye.get('y', [])
    left_timestamps = left_eye.get('timestamps', [])
    left_diameter = left_eye.get('diameter', [])
    left_speed = left_eye.get('speed', [])

    right_x = right_eye.get('x', [])
    right_y = right_eye.get('y', [])
    right_timestamps = right_eye.get('timestamps', [])
    right_diameter = right_eye.get('diameter', [])
    right_speed = right_eye.get('speed', [])

    # ===== ВКЛАДКА 1: ДАННЫЕ И СТАТИСТИКА =====
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Данные и статистика")

    text_frame = ttk.Frame(data_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text_widget = tk.Text(text_frame, font=("Courier", 10))
    scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)

    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_widget.insert(tk.END, f"Данные айтрекинга для видео: {video_title}\n\n")

    for eye_name, eye_data in [('left_eye', left_eye), ('right_eye', right_eye)]:
        timestamps = eye_data.get('timestamps', [])
        x_coords = eye_data.get('x', [])
        y_coords = eye_data.get('y', [])
        diameters = eye_data.get('diameter', [])
        speeds = eye_data.get('speed', [])

        saccade_metrics = calculate_saccade_metrics(speeds, timestamps)
        fixation_metrics = calculate_fixation_metrics(x_coords, y_coords, timestamps)
        pursuit_gain = calculate_smooth_pursuit_gain(x_coords, timestamps)

        text_widget.insert(tk.END, f"\n{'='*50}\n")
        text_widget.insert(tk.END, f"{eye_name.upper()}\n")
        text_widget.insert(tk.END, f"{'='*50}\n\n")

        text_widget.insert(tk.END, "ОСНОВНЫЕ ПОКАЗАНИЯ:\n")
        text_widget.insert(tk.END, f"{'-'*40}\n")
        text_widget.insert(tk.END, f"Количество кадров: {len(timestamps)}\n")
        if timestamps:
            text_widget.insert(tk.END, f"Диапазон времени: {timestamps[0]:.2f} - {timestamps[-1]:.2f} сек\n")
            text_widget.insert(tk.END, f"Длительность записи: {timestamps[-1] - timestamps[0]:.2f} сек\n")
        if diameters:
            text_widget.insert(tk.END, f"Средний диаметр зрачка: {sum(diameters)/len(diameters):.2f} px\n")
            text_widget.insert(tk.END, f"Мин диаметр: {min(diameters):.2f} px\n")
            text_widget.insert(tk.END, f"Макс диаметр: {max(diameters):.2f} px\n")
        if speeds:
            text_widget.insert(tk.END, f"Средняя скорость: {sum(speeds)/len(speeds):.2f} px/s\n")
            text_widget.insert(tk.END, f"Макс скорость: {max(speeds):.2f} px/s\n")

        text_widget.insert(tk.END, f"\nМЕТРИКИ САККАД:\n")
        text_widget.insert(tk.END, f"{'-'*40}\n")
        text_widget.insert(tk.END, f"Количество саккад: {saccade_metrics['saccade_count']}\n")
        if saccade_metrics['avg_peak_velocity'] > 0:
            text_widget.insert(tk.END, f"Средняя пиковая скорость: {saccade_metrics['avg_peak_velocity']:.2f} px/s\n")
            text_widget.insert(tk.END, f"Латентность первой саккады: {saccade_metrics['avg_latency']*1000:.2f} мс\n")
        else:
            text_widget.insert(tk.END, f"Средняя пиковая скорость: 0 px/s\n")
            text_widget.insert(tk.END, f"Латентность первой саккады: 0 мс\n")

        text_widget.insert(tk.END, f"\nМЕТРИКИ ФИКСАЦИЙ:\n")
        text_widget.insert(tk.END, f"{'-'*40}\n")
        text_widget.insert(tk.END, f"Количество фиксаций: {fixation_metrics['fixation_count']}\n")
        if fixation_metrics['avg_fixation_duration'] > 0:
            text_widget.insert(tk.END, f"Средняя продолжительность фиксации: {fixation_metrics['avg_fixation_duration']*1000:.2f} мс\n")
        else:
            text_widget.insert(tk.END, f"Средняя продолжительность фиксации: 0 мс\n")

        text_widget.insert(tk.END, f"\nСЛЕДЯЩИЕ ДВИЖЕНИЯ:\n")
        text_widget.insert(tk.END, f"{'-'*40}\n")
        text_widget.insert(tk.END, f"Усиление следящих движений: {pursuit_gain:.2f}\n\n")

    text_widget.config(state=tk.DISABLED)

    # ===== ВКЛАДКА 2: ТРАЕКТОРИИ ОБОИХ ГЛАЗ (СРАВНЕНИЕ) =====
    if left_x or right_x:
        traj_comparison_frame = ttk.Frame(notebook)
        notebook.add(traj_comparison_frame, text="Траектории глаз (сравнение)")

        fig_traj, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

        if left_x:
            ax_left.plot(left_x, left_y, 'r-', alpha=0.7, linewidth=1.5)
            ax_left.set_title('Левый глаз', fontsize=12, fontweight='bold')
        else:
            ax_left.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax_left.transAxes)

        ax_left.set_xlabel('X (пиксели)')
        ax_left.set_ylabel('Y (пиксели)')
        ax_left.invert_yaxis()
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        if right_x:
            ax_right.plot(right_x, right_y, 'g-', alpha=0.7, linewidth=1.5)
            ax_right.set_title('Правый глаз', fontsize=12, fontweight='bold')
        else:
            ax_right.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax_right.transAxes)

        ax_right.set_xlabel('X (пиксели)')
        ax_right.set_ylabel('Y (пиксели)')
        ax_right.invert_yaxis()
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

        plt.tight_layout()

        canvas_traj = FigureCanvasTkAgg(fig_traj, traj_comparison_frame)
        canvas_traj.draw()
        canvas_traj.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===== ВКЛАДКА 3: ПАРАМЕТРЫ ОБОИХ ГЛАЗ (СРАВНЕНИЕ) =====
    if left_timestamps or right_timestamps:
        params_comparison_frame = ttk.Frame(notebook)
        notebook.add(params_comparison_frame, text="Параметры глаз (сравнение)")

        fig_params = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig_params, hspace=0.3, wspace=0.3)

        ax_diameter = fig_params.add_subplot(gs[0, :])
        ax_diameter.set_title('Сравнение диаметра зрачков', fontsize=12, fontweight='bold')

        if left_timestamps and left_diameter:
            ax_diameter.plot(left_timestamps, left_diameter,
                            'r-', linewidth=1.5, label='Левый глаз', alpha=0.8)
        if right_timestamps and right_diameter:
            ax_diameter.plot(right_timestamps, right_diameter,
                            'g-', linewidth=1.5, label='Правый глаз', alpha=0.8)

        ax_diameter.set_xlabel('Время (сек)')
        ax_diameter.set_ylabel('Диаметр (пиксели)')
        ax_diameter.grid(True, alpha=0.3)
        ax_diameter.legend()

        ax_speed = fig_params.add_subplot(gs[1, :])
        ax_speed.set_title('Сравнение скорости движения', fontsize=12, fontweight='bold')

        if left_timestamps and left_speed:
            ax_speed.plot(left_timestamps, left_speed,
                        'r-', linewidth=1.5, label='Левый глаз', alpha=0.8)
        if right_timestamps and right_speed:
            ax_speed.plot(right_timestamps, right_speed,
                        'g-', linewidth=1.5, label='Правый глаз', alpha=0.8)

        ax_speed.set_xlabel('Время (сек)')
        ax_speed.set_ylabel('Скорость (пиксели/сек)')
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend()

        plt.tight_layout()

        canvas_params = FigureCanvasTkAgg(fig_params, params_comparison_frame)
        canvas_params.draw()
        canvas_params.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===== ВКЛАДКА 4: ЛЕВЫЙ ГЛАЗ (ДЕТАЛЬНО) =====
    if left_timestamps:
        left_detailed_frame = ttk.Frame(notebook)
        notebook.add(left_detailed_frame, text="Левый глаз (детально)")

        fig_left, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        if left_x:
            ax1.plot(left_x, left_y, 'r-', alpha=0.7, linewidth=1)
            ax1.set_title('Траектория левого глаза')
            ax1.set_xlabel('X (пиксели)')
            ax1.set_ylabel('Y (пиксели)')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
        else:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax1.transAxes)

        ax2.plot(left_timestamps, left_diameter, 'b-', linewidth=1.5)
        ax2.set_title('Диаметр зрачка левого глаза')
        ax2.set_xlabel('Время (сек)')
        ax2.set_ylabel('Диаметр (пиксели)')
        ax2.grid(True, alpha=0.3)

        ax3.plot(left_timestamps, left_speed, 'r-', linewidth=1.5)
        ax3.set_title('Скорость движения левого глаза')
        ax3.set_xlabel('Время (сек)')
        ax3.set_ylabel('Скорость (пиксели/сек)')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas_left = FigureCanvasTkAgg(fig_left, left_detailed_frame)
        canvas_left.draw()
        canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ===== ВКЛАДКА 5: ПРАВЫЙ ГЛАЗ (ДЕТАЛЬНО) =====
    if right_timestamps:
        right_detailed_frame = ttk.Frame(notebook)
        notebook.add(right_detailed_frame, text="Правый глаз (детально)")

        fig_right, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        if right_x:
            ax1.plot(right_x, right_y, 'g-', alpha=0.7, linewidth=1)
            ax1.set_title('Траектория правого глаза')
            ax1.set_xlabel('X (пиксели)')
            ax1.set_ylabel('Y (пиксели)')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
        else:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax1.transAxes)

        ax2.plot(right_timestamps, right_diameter, 'b-', linewidth=1.5)
        ax2.set_title('Диаметр зрачка правого глаза')
        ax2.set_xlabel('Время (сек)')
        ax2.set_ylabel('Диаметр (пиксели)')
        ax2.grid(True, alpha=0.3)

        ax3.plot(right_timestamps, right_speed, 'r-', linewidth=1.5)
        ax3.set_title('Скорость движения правого глаза')
        ax3.set_xlabel('Время (сек)')
        ax3.set_ylabel('Скорость (пиксели/сек)')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas_right = FigureCanvasTkAgg(fig_right, right_detailed_frame)
        canvas_right.draw()
        canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)