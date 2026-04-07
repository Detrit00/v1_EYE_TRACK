# chart_utils.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import math
import numpy as np

def calculate_saccade_metrics(speed_data, timestamps,
                              velocity_threshold=100,
                              min_saccade_duration=0.01):
    """
    Расчёт метрик саккад по I‑VT.
    Саккада – интервал, где speed > velocity_threshold,
    длительностью не менее min_saccade_duration.
    """
    if len(speed_data) < 10:
        return {'saccade_count': 0, 'avg_peak_velocity': 0, 'avg_latency': 0}

    is_saccade = False
    saccade_start = 0
    saccades = []

    for i, speed in enumerate(speed_data):
        if not is_saccade and speed > velocity_threshold:
            is_saccade = True
            saccade_start = i
        elif is_saccade and speed <= velocity_threshold:
            is_saccade = False
            duration = timestamps[i-1] - timestamps[saccade_start]
            if duration >= min_saccade_duration:
                saccade_speeds = speed_data[saccade_start:i]
                peak_vel = max(saccade_speeds)
                saccades.append({
                    'start_idx': saccade_start,
                    'end_idx': i-1,
                    'peak_velocity': peak_vel,
                    'duration': duration
                })
    # Если последняя саккада доходит до конца
    if is_saccade:
        i = len(speed_data) - 1
        duration = timestamps[i] - timestamps[saccade_start]
        if duration >= min_saccade_duration:
            saccade_speeds = speed_data[saccade_start:i+1]
            peak_vel = max(saccade_speeds)
            saccades.append({
                'start_idx': saccade_start,
                'end_idx': i,
                'peak_velocity': peak_vel,
                'duration': duration
            })

    if saccades:
        avg_peak = sum(s['peak_velocity'] for s in saccades) / len(saccades)
        first_latency = timestamps[saccades[0]['start_idx']]
        return {
            'saccade_count': len(saccades),
            'avg_peak_velocity': avg_peak,
            'avg_latency': first_latency
        }
    return {'saccade_count': 0, 'avg_peak_velocity': 0, 'avg_latency': 0}


def calculate_fixation_metrics(speed_data, timestamps,
                               velocity_threshold=100,
                               min_fixation_duration=0.04):
    """
    Расчёт метрик фиксаций по I‑VT.
    Фиксация – интервал, где speed <= velocity_threshold,
    длительностью не менее min_fixation_duration.
    """
    if len(speed_data) < 10:
        return {'fixation_count': 0, 'avg_fixation_duration': 0}

    is_fixation = False
    fixation_start = 0
    fixations = []

    for i, speed in enumerate(speed_data):
        if not is_fixation and speed <= velocity_threshold:
            is_fixation = True
            fixation_start = i
        elif is_fixation and speed > velocity_threshold:
            is_fixation = False
            duration = timestamps[i-1] - timestamps[fixation_start]
            if duration >= min_fixation_duration:
                fixations.append({
                    'start_idx': fixation_start,
                    'end_idx': i-1,
                    'duration': duration
                })
    # Если последняя фиксация доходит до конца
    if is_fixation:
        i = len(speed_data) - 1
        duration = timestamps[i] - timestamps[fixation_start]
        if duration >= min_fixation_duration:
            fixations.append({
                'start_idx': fixation_start,
                'end_idx': i,
                'duration': duration
            })

    if fixations:
        avg_duration = sum(f['duration'] for f in fixations) / len(fixations)
        return {
            'fixation_count': len(fixations),
            'avg_fixation_duration': avg_duration
        }
    return {'fixation_count': 0, 'avg_fixation_duration': 0}


def show_charts_window(parent, video_title, data,
                       source_message="",
                       velocity_threshold=100,
                       min_fixation_duration=0.04,
                       min_saccade_duration=0.01):
    """
    Отображает окно с графиками и статистикой, используя переданные параметры.
    """
    charts_window = tk.Toplevel(parent)
    charts_window.title(f"Графики и статистика - {video_title} {source_message}")
    charts_window.geometry("1400x900")

    notebook = ttk.Notebook(charts_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

    text_widget.insert(tk.END, f"Данные айтрекинга для видео: {video_title}\n")
    text_widget.insert(tk.END, f"Параметры классификации: порог скорости = {velocity_threshold:.0f} px/s, "
                               f"мин. длительность фиксации = {min_fixation_duration*1000:.0f} мс, "
                               f"мин. длительность саккады = {min_saccade_duration*1000:.0f} мс\n\n")

    for eye_name, eye_data in [('left_eye', left_eye), ('right_eye', right_eye)]:
        timestamps = eye_data.get('timestamps', [])
        x_coords = eye_data.get('x', [])
        y_coords = eye_data.get('y', [])
        diameters = eye_data.get('diameter', [])
        speeds = eye_data.get('speed', [])

        saccade_metrics = calculate_saccade_metrics(speeds, timestamps,
                                                    velocity_threshold,
                                                    min_saccade_duration)
        fixation_metrics = calculate_fixation_metrics(speeds, timestamps,
                                                      velocity_threshold,
                                                      min_fixation_duration)

        text_widget.insert(tk.END, f"\n{'='*50}\n")
        if eye_name == 'left_eye':
            eye_title = 'ЛЕВЫЙ ГЛАЗ'
        else:
            eye_title = 'ПРАВЫЙ ГЛАЗ'
        text_widget.insert(tk.END, f"{eye_title}\n")
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