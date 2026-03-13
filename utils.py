# utils.py
import math

# Константы для индексов глаз MediaPipe
# Левый глаз (с точки зрения наблюдателя) - правый глаз человека
LEFT_EYE = {
    'center': 468,  # центр левого глаза
    'right': 469,   # правая точка левого глаза
    'left': 471     # левая точка левого глаза
}

# Правый глаз (с точки зрения наблюдателя) - левый глаз человека
RIGHT_EYE = {
    'center': 473,  # центр правого глаза
    'right': 474,   # правая точка правого глаза
    'left': 476     # левая точка правого глаза
}

# Цвета для отрисовки
COLORS = {
    'left_eye': (0, 0, 255),    # красный
    'right_eye': (0, 255, 0),   # зеленый
    'text': (255, 255, 255)     # белый
}

def calculate_distance(p1, p2, img_w, img_h):
    """Считает расстояние между двумя точками"""
    x1, y1 = p1.x * img_w, p1.y * img_h
    x2, y2 = p2.x * img_w, p2.y * img_h
    return math.hypot(x2 - x1, y2 - y1)

def center_window(window, width, height):
    """Центрирует окно на экране"""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f'{width}x{height}+{x}+{y}')