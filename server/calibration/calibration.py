from __future__ import division
import cv2
from server.models.pupil import Pupil


class Calibration(object):
    """
    Этот класс калибрует алгоритм детекции зрачка, находя
    оптимальное значение порога бинаризации для человека и веб-камеры.
    """


    def __init__(self):
        """
        Инициализирует объект Calibration

        Атрибуты:
            nb_frames (int): Количество кадров, используемых для калибровки
            thresholds_left (list): Список пороговых значений для левого глаза
            thresholds_right (list): Список пороговых значений для правого глаза
        """
        self.nb_frames = 20 # Количество кадров, используемых для калибровки
        self.thresholds_left = []  # Список пороговых значений для левого глаза
        self.thresholds_right = [] # Список пороговых значений для правого глаза

    def is_complete(self):
        """Возвращает True, если калибровка завершена"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Возвращает значение порога для указанного глаза.

    Аргументы:
        side: Указывает, является ли это левым глазом (0) или правым глазом (1)
        """
        if side == 0:
            # Возвращает среднее значение порога для левого глаза.
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
              # Возвращает среднее значение порога для правого глаза.
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Возвращает процент пространства, занимаемого радужкой на поверхности глаза.

        Аргументы:
        frame (numpy.ndarray): Бинаризованный кадр с радужкой глаза
        """
        # Обрезает рамку, чтобы исключить возможные некорректные краевые пиксели.
        frame = frame[5:-5, 5:-5]
        # Получение высоты и ширины обрезанной рамки.
        height, width = frame.shape[:2]
         # Вычисление общего числа пикселей и числа черных пикселей в рамке.
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        # Возвращает процент черных пикселей от общего числа пикселей.
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Вычисляет оптимальное пороговое значение для бинаризации
        кадра для данного глаза.

        Аргумент:
            eye_frame (numpy.ndarray): Кадр глаза для анализа
        """
         # Средний размер радужки глаза (в процентах).
        average_iris_size = 0.48
         # Словарь для хранения результатов испытаний с разными пороговыми значениями.
    
        trials = {}
        # Итерации по различным пороговым значениям от 5 до 100 с шагом 5.
        for threshold in range(5, 100, 5):
             # Обработка изображения глаза с использованием текущего порогового значения.
        
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            # Запись размера радужки для данного порогового значения.
        
            trials[threshold] = Calibration.iris_size(iris_frame)
        # Выбор наилучшего порогового значения на основе минимального отклонения от среднего размера радужки.
    
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
          # Возвращает найденное оптимальное пороговое значение.
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Улучшает калибровку, учитывая предоставленное изображение.

        Аргументы:
            eye_frame (numpy.ndarray): Кадр глаза
            side: Указывает, является ли это левым глазом (0) или правым глазом (1)
        """
        # Находит оптимальное пороговое значение для переданного изображения глаза.
        threshold = self.find_best_threshold(eye_frame)
        # Добавляет оптимальное пороговое значение в соответствующий список.
        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
