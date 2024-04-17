import numpy as np
import cv2


class Pupil(object):
    """
   Этот класс определяет радужную оболочку глаза и оценивает
    положение зрачка
    """

    def __init__(self, eye_frame, threshold):
        # Инициализация объекта. eye_frame - изображение глаза, threshold - порог для обработки.
        self.iris_frame = None # Изображение радужки будет храниться здесь.
        self.threshold = threshold # Порог для применения в обработке изображения глаза.
        self.x = None # Переменная для хранения координаты x (горизонтальная) глаза.
        self.y = None # Переменная для хранения координаты y (вертикальная) глаза.

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Выполняет операции на оправе глаза для изоляции радужной оболочки

        Аргументы:
            eye_frame (numpy.ndarray): кадр, содержащий глаз и ничего больше
            threshold (int): Пороговое значение, используемое для бинаризации кадра глаза

        Возвращается:
            Рамка с единственным элементом, представляющим радужную оболочку
        """
        kernel = np.ones((3, 3), np.uint8)
        # Фильтрация изображения с использованием билатерального фильтра.
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15) 
        # Эрозия изображения с использованием ядра 3x3.
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        # Применение бинаризации с порогом кадра глаза.
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Обнаруживает радужную оболочку и оценивает положение радужной оболочки путем
        вычисления центра тяжести.

        Аргументы:
            eye_frame (numpy.ndarray): кадр, содержащий глаз и ничего больше
        """
        # Использует метод image_processing для обработки кадра глаза и изоляции радужной оболочки.
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        # Находит контуры на изображении радужки.
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
             # Вычисляет центр тяжести для второго по размеру контура (радужки).
        
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            # Обработка исключений, возникающих при отсутствии контуров или делении на ноль.
            pass
