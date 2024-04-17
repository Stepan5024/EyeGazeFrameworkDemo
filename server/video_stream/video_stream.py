import cv2
import numpy as np
import time

class VideoStream:
    def __init__(self, source=0, fps=None):
        # Инициализируем захват видео с указанного источника
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError("Не удается открыть видеопоток")
        self.fps = fps
        # Если fps установлено, вычисляем интервал времени между кадрами в секундах
        if fps is not None:
            self.frame_interval = 1.0 / fps
        else:
            self.frame_interval = 0
        # Получаем разрешение видео
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __del__(self):
        # Освобождаем ресурсы
        if self.cap.isOpened():
            self.cap.release()
    
    def read_frame(self):
        if self.fps is not None:
            # Запоминаем время начала для контроля частоты кадров
            start_time = time.time()
        # Чтение одного кадра
        ret, frame = self.cap.read()
        if not ret:
            print("Не удается прочитать видеокадр")  # Логирование ошибки
            return None  # Возврат None в случае ошибки
            #raise IOError("Не удается прочитать видеокадр")
        if self.fps is not None:
            # Вычисляем время, которое необходимо подождать до следующего кадра
            elapsed_time = time.time() - start_time
            time_to_wait = self.frame_interval - elapsed_time
            if time_to_wait > 0:
                print(f"wait {time_to_wait}")
                time.sleep(time_to_wait)
        return frame

    def convert_color(self, frame, to_rgb=True):
        # Преобразование цвета кадра
        if to_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def preprocess_for_resnet50(self, frame):
        # Подготовка изображения для ResNet50
        frame_rgb = self.convert_color(frame, to_rgb=True)
        # Масштабируем пиксели к диапазону [0, 1], а затем нормализуем
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_preprocessed = np.expand_dims(frame_resized, axis=0) / 255.0
        return frame_preprocessed

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
