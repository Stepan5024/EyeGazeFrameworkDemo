import logging
import os
import pathlib
import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtGui import QPainter, QColor
import torch
from torchvision import transforms
import yaml

from logger import create_logger
from server.models.emotionModel import EmotionModel

class VideoWindow(QMainWindow):
    def __init__(self, model, emotion_dict, cascade_path):
        super().__init__()
        self.model = model
        self.emotion_dict = emotion_dict
        self.cascade = cv2.CascadeClassifier(cascade_path)

        self.setWindowTitle("Клиент по распознаванию эмоций")
        self.setGeometry(100, 100, 640, 480)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.resize(640, 480)

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Convert the frame to QImage for easier text rendering
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QPainter instance to draw text over the image
        painter = QPainter(qt_image)
        painter.setFont(QFont("Arial", 12))  # Ensure the font is appropriate

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            tensor = transforms.ToTensor()(roi_gray).unsqueeze(0)
            prediction = self.model(tensor)
            emotion = torch.argmax(prediction, dim=1)
            emotion_text = self.emotion_dict[emotion.item()]
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, y, emotion_text)

        painter.end()

        # Display the image with text
        p = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(p.scaled(640, 480, QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()


def resource_path(relative_path) -> str:
    """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
    try:
        # временная папку _MEIPASS для ресурсов
        base_path = sys._MEIPASS
    except Exception:
        # Если приложение запущено из исходного кода, то используется обычный путь
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(f"Connected to server")
    
    model = EmotionModel()
    relative_path_emotion_model: str = os.path.join("resources",  "emotion", "model.pth")
    abs_path: str = resource_path(relative_path_emotion_model)
    model.load_state_dict(torch.load(abs_path))
    model.eval()
    emotion_dict = {0: "Злой", 1: "Чувствующий отвращение", 2: "Напуганный", 3: "Счастливый", 4: "Нейтральное состояние", 5: "Огорченный", 6: "Удивленный"}
    relative_path_haar_model: str = os.path.join("resources", "emotion", "haarcascade_frontalface_default.xml")
    abs_path: str = resource_path(relative_path_haar_model)
    main_window = VideoWindow(model, emotion_dict, abs_path)
    main_window.show()
    sys.exit(app.exec_())
