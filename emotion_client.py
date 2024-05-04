import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtGui import QPainter, QColor
import torch
from torchvision import transforms

from server.emotion.clean.em_pytorch import SimpleCNN

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    print(f"Connected to server")
    model = SimpleCNN()  # Make sure you load your model correctly before this line
    model.load_state_dict(torch.load(r'C:\Users\bokar\Documents\EyeGazeFrameworkDemo\resources\emotion\model.pth'))
    model.eval()
    emotion_dict = {0: "Злой", 1: "Чувствующий отвращение", 2: "Напуганный", 3: "Счастливый", 4: "Нейтральное состояние", 5: "Огорченный", 6: "Удивленный"}
    cascade_path = r'C:\Users\bokar\Documents\EyeGazeFrameworkDemo\resources\emotion\haarcascade_frontalface_default.xml'
    main_window = VideoWindow(model, emotion_dict, cascade_path)
    main_window.show()
    sys.exit(app.exec_())
