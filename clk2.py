from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import sys
import numpy as np
import cv2
import collections
from random import randint

class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.gaze_points = collections.deque(maxlen=64)
        self.monitor_pixels = (1700, 800)  # Оконные размеры для второго окна

    def run(self):
        display = np.zeros((self.monitor_pixels[1], self.monitor_pixels[0], 3), dtype=np.uint8)
        while True:
            point_on_screen = (randint(0, self.monitor_pixels[0]), randint(0, self.monitor_pixels[1]))
            self.gaze_points.appendleft(point_on_screen)
            display.fill(0)
            for idx in range(1, len(self.gaze_points)):
                thickness = round((len(self.gaze_points) - idx) / len(self.gaze_points) * 5) + 1
                cv2.line(display, self.gaze_points[idx - 1], self.gaze_points[idx], (0, 0, 255), thickness)

            rgbImage = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(self.monitor_pixels[0], self.monitor_pixels[1], Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            self.msleep(100)

class DrawingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Drawing Window')
        layout = QVBoxLayout()
        self.label = QLabel(self)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(1700, 800)
        self.showFullScreen()

        self.thread = VideoThread()
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()  # Закрывает окно при нажатии Q или Esc

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(400, 300)
        self.show_button = QPushButton('Show Drawing Window', self)
        self.show_button.clicked.connect(self.showDrawingWindow)
        self.show()

    def showDrawingWindow(self):
        self.drawing_window = DrawingWindow()
        self.drawing_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
