import collections
import sys
import socket
import json
import cv2
import numpy as np

import pyautogui as pag
from l2cs.results import GazeResultContainer
from l2cs import render

from PyQt5.QtWidgets import  QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QDesktopWidget, QComboBox
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from random import randint

width, height = pag.size()
detection_is_on = False
client_socket = None

class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.gaze_points = collections.deque(maxlen=64)
        self.monitor_pixels = (1700, 800)  # Оконные размеры для второго окна

    def run(self):
        display = np.zeros((self.monitor_pixels[1], self.monitor_pixels[0], 3), dtype=np.uint8)
        while True:

            if not client_socket:  # Если сокет закрыт или не определен
                print("Соединение не установлено. Попытка переподключения...")
                # Попытка переподключения...
            elif client_socket is not None:
                print('Получение данных от сервера')
                # Получение размера сообщения
                message_length_bytes = client_socket.recv(4)
                if not message_length_bytes:
                    break
                message_length = int.from_bytes(message_length_bytes, 'big')
                # Получение данных от сервера
                data = b''
                while len(data) < message_length:
                    packet = None
                    try:
                        packet = client_socket.recv(message_length - len(data))
                    except OSError as e:
                        print(f"Ошибка получения данных на клиенте: {e}")
                    if not packet:
                        break
                    data += packet
                if data is not None:
                    results = json.loads(data.decode('utf-8'))
                                # Вывод типа переменной results
                    print(f"Тип переменной 'results': {type(results)}")
                    # Проверка, является ли results словарём
                    if isinstance(results, dict):
                        print("Содержимое словарём 'results':")
                        for key, value in results.items():
                            print(f"{key}: ")
                    elif isinstance(results, list):  # Если results - список
                        print("Содержимое список 'results':")
                        #for item in results:
                            #if(item != 'img'):
                            #print(item)
                    else:
                        # Для других типов данных, которые могут быть напечатаны напрямую
                        print(f"Содержимое 'results': {len(results)}")
                    
                    x_value = int(results['x'])
                    y_value = int(results['y'])
                    print(f"x_value {x_value} y_value {y_value}")
                   
            
            point_on_screen = (x_value, y_value)
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
            """if detection_is_on == True:
                #eye_x, eye_y = results['pitch'][0], results['yaw'][0]  # Пример получения координат
                        #print(f"Eye coordinates: {eye_x}, {eye_y}")  # Здесь могут быть реализованы дополнительные действия с полученными координатами
                        # Теперь можно безопасно извлекать данные изображения
                        #img_bytes = bytes.fromhex(results['img'])
                        #img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        #img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        #gaze_result = GazeResultContainer(
                        #pitch=np.array(results['pitch']),
                        #yaw=np.array(results['yaw']),
                        #bboxes=np.array(results['bboxes']),
                        #landmarks=np.array(results['landmarks']),
                        #scores=np.array(results['scores'])
                        #)
                        x_value = results['x']
                        y_value = results['y']
                        print(f" x = {x_value} y = {y_value}")
                        
                        ////
                        img, eye_x, eye_y, h_, w_ = render(img, gaze_result)
                        #print(f'results = {results}')
                        #print(f'img = {img}')
                        eye_x -= w_/2
                        eye_y -= h_/2
                        e_x = width - (width/2 + eye_x * width / w_*4)
                        e_y = eye_y * height / h_*4
                        print(e_x)
                        if 0 < e_x < width and 0 < e_y < height :
                            pag.moveTo(e_x, e_y)
                        # Отображение кадра в интерфейсе
                        print(f"conveting to PyQt")
                        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgbImage.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(int(width*4/9), int(height*4/9), Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)*///
            else:
                empty_cam_img = QImage('No cam.png').scaled(int(width*4/9), int(height*4/9), Qt.KeepAspectRatio)
                self.changePixmap.emit(empty_cam_img)"""
#Interface
class EyeSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = int(width*1/6)
        self.top = int(height*1/6)
        self.width = int(width*2/3)
        self.height = int(height*2/3)
        self.initUI()
    
    def back_to_main_window(self):
        main_window.show()
        main_window.move(self.geometry().x() - 1, self.geometry().y() - 45)
        self.hide()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(int(width*2/3), int(height*2/3))
        self.setStyleSheet('background-color: #f2f2f2;')

        self.label = QLabel(self)
        self.label.setText("Настройки отслеживания глаз")
        self.label.move(int(self.geometry().width()/2) - 150*2, 15)
        self.label.resize(self.width, 15*4)
        self.label.setStyleSheet("QLabel {font-size: 46px; color: black;}")

        B_back = QPushButton('', self)
        B_back.setToolTip('Назад')
        B_back_w = int(self.width/24)
        B_back_h = int(self.width/24)
        B_back.resize(B_back_w, B_back_h)
        B_back.setIcon(QIcon('Arrow left.png'))
        B_back.setIconSize(QSize(B_back_w - 16, B_back_h - 16))
        B_back.move(15, 15)
        B_back.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')
        B_back.clicked.connect(self.back_to_main_window)

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
        self.resize(width, height)
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
        self.left = int(width*1/6)
        self.top = int(height*1/6)
        self.width = int(width*2/3)
        self.height = int(height*2/3)
        self.videoThread = None  # Инициализация переменной для потока
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot()
    def show_cam(self):
        global detection_is_on
        global client_socket
        if detection_is_on == False:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('127.0.0.1', 9556))
            except socket.error as e:
                print(f"Не удалось подключиться к серверу: {e}")
                exit(1)
            #
            #client_socket.connect(('localhost', 9556))
            
            detection_is_on = True
            print(f'detection_is_on = {detection_is_on}')
            
        else:
            
            detection_is_on = False
            print(f'detection_is_on = {detection_is_on}')
    
    
    def show_eye_settings(self):
        self.eye_settings_window.show()
        self.eye_settings_window.move(self.geometry().x() - 1, self.geometry().y() - 45)
        self.hide()
        global main_window
        main_window = self

    def initUI(self):
        self.setWindowTitle('Eye Tracking Display')
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.resize(int(width*2/3), int(height*2/3))
        self.setStyleSheet('background-color: #f2f2f2;')
        self.label = QLabel(self)
        self.label.move(int(self.width*1/6) - 100, int(self.height*1/12))
        self.label.resize(int(self.width*2/3), int(self.height*2/3))
        #th = VideoThread()
        #th.changePixmap.connect(self.setImage)
        #th.start()
        self.show_button = QPushButton('Show Drawing Window', self)
        self.show_button.clicked.connect(self.showDrawingWindow)

        B_detection_is_on = QPushButton('', self)
        B_detection_is_on.setToolTip('Включить|Выключить отслеживание')
        B_detection_is_on_w = int(self.width*2.8/24)
        B_detection_is_on_h = int(self.width*2.8/24)
        B_detection_is_on.resize(B_detection_is_on_w, B_detection_is_on_h)
        B_detection_is_on.setIcon(QIcon('B_cam_show.png'))
        B_detection_is_on.setIconSize(QSize(B_detection_is_on_w - 16, B_detection_is_on_h - 16))
        B_detection_is_on.move(int(self.width/2)-int(B_detection_is_on_w/2) - 100, int(self.height*21/24)-int(B_detection_is_on_h/2))
        B_detection_is_on.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')
        #B_detection_is_on.clicked.connect(self.show_cam)
        #B_detection_is_on.clicked.connect(self.toggleVideo)
        B_detection_is_on.clicked.connect(self.showDrawingWindow)
        self.eye_settings_window = EyeSettings()

        B_eye_detection_settings = QPushButton('', self)
        B_eye_detection_settings.setToolTip('Настройки отслеживания глаз')
        B_eye_detection_settings_w = int(self.width*2.7/24)
        B_eye_detection_settings_h = int(self.width*2.7/24)
        B_eye_detection_settings.resize(B_eye_detection_settings_w, B_eye_detection_settings_h)
        B_eye_detection_settings.setIcon(QIcon('Настройки глаз.png'))
        B_eye_detection_settings.setIconSize(QSize(B_eye_detection_settings_w - 15*2, B_eye_detection_settings_h - 15*2))
        B_eye_detection_settings.move(int(self.width)-int(B_eye_detection_settings_w*2.5) - 15*2, int(self.height*21/24)-int(B_eye_detection_settings_h/2))
        B_eye_detection_settings.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')
        B_eye_detection_settings.clicked.connect(self.show_eye_settings)
        
        self.show()

    def startVideoThread(self):
        if not self.videoThread:  # Запускаем поток только если он ещё не запущен
            self.videoThread = VideoThread()
            self.videoThread.changePixmap.connect(self.setImage)
            self.videoThread.start()

    def stopVideoThread(self):
        if self.videoThread:  # Останавливаем поток, если он запущен
            self.videoThread.terminate()  # Остановка потока
            self.videoThread = None
    def showDrawingWindow(self):
        global detection_is_on
        global client_socket
        if not detection_is_on:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('127.0.0.1', 9556))
                self.drawing_window = DrawingWindow()
                self.drawing_window.show()


                """if data is not None:
                    results = json.loads(data.decode('utf-8'))
                    if isinstance(results, dict):
                        x_value = int(results['x'])
                        y_value = int(results['y'])
                        self.drawing_window.update_coordinates.emit(x_value, y_value)
            """
            except socket.error as e:
                print(f"Не удалось подключиться к серверу: {e}")
                exit(1)
            detection_is_on = True
        else:
            self.stopVideoThread()  # Останавливаем видеопоток
            detection_is_on = False

    def toggleVideo(self):
        global detection_is_on
        if not detection_is_on:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('127.0.0.1', 9556))
                self.startVideoThread()  # Запускаем видеопоток
            except socket.error as e:
                print(f"Не удалось подключиться к серверу: {e}")
                exit(1)
            detection_is_on = True
        else:
            self.stopVideoThread()  # Останавливаем видеопоток
            detection_is_on = False
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())