import base64
import collections
import os
import pathlib
import sys
import socket
import json
import traceback
import cv2
import numpy as np
import yaml

import pyautogui as pag

from PyQt5.QtWidgets import  QVBoxLayout, QWidget, QLabel, QApplication, QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon

from logger import create_logger

width, height = pag.size()
detection_is_on = False
client_socket = None
logger = None


class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.gaze_points = collections.deque(maxlen=64)
        screen_width, screen_height = pag.size()
        self.monitor_pixels = (screen_width, screen_height)

    def process_image_and_coordinates(self, results):
        # Преобразование данных изображения
        image_data = results['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Получение и вывод координат
        self.x_value, self.y_value = map(int, results['coordinates'])
        #print(f"x {self.x_value} y {self.y_value}")

        # Вывод эмоций
        self.emotions = results['emotions']
        print("Detected emotions:", self.emotions)
        logger.info(f"Detected emotions: {self.emotions}\n")

    def run(self):
        global detection_is_on
        display = np.ones((self.monitor_pixels[1], self.monitor_pixels[0], 3), dtype=np.uint8)
        while True:
            if detection_is_on:
                if not client_socket:
                    print("Соединение не установлено. Попытка переподключения...")
                    logger.error(f"Соединение не установлено. Попытка переподключения...\n")
                elif client_socket is not None:
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
                            print(f"Ошибка: {e}")
                            logger.error(f"Ошибка при подключении к серверу: {e}\n")
                        if not packet:
                            break
                        data += packet
                    if data is not None:
                        results = json.loads(data.decode('utf-8'))
                        status = results.get('status')

                        if status == '1':
                            print("System error: Произошла системная ошибка.")
                            logger.error(f"System error: Произошла системная ошибка.\n")
                        elif status == '2':
                            #print("Image and features processing successful.")
                            # Продолжение обработки изображения и координат
                            self.process_image_and_coordinates(results)
                        elif status == '3':
                            print("User not detected: Пользователь не обнаружен.")
                            logger.error(f"User not detected: Пользователь не обнаружен.\n")
                        elif status == '4':
                            print("Image capture failed: Ошибка при захвате изображения.")
                            logger.error(f"Image capture failed: Ошибка при захвате изображения.\n")
                        elif status == '5':
                            print("Image processing error: Ошибка обработки изображения.")
                            logger.error(f"Image processing error: Ошибка обработки изображения.\n")
                        else:
                            print("Unknown status received.")
                            logger.error(f"Unknown status received.\n")
                                            
                point_on_screen = (self.x_value, self.y_value)
                self.gaze_points.appendleft(point_on_screen)
                display.fill(255)
                for idx in range(1, len(self.gaze_points)):
                    thickness = round((len(self.gaze_points) - idx) / len(self.gaze_points) * 5) + 1
                    cv2.line(display, self.gaze_points[idx - 1], self.gaze_points[idx], (0, 0, 255), thickness)

                rgbImage = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(self.monitor_pixels[0], self.monitor_pixels[1], Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            else:
                detection_is_on = False
                break


class EyeSettings(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.title = config['eye_title']
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
        B_back.setIcon(QIcon(os.path.join(self.config['images_path'], "Arrow_left.png")))
        B_back.setIconSize(QSize(B_back_w - 16, B_back_h - 16))
        B_back.move(15, 15)
        B_back.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')
        B_back.clicked.connect(self.back_to_main_window)

class DrawingWindow(QWidget):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.config['gaze_title'])
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
            self.close()


class App(QWidget):
    def __init__(self):
        super().__init__()
        logger.info(f"Клиент начал работу\n")
        self.readConfig(os.path.join('configs', 'gaze_client.yaml'))
        self.title:str = self.configs['main_title']
        self.left = int(width*1/6)
        self.top = int(height*1/6)
        self.width = int(width*2/3)
        self.height = int(height*2/3)
        self.videoThread = None
        self.initUI()
    
    def readConfig(self, path: str):
        path_to_config = resource_path(path)
        self.configs: dict = self.load_config(path_to_config)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    
    
    def show_eye_settings(self):
        self.eye_settings_window.show()
        self.eye_settings_window.move(self.geometry().x() - 1, self.geometry().y() - 45)
        self.hide()
        global main_window
        main_window = self

    # Функция для загрузки конфигов из YAML файла
    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.resize(int(width*2/3), int(height*2/3))
        self.setStyleSheet('background-color: #f2f2f2;')
        self.label = QLabel(self)
        self.label.move(int(self.width*1/6) - 100, int(self.height*1/12))
        self.label.resize(int(self.width*2/3), int(self.height*2/3))

        self.errorLabel = QLabel(self)
        self.errorLabel.move(10, self.height - 400)
        self.errorLabel.resize(self.width - 20, 30)
        self.errorLabel.setStyleSheet("QLabel { color: red; font-size: 14px; }")

        B_detection_is_on = QPushButton('', self)
        B_detection_is_on.setToolTip('Включить|Выключить отслеживание')
        B_detection_is_on_w = int(self.width*2.8/24)
        B_detection_is_on_h = int(self.width*2.8/24)
        B_detection_is_on.resize(B_detection_is_on_w, B_detection_is_on_h)
        path = resource_path(os.path.join(self.configs['images_path'], "B_cam_show.png"))
        B_detection_is_on.setIcon(QIcon(path))
        B_detection_is_on.setIconSize(QSize(B_detection_is_on_w - 16, B_detection_is_on_h - 16))
        B_detection_is_on.move(int(self.width/2)-int(B_detection_is_on_w/2) - 100, int(self.height*21/24)-int(B_detection_is_on_h/2))
        B_detection_is_on.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')

        B_detection_is_on.clicked.connect(self.showDrawingWindow)
        self.eye_settings_window = EyeSettings(self.configs)

        B_eye_detection_settings = QPushButton('', self)
        B_eye_detection_settings.setToolTip('Настройки отслеживания глаз')
        B_eye_detection_settings_w = int(self.width*2.7/24)
        B_eye_detection_settings_h = int(self.width*2.7/24)
        B_eye_detection_settings.resize(B_eye_detection_settings_w, B_eye_detection_settings_h)
        B_eye_detection_settings.setIcon(QIcon(os.path.join(self.configs['images_path'], "settings_eye.png")))
        B_eye_detection_settings.setIconSize(QSize(B_eye_detection_settings_w - 15*2, B_eye_detection_settings_h - 15*2))
        B_eye_detection_settings.move(int(self.width)-int(B_eye_detection_settings_w*2.5) - 15*2, int(self.height*21/24)-int(B_eye_detection_settings_h/2))
        B_eye_detection_settings.setStyleSheet('QPushButton {background-color: #d6d6d6; color: black;}')
        B_eye_detection_settings.clicked.connect(self.show_eye_settings)
        
        self.show()

    
    def stopVideoThread(self):
        if self.videoThread:  
            self.videoThread.terminate()
            self.videoThread = None
    
    def showDrawingWindow(self):
        global detection_is_on
        global client_socket
        if not detection_is_on:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(('127.0.0.1', 9556))
                self.errorLabel.setText("")
                self.drawing_window = DrawingWindow(self.configs) # отрисовка окна с отслеживанием взгляда
                self.drawing_window.show()

            except socket.error as e:
                tb = traceback.format_exc()
                self.errorLabel.setText(f"Ошибка при подключении к серверу {e}\n{tb}")
                logger.error(f"Ошибка при подключении к серверу: {e}\n{tb}")

            detection_is_on = True
        else:
            self.stopVideoThread()
            detection_is_on = False
            
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()

def resource_path(relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        #if getattr(sys, 'frozen', False):
        try:
            # PyInstaller создаёт временную папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
    
        return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    width, height = pag.size()
    rev_path =  os.path.join("logs", "clients")
    abs_path = pathlib.Path(resource_path(rev_path))
    logger = create_logger("ClientGaze", abs_path, 'client_log.txt')
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())