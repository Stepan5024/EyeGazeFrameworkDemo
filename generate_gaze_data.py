from datetime import datetime
import os
import sys
import random
from typing import Tuple
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
import numpy as np
import pyautogui as pag
import yaml
from scipy.io import savemat
import mediapipe as mp

from server.video_stream.video_stream import VideoStream

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.good_points = 0  # счётчик хороших точек
        self.readConfig(os.path.join('configs', 'gaze.yaml'))
        self.initUI()
        self.createMATFiles()
        self.video_stream = VideoStream(capture_width=1280, capture_height=720)
        self.last_circle_zero = None
        self.paths_list = []
        


    def initUI(self):
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.setWindowTitle("Main Window")
        self.setStyleSheet("background-color: white;")
        self.radius = 30
        self.circle_radius = self.radius  # радиус круга
        self.target_radius = 0  # целевой радиус (для анимации уменьшения)
        self.circle_pos = QtCore.QPoint(0, 0)  # позиция центра круга

         # Таймер для анимации круга
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_circle)
        self.timer.start(80)  # обновление каждые 50 мс
        self.x_point, self.y_point = 0, 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5)
        width, height = pag.size()
        # Счётчик хороших точек
        self.score_label = QtWidgets.QLabel(f"Было записано хороших точек: {self.good_points}", self)
        self.score_label.move(width // 2, 10)  # Позиционирование надписи в верхнем левом углу
        self.score_label.resize(400, 30)

        self.circle = QtWidgets.QLabel(self)
    
    def createDir(self, path):
        
        # path = data_root / person_id / day_id / Calibration

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' was created.")
        else:
            print(f"Directory '{path}' already exists.")
      
    def createMATFiles(self):
        person_id: str = self.configs['person_id']
        day_id: str = self.configs['day']
        data_root: str = self.configs['data_root']

        self.path_days = os.path.join(data_root, person_id, day_id)
        self.path_calib = os.path.join(data_root, person_id, 'Calibration')
        self.createDir(self.path_days)
        self.createDir(self.path_calib)

        self.writeScreenSize()
        self.writeMonitorPose()
        self.writeCameraParam()

        # screenSize dict_keys(['height_mm', 'height_pixel', 'width_mm', 'width_pixel'])
        #  monitorPose: dict_keys(['__header__', '__version__', '__globals__', 'rvects', 'tvecs'])
        # Camera: [[-0.16321888  0.66783406 -0.00121854 -0.00303158 -1.02159927]]
    
    def writeScreenSize(self):
        self.monitor_pixels = tuple(map(int, self.configs.get('monitor_pixels', '1920,1080').split(',')))
        width_mm = self.configs.get('width_mm', 400) 
        height_mm = self.configs.get('height_mm', 250)
        screenSize = {
            'width_pixel': self.monitor_pixels[0],
            'height_pixel': self.monitor_pixels[1],
            'width_mm': width_mm,
            'height_mm': height_mm
        }
        file = os.path.join(self.path_calib, 'screenSize.mat')
        savemat(file, screenSize)
        print(f"Data saved to {file}.mat")

        
    def writeMonitorPose(self):
        rvecs = self.configs['rvecs']
        tvecs = self.configs['tvecs']
        rvecs_np = np.array(rvecs)
        tvecs_np = np.array(tvecs)
        file = os.path.join(self.path_calib, 'monitorPose.mat')
        savemat(file, {'rvecs': rvecs_np, 'tvecs': tvecs_np})
        print(f"Monitor pose data saved to {file}")

    def writeCameraParam(self):
        camera_matrix = self.configs['camera_matrix']
        camera_matrix_np = np.array(camera_matrix)
        file = os.path.join(self.path_calib, 'Camera.mat')
        savemat(file, {'camera_matrix': camera_matrix_np})
        print(f"Camera parameters saved to {file}")

    def is_cursor_in_circle(self, center: Tuple[int, int]) -> bool:
        """
        Check if the cursor is within 20% of the center of the circle.
    
        :param cursor_pos: Current cursor position (x, y)
        :param center: Center of the circle (x, y)
        :param monitor_pixels: monitor dimensions in pixels (width, height)
        :return: True if cursor is within the area, False otherwise
        """
        # Calculate 20% of the shortest monitor dimension as the "radius"
        percent = 0.2
        cursor_pos = pag.position()
        radius = percent * min(self.monitor_pixels) / 2
    
        # Calculate the distance between the cursor and the center of the circle
        distance = ((cursor_pos.x - center[0]) ** 2 + (cursor_pos.y - center[1]) ** 2) ** 0.5
    
        # Check if the cursor is within 20% of the radius from the center
        return distance <= radius
    

    def readConfig(self, path: str):
        path_to_config = self.resource_path(path)
        self.configs: dict = self.load_config(path_to_config)

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def resource_path(self, relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        #if getattr(sys, 'frozen', False):
        try:
            # PyInstaller создаёт временную папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
    
        return os.path.join(base_path, relative_path)
    
    def update_circle(self):
        if self.circle_radius > self.target_radius:
            self.circle_radius -= 1
        else:
            if self.circle_radius == 0:  # Увеличиваем счётчик, когда круг полностью исчез

                if(self.is_cursor_in_circle((self.x_point, self.y_point))):
                    self.good_points += 1
                    self.score_label.setText(f"Было записано хороших точек: {self.good_points}")
                    self.save_images()
            self.circle_radius = self.radius  # восстанавливаем исходный размер круга
            self.x_point = random.randint(0, self.width() - 40)
            self.y_point = random.randint(0, self.height() - 40)
            self.circle_pos = QtCore.QPoint(self.x_point + 20, self.y_point + 20)
        self.update()  # перерисовка виджета
    
    def save_images(self):
        image = self.video_stream.read_frame()
        img_rgb = self.video_stream.convert_color(image, to_rgb=True)
        # обрезать лицо и получить изображение где оно находится в кадре 96 на 96
        # обрезать правый глаз и получить изображение где оно находится в кадре 96 на 64
        # обрезать левый глаз и получить изображение где оно находится в кадре 96 на 64

        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Assuming facial landmarks for eyes and face are available
                # For simplicity, taking bounding box around all landmarks for the face
                # Adjust indexes for specific landmarks (e.g., eyes)
                face_image = self.extract_area(image, face_landmarks, scale=1.5)
                right_eye_image = self.extract_area(image, face_landmarks, specific_landmarks=[33, 133, 160, 158], scale=1.5, desired_size=(96, 64))  # Right eye indices
                left_eye_image = self.extract_area(image, face_landmarks, specific_landmarks=[362, 263, 387, 385], scale=1.5, desired_size=(96, 64))  # Left eye indices

                if face_image is not None:
                    self.create_image(face_image, 'full_face')
                    #cv2.imwrite('face_image.png', )
                else:
                    print("Failed to extract face image.")

                if right_eye_image is not None:
                    self.create_image(right_eye_image, 'right_eye')
                    #cv2.imwrite('right_eye_image.png', right_eye_image)
                else:
                    print("Failed to extract right eye image.")

                if left_eye_image is not None:
                    self.create_image(left_eye_image, 'left_eye')
                    #cv2.imwrite('left_eye_image.png', left_eye_image)
                else:
                    print("Failed to extract left eye image.")
                print("Images saved.")
        else:
            print("No faces detected.")


    def extract_area(self, image, face_landmarks, specific_landmarks=None, scale=1, desired_size=(96, 96)):
        if specific_landmarks:
            landmarks = [face_landmarks.landmark[i] for i in specific_landmarks]
        else:
            landmarks = face_landmarks.landmark

        xs = [landmark.x * image.shape[1] for landmark in landmarks]
        ys = [landmark.y * image.shape[0] for landmark in landmarks]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = (x_max - x_min) * scale
        height = (y_max - y_min) * scale
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        x_min = max(0, int(x_center - width / 2))
        x_max = min(image.shape[1], int(x_center + width / 2))
        y_min = max(0, int(y_center - height / 2))
        y_max = min(image.shape[0], int(y_center + height / 2))
            # Extract the region
        cropped_image = image[y_min:y_max, x_min:x_max] if y_max > y_min and x_max > x_min else None
        if cropped_image is not None:
            # Resize the extracted region to the desired size
            return cv2.resize(cropped_image, desired_size)
        else:
            return None


    def create_image(self, image, postfix: str):
        file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        full_file_path = os.path.join(self.path_days, f"{file_name}-{postfix}.jpg")
        self.add_to_paths(postfix, full_file_path)
        cv2.imwrite(full_file_path, image)


    
    def add_to_paths(self, postfix: str, path:str):
         # Список значений postfix, при которых функция не будет выполнять сохранение
        skip_postfixes = ['left_eye', 'right_eye', 'landmark']
        #
        # pxx, dayxx, filename-postfix
        
        parts = path.split(os.sep)
        # Выбрать необходимые части пути (pxx, dayxx, filename без расширения)
        relevant_parts = parts[-3:-1]  # pxx и dayxx
        
        print(f"relevant_parts {relevant_parts}")

        # Проверяем, содержится ли postfix в списке skip_postfixes
        if postfix in skip_postfixes:
            print(f"Skipping saving for postfix '{postfix}'.")
            return
        #
        self.paths_list.append(relevant_parts)


    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))  # красный цвет круга
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(self.circle_pos, self.circle_radius, self.circle_radius)


        # Catch key press events
        self.keyPressEvent = self.handleKeyPressEvents


    def handleKeyPressEvents(self, event):
        if event.key() in [QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape]:
            self.close()
            self.stat_window = StatApp()
            self.stat_window.show()

class StatApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Statistics")
        self.setGeometry(100, 100, 200, 100)
        label = QtWidgets.QLabel("Статистика", self)
        label.move(50, 40)

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
