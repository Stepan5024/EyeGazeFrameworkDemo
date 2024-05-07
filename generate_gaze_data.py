import collections
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
import skimage
import torch
import yaml
from scipy.io import savemat
import mediapipe as mp
import pandas as pd
import h5py

from server.models.face_model import face_model_all
from server.processors.mpii_face_gaze_preprocessing import normalize_single_image
from server.utils.camera_utils import get_face_landmarks_in_ccs
from server.gaze_predictors.gazePredictor import GazePredictor
from server.models.gazeModel import GazeModel
from server.utils.camera_utils import get_camera_matrix
from server.video_stream.video_stream import VideoStream

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.good_points = 0  # счётчик хороших точек
        self.readConfig(os.path.join('configs', 'gaze_win10.yaml'))
        self.initUI()
        self.createMATFiles()
        # F:\EyeGazeDataset\MPIIFaceGaze_post_proccessed_author_pperle
        file = 'data.h5'
        self.path_to_h5 = os.path.join(self.configs['data_root'], file)
        self.df = self.open_or_create_h5(self.path_to_h5)
        print(self.df)
        self.video_stream = VideoStream(capture_width=1280, capture_height=720)
        self.last_circle_zero = None
        #self.paths_list = []
        self.used_file_ids = []
        self.landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
        self.face_model = np.asarray([face_model_all[i] for i in self.landmarks_ids])
        self.rvec_buffer = collections.deque(maxlen=3)
        self.tvec_buffer = collections.deque(maxlen=3)
        self.smoothing_buffer = collections.deque(maxlen=3)

        self.setDevice()
        self.initCamera()
        self.initGazeModel()

    def initCamera(self):
        self.video_stream = VideoStream()
        calibration_matrix_path = os.path.join("resources", "calib", "calibration_matrix.yaml")
        abs_calib_path = self.resource_path(calibration_matrix_path)
        self.camera_matrix, self.dist_coefficients = get_camera_matrix(abs_calib_path)
        print(f"self.camera_matrix {self.camera_matrix}, self.dist_coefficients  {self.dist_coefficients}")

    def initGazeModel(self):
        """Инициализация СНС определяющей взгляд"""
        relative_path_gaze_model = os.path.join("resources", "models", "gaze", "p00.ckpt")
        abs_path = self.resource_path(relative_path_gaze_model)
        gaze_model = GazeModel.load_checkpoint(abs_path) 
        gaze_model.to(self.device)
        gaze_model.eval()
        self.gaze_pipeline_CNN = GazePredictor(gaze_model, 
            self.camera_matrix, 
            self.dist_coefficients)

    def setDevice(self):
        dev = self.configs['device']
        if dev == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')



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
        self.timer.start(60)  # обновление каждые 50 мс
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
    
    def equalize_hist_rgb(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Equalize the histogram of a RGB image.

        :param rgb_img: RGB image
        :return: equalized RGB image
        """
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)  # convert from RGB color-space to YCrCb
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])  # equalize the histogram of the Y channel
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)  # convert back to RGB color-space from YCrCb
        return equalized_img
    
    def get_matrices(self, camera_matrix: np.ndarray, distance_norm: int, center_point: np.ndarray, focal_norm: int, head_rotation_matrix: np.ndarray, image_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate rotation, scaling and transformation matrix.

        :param camera_matrix: intrinsic camera matrix
        :param distance_norm: normalized distance of the camera
        :param center_point: position of the center in the image
        :param focal_norm: normalized focal length
        :param head_rotation_matrix: rotation of the head
        :param image_output_size: output size of the output image
        :return: rotation, scaling and transformation matrix
        """
        # normalize image
        distance = np.linalg.norm(center_point)  # actual distance between center point and original camera
        z_scale = distance_norm / distance

        cam_norm = np.array([
            [focal_norm, 0, image_output_size[0] / 2],
            [0, focal_norm, image_output_size[1] / 2],
            [0, 0, 1.0],
        ])

        scaling_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        forward = (center_point / distance).reshape(3)
        down = np.cross(forward, head_rotation_matrix[:, 0])
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)

        rotation_matrix = np.asarray([right, down, forward])
        transformation_matrix = np.dot(np.dot(cam_norm, scaling_matrix), np.dot(rotation_matrix, np.linalg.inv(camera_matrix)))

        return rotation_matrix, scaling_matrix, transformation_matrix

    def createDir(self, path):
        
        # path = data_root / person_id / day_id / Calibration

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' was created.")
        else:
            print(f"Directory '{path}' already exists.")
    
    def open_or_create_h5(self, path):
        # Check if the file exists
        if os.path.exists(path):
            # Read data from the HDF5 file
            with h5py.File(path, 'r') as h5file:
                # Preprocess multi-dimensional data before DataFrame creation
                file_name_base = list(h5file['file_name_base'])
                gaze_location = h5file['gaze_location'][:]
                gaze_pitch = h5file['gaze_pitch'][:]
                gaze_yaw = h5file['gaze_yaw'][:]
                screen_size = h5file['screen_size'][:]

                # Convert multi-dimensional arrays to list of tuples
                gaze_location_list = gaze_location.tolist()
                screen_size_list = screen_size.tolist()

            # Create DataFrame from 1D lists
            df = pd.DataFrame({
                'file_name_base': file_name_base,
                'gaze_pitch': gaze_pitch,
                'gaze_yaw': gaze_yaw,
            })

            if len(gaze_location_list) == len(df):
            # Split tuples into separate columns
                df[['gaze_location_0', 'gaze_location_1']] = pd.DataFrame(gaze_location_list, index=df.index)
            else:
                print("Error: Length of gaze_location_list does not match DataFrame index length.")
            if len(screen_size_list) == len(df):
                df[['screen_size_0', 'screen_size_1']] = pd.DataFrame(screen_size_list, index=df.index)
            else:
                print("Error: Length of screen_size_list does not match DataFrame index length.")
            return df

        else:
            # Create a new HDF5 file with empty datasets
            with h5py.File(path, 'w') as h5file:
                dt_str = h5py.special_dtype(vlen=str)  # for strings
                h5file.create_dataset('file_name_base', (0,), maxshape=(None,), dtype=dt_str)
                h5file.create_dataset('gaze_location', (0, 2), maxshape=(None, 2), dtype=np.int32)
                h5file.create_dataset('gaze_pitch', (0,), maxshape=(None,), dtype=np.float32)
                h5file.create_dataset('gaze_yaw', (0,), maxshape=(None,), dtype=np.float32)
                h5file.create_dataset('screen_size', (0, 2), maxshape=(None, 2), dtype=np.int32)
            # Create an empty DataFrame with necessary columns
            columns = ['file_name_base', 'gaze_location_0', 'gaze_location_1', 'gaze_pitch', 'gaze_yaw', 'screen_size_0', 'screen_size_1']
            df = pd.DataFrame(columns=columns)
            return df

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
        self.camera_matrix_np = np.array(camera_matrix)
        file = os.path.join(self.path_calib, 'Camera.mat')
        savemat(file, {'camera_matrix': self.camera_matrix_np})
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
        image_rgb = self.video_stream.convert_color(image, to_rgb=True)
        # обрезать лицо и получить изображение где оно находится в кадре 96 на 96
        # обрезать правый глаз и получить изображение где оно находится в кадре 96 на 64
        # обрезать левый глаз и получить изображение где оно находится в кадре 96 на 64
        # Индексы для ключевых точек
        height, width, _ = image_rgb.shape
        rvec, tvec = None, None
        results = self.face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            # head pose estimation
            #landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
    
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] 
                                         for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in self.landmarks_ids])
            self.smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(self.smoothing_buffer).mean(axis=0)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(self.face_model, 
                                                              face_landmarks,
                                                               self.camera_matrix, 
                                                               self.dist_coefficients, 
                                                               rvec=rvec, tvec=tvec, 
                                                               useExtrinsicGuess=True, 
                                                               flags=cv2.SOLVEPNP_EPNP)  # Initial fit
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(self.face_model, 
                                                   face_landmarks, 
                                                   self.camera_matrix, 
                                                   self.dist_coefficients, 
                                                   rvec=rvec, tvec=tvec, 
                                                   useExtrinsicGuess=True, 
                                                   flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy
            self.rvec_buffer.append(rvec)
            rvec = np.asarray(self.rvec_buffer).mean(axis=0)
            self.tvec_buffer.append(tvec)
            tvec = np.asarray(self.tvec_buffer).mean(axis=0)
            # data preprocessing
            (face_model_transformed, 
             face_model_all_transformed) = get_face_landmarks_in_ccs(self.camera_matrix,
                self.dist_coefficients, image_rgb.shape, results, self.face_model, 
                face_model_all, self.landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # center eye
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # center eye
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))
            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, 
                                                               rvec, None, 
                                                               left_eye_center, self.camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb,
                                                                 rvec, None, 
                                                                 right_eye_center, self.camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, 
                                                                         rvec, None, 
                                                                         face_center, self.camera_matrix, is_eye=False)
            file_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            if image_rgb is not None:
                #name = file_name + f"=x={self.x_point}=y={self.y_point}=w={self.monitor_pixels[0]}=h={self.monitor_pixels[1]}"
                self.create_image(image_rgb, 'origin', file_name)
                #cv2.imwrite('face_image.png', )
                
            if img_warped_face is not None:
                print
                self.create_image(img_warped_face, 'full_face', file_name)
                #cv2.imwrite('face_image.png', )
            else:
                print("Failed to extract face image.")
            if img_warped_right_eye is not None:
                self.create_image(img_warped_right_eye, 'right_eye', file_name)
                #cv2.imwrite('right_eye_image.png', right_eye_image)
            else:
                print("Failed to extract right eye image.")
            if img_warped_left_eye is not None:
                self.create_image(img_warped_left_eye, 'left_eye', file_name)
                #cv2.imwrite('left_eye_image.png', left_eye_image)
            else:
                print("Failed to extract left eye image.")
         

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


    def create_image(self, image, postfix: str, file_name: str):
        if postfix=='origin':
            
            name = file_name + f"=x={self.x_point}=y={self.y_point}=w={self.monitor_pixels[0]}=h={self.monitor_pixels[1]}"
            temp_file_path = os.path.join(self.path_days, f"{file_name}-{postfix}.png")
            file_id: str = self.add_to_paths(postfix, temp_file_path)
            full_file_path = os.path.join(self.path_days, f"{name}-{postfix}.png")
            skimage.io.imsave(full_file_path, image.astype(np.uint8), check_contrast=False)
        else:

            full_file_path = os.path.join(self.path_days, f"{file_name}-{postfix}.png")
            file_id: str = self.add_to_paths(postfix, full_file_path)
            print(f"file_id {file_id}")
            # нормализовать изображение и сохранить в папку датасета
            #cv2.imwrite(full_file_path, image)
            skimage.io.imsave(full_file_path, image.astype(np.uint8), check_contrast=False)

            if file_id is None:
                return
            self.gaze_pipeline_CNN.calculate_gaze_point(image)
        new_row = pd.DataFrame({
                'file_name_base': [file_id.encode('utf-8')],
                'gaze_location_0': [self.x_point],
                'gaze_location_1': [self.y_point],
                'gaze_pitch': self.gaze_pipeline_CNN.get_pitch(),
                'gaze_yaw': self.gaze_pipeline_CNN.get_yaw(),
                'screen_size_0': [self.monitor_pixels[0]],
                'screen_size_1': [self.monitor_pixels[1]]
            })

        # Add row to DataFrame using concat
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        print(self.df)


    
    def add_to_paths(self, postfix: str, path:str) -> str:
         # Список значений postfix, при которых функция не будет выполнять сохранение
        skip_postfixes = ['left_eye', 'right_eye', 'original']
        #
        # pxx, dayxx, filename-postfix
        if postfix in skip_postfixes:
            print(f"Skipping saving for postfix '{postfix}'.")
            return
        parts = path.split(os.sep)
        # Выбрать необходимые части пути (pxx, dayxx, filename без расширения)
        relevant_parts = parts[-3:-1]  # pxx и dayxx
        # Concatenate relevant parts into a string
        file_id = '/'.join(relevant_parts) + '/' + os.path.splitext(parts[-1])[0]
        print(f"relevant_parts {file_id}")
        #a = file_id.split('-')[0]
        # Проверяем, содержится ли postfix в списке skip_postfixes
        if file_id.split('-')[0] in self.used_file_ids:
            print(f"File ID '{file_id.split('-')[0]}' has already been used. Skipping.")
            return None
        else:
            # Если file_id новый, добавляем его в список использованных
            self.used_file_ids.append(file_id.split('-')[0])
            input_string = file_id.split('-')[0]
            output_string = input_string.replace("/", "\\")
            print(f"file_id {output_string}")
            return output_string
    

    def closeEvent(self, event):
        # This method is called automatically when the window is about to close.
        self.save_data(self.path_to_h5)  # Save the data
        #print("Data saved on window close.")
        event.accept()  # Accept the close event to close the window

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))  # красный цвет круга
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(self.circle_pos, self.circle_radius, self.circle_radius)


        # Catch key press events
        self.keyPressEvent = self.handleKeyPressEvents

    def save_data(self, path='data.h5'):
        with h5py.File(path, 'w') as h5file:
            # Convert DataFrame columns to lists or numpy arrays as needed
            file_name_base = self.df['file_name_base'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x).tolist()
            gaze_location = self.df[['gaze_location_0', 'gaze_location_1']].to_numpy(dtype=np.int32)  # Ensure float32 for better compatibility
            gaze_pitch = self.df['gaze_pitch'].to_numpy(dtype=np.float32)
            gaze_yaw = self.df['gaze_yaw'].to_numpy(dtype=np.float32)
            screen_size = self.df[['screen_size_0', 'screen_size_1']].to_numpy(dtype=np.int32)

            # Create datasets
            h5file.create_dataset('file_name_base', data=np.array(file_name_base, dtype='S'), maxshape=(None,)) 
            h5file.create_dataset('gaze_location', data=gaze_location, maxshape=(None, 2), dtype=np.int32)
            h5file.create_dataset('gaze_pitch', data=gaze_pitch, maxshape=(None,), dtype=np.float32)
            h5file.create_dataset('gaze_yaw', data=gaze_yaw, maxshape=(None,), dtype=np.float32)
            h5file.create_dataset('screen_size', data=screen_size, maxshape=(None, 2), dtype=np.int32)

            print("Data successfully saved to", path)
            print(self.df)

    def handleKeyPressEvents(self, event):
        if event.key() in [QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape]:
            self.save_data(self.path_to_h5)
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
