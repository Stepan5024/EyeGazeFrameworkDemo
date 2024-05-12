import logging
import pathlib
import traceback
from typing import Any, Dict, List, Optional
import cv2
import socket
import sys
import os
import json
import numpy as np
import torch
import threading
from threading import Timer
import time  
from datetime import datetime
import base64

import yaml

from logger import create_logger
from server.emotion.emotion_estimator import EmotionRecognizer
from server.gaze_predictors.gazePredictor import GazePredictor
from server.models.gazeModel import GazeModel
from server.video_stream.video_stream import VideoStream
from server.utils.camera_utils import get_camera_matrix

class GazeTrackingServer:
    def __init__(self):
        self.readConfig(os.path.join('configs', 'server.yaml'))
        self.initLogger()
        self.setDevice()
        self.initCamera()
        
        self.host: str = self.configs['host']
        self.port: int  = self.configs['port']
        self.active_clients: int  = 0
        self.inactivity_timer: Optional[threading.Timer] = None
        self.is_active: bool = True
        self.img: Optional[Any]  = None
        self.shared_data: Optional[Any] = None # общий буфер
        self.data_lock: threading.Lock = threading.Lock() # поток обработки
        self.initialize_server_socket()
        self.initGazeModel()
        self.initEmotionModel()


    def start_frame_processing_thread(self) -> None:
        """
        Запускает поток обработки кадров. 
        """
        frame_thread = threading.Thread(target=self.frame_processor)
        frame_thread.daemon = True
        frame_thread.start()

    def frame_processor(self) -> None:
        """
        Метод, который выполняется в потоке и обрабатывает видеокадры до тех пор, пока сервер активен.
        img: Optional[np.array]  # Изображение, полученное от видеопотока, может быть None
        """
        while self.is_active:
            self.img = self.video_stream.read_frame()
            if self.img is not None and self.img.size != 0:
                self.process_and_store_frame(self.img)

    def process_and_store_frame(self, img: np.array) -> None:
        """
        Обрабатывает и сохраняет кадр, используя нейросеть для расчёта точки взгляда и распознавания эмоций.
        img: np.array                  # Изображение для обработки
        x_value: float                 # Рассчитанное значение X точки взгляда
        y_value: float                 # Рассчитанное значение Y точки взгляда
        top_emotions: List[str]        # Список распознанных эмоций
        img_encoded: bytes             # Кодированное изображение
        data: Any                      # Подготовленные данные для общего доступа
        """
        self.gaze_pipeline_CNN.calculate_gaze_point(img)
        x_value: int = self.gaze_pipeline_CNN.get_x()
        y_value: int = self.gaze_pipeline_CNN.get_y()
        top_emotions: List[str] = self.recognizer.predict_emotions(img)
        img_encoded: bytes = self.encode_image(img)
        data = self.prepare_data("2", x_value, y_value, img_encoded, top_emotions)
        with self.data_lock:
            self.shared_data = data
        

    def prepare_data(self, status: str, x_value: int, y_value: int, img_base64: str, emotion: List[List[str]]) -> Dict[str, Any]:
        """Формирование JSON"""
        return {
            'status': status,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'coordinates': [x_value, y_value],
            'datetime': datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
            'emotions': emotion
        }
    
    def send_data(self, conn: socket.socket, data: Dict[str, Any]):
        """Отправляет переданные данные на переданный сокет"""
        message: bytes = json.dumps(data).encode('utf-8')
        message_length: int = len(message)
        conn.sendall(message_length.to_bytes(4, 'big'))
        conn.sendall(message)


    def initLogger(self) -> None:
        """Инициализация логера"""
        rev_path: str = os.path.join("logs", "server")
        abs_path: pathlib.Path = pathlib.Path(self.resource_path(rev_path))
        self.logger: logging.Logger = create_logger("EyeGazeServer", abs_path, 'server_log.txt')
    
    def initCamera(self) -> None:
        """Инициализация видеопотока и загрузка калибровочных данных камеры."""
        self.video_stream: VideoStream = VideoStream()
        calibration_matrix_path: str = os.path.join("resources", "calib", "calibration_matrix.yaml")
        abs_calib_path: str = self.resource_path(calibration_matrix_path)
        self.camera_matrix: np.ndarray = None
        self.dist_coefficients: np.ndarray = None
        self.camera_matrix, self.dist_coefficients = get_camera_matrix(abs_calib_path)
        self.logger.info(f"Camera parametrs camera_matrix {self.camera_matrix} dist_coefficients {self.dist_coefficients}")
    
    def initEmotionModel(self):
        self.recognizer: EmotionRecognizer = EmotionRecognizer()

    def initGazeModel(self) -> None:
        """Инициализация СНС определяющей взгляд"""
        relative_path_gaze_model: str = os.path.join("resources", "models", "gaze", "p00.ckpt")
        abs_path: str = self.resource_path(relative_path_gaze_model)
        gaze_model: GazeModel = GazeModel.load_checkpoint(abs_path) 
        gaze_model.to(self.device)
        gaze_model.eval()
        self.gaze_pipeline_CNN: GazePredictor = GazePredictor(gaze_model, 
            self.camera_matrix, 
            self.dist_coefficients)

    def setDevice(self) -> None:
        dev: str = self.configs['device']
        self.device: torch.device = None
        if dev == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

    def readConfig(self, path: str) -> None:
        path_to_config: str = self.resource_path(path)
        self.configs: dict = self.load_config(path_to_config)

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def resource_path(self, relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        try:
            # временная папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    def initialize_server_socket(self) -> None:
        """Инициализация сокета сервера."""
        self.logger.info("Application start")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f'Server listening on {self.host}:{self.port}')
        self.logger.info(f'Server listening on {self.host}:{self.port}')
    
    def shutdown_server(self) -> None:
        if self.active_clients == 0:
            #print("No active clients. Shutting down the server.")
            self.logger.info("No active clients. Shutting down the server.")
            self.close_resources()
            self.is_active = False
            threading.current_thread()._delete()
        else:
            self.inactivity_timer.cancel()

    def reset_inactivity_timer(self) -> None:
        if self.active_clients == 0:
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
            self.inactivity_timer = Timer(35.0, self.shutdown_server)
            self.inactivity_timer.start()
        else:
            self.inactivity_timer.cancel()
        
    def close_resources(self) -> None:
        """Освобождение ресурсов после закрытия сокета"""
        self.server_socket.close()
        self.video_stream.release()

    def encode_image(self, img: np.ndarray) -> bytes:
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def client_handler(self, conn: socket.socket, addr: tuple):
        self.active_clients += 1
        self.logger.info(f"Client {addr} connected. Total clients: {self.active_clients}")
        try:
            while True:
                with self.data_lock:
                    data_to_send = self.shared_data
                if data_to_send:
                    try:
                        self.send_data(conn, data_to_send)
                    except socket.error as e:
                        tb = traceback.format_exc()
                        self.logger.error(f"Socket send error: {e}\n{tb}")
                        break
                time.sleep(0.05)           
        except socket.error as e:
            tb = traceback.format_exc()
            self.logger.error(f"Socket error: {e}\n{tb}")
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Exception in client_handler: {e}\n{tb}")
        finally:
            conn.close()
            self.active_clients -= 1
            self.logger.info(f"Client {addr} disconnected. Total clients: {self.active_clients}")
            self.reset_inactivity_timer()
        
    def run(self):
        self.reset_inactivity_timer()
        self.start_frame_processing_thread()
        while self.is_active:
            try:
                conn, addr = self.server_socket.accept()
                self.reset_inactivity_timer()
                
                thread = threading.Thread(target=self.client_handler, args=(conn, addr))
                thread.start()
            except Exception as e:
                self.logger.error(f"server error: {e}")
                self.close_resources()
                time.sleep(5)

def main():
    server: GazeTrackingServer = GazeTrackingServer()
    server.run()

if __name__ == '__main__':
    main()