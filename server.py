import collections
import cv2
import socket
import sys
import os
import json
#from l2cs import Pipeline
import torch
#import pytorch_lightning as pl
import threading
from threading import Timer
import time  # Импортируем модуль time
import importlib.metadata

from server.gaze_predictors.gazeCNN import GazeCNN
from server.models.modelCNN import ModelCNN
from server.video_stream.video_stream import VideoStream
from server.utils.camera_utils import get_camera_matrix

class GazeTrackingServer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.host = '127.0.0.1'
        self.port = 9556
        self.active_clients = 0
        self.inactivity_timer = None
        self.is_active = True
        self.initialize_server_socket()
        self.monitor_mm = (200, 100)  # размер монитора в миллиметрах
        self.monitor_pixels = (1700, 800)  # разрешение монитора в пикселях
        self.gaze_points = collections.deque(maxlen=64)
        self.video_stream = VideoStream()
        self.model = self.init_model_CNN()
        self.gaze_pipeline_CNN = GazeCNN(self.model, 
                                self.camera_matrix, 
                                self.dist_coefficients, 
                         self.device)
        self.model = self.init_prod_model()
    
    def initialize_server_socket(self):
        """Initialize or reinitialize the server socket."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f'Server listening on {self.host}:{self.port}')
    
    def shutdown_server(self):
        print("No active clients. Shutting down the server.")
        self.close_resources()
        self.is_active = False
        threading.current_thread()._delete()

    def reset_inactivity_timer(self):
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        self.inactivity_timer = Timer(20.0, self.shutdown_server)
        self.inactivity_timer.start()
        
    def close_resources(self):
        """Close resources before restarting or shutting down."""
        self.server_socket.close()
        self.video_stream.release()

    def init_prod_model(self):
        if (self.gaze_pipeline_CNN is not None):
            
            self.model = self.gaze_pipeline_CNN
        
        print(f"inited model {self.model}")


    def init_model_CNN(self):
        # Инициализация модели
        # Путь к базовой директории
        if getattr(sys, 'frozen', False):
            # Если приложение 'заморожено', используйте этот путь
            base_path = sys._MEIPASS
        else:
            # Иначе используйте обычный путь
            base_path = os.path.abspath(".")

        # Используйте os.path.join для создания пути к файлам
        self.calibration_matrix_path = os.path.join(base_path, "server", "calibration", "calibration_matrix.yaml")
        self.camera_matrix, self.dist_coefficients = get_camera_matrix(self.calibration_matrix_path)
        # Загрузите чекпоинт
        model_path = os.path.join(base_path, "server", "models_cnn", "p01.ckpt")
        model = ModelCNN.load_checkpoint(model_path)  # Измененный метод загрузки модели
        model.to(self.device)
        model.eval()
        self.model = model
        return model
    
   
    def client_handler(self, conn, addr):
        print(f"Connected by {addr}")
        while True:
            img = self.video_stream.read_frame()
            if self.gaze_pipeline_CNN is not None:
                self.gaze_pipeline_CNN.calculate_gaze_point(img)
                x_value = self.gaze_pipeline_CNN.get_x()
                y_value = self.gaze_pipeline_CNN.get_y()
                print(f"Point on screen {x_value} {y_value}")
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            img_hex = img_bytes.hex()
            data = {
                'x': x_value,
                'y': y_value,
                'img': img_hex
            }
            message = json.dumps(data).encode('utf-8')
            message_length = len(message)
            conn.sendall(message_length.to_bytes(4, 'big'))
            conn.sendall(message)
        
        
    def run(self):
        self.reset_inactivity_timer()  # Start the inactivity timer when server starts
        while self.is_active:
            try:
                conn, addr = self.server_socket.accept()
                self.reset_inactivity_timer()  # Reset the timer on every new connection
                self.active_clients += 1
                print(f'Client {addr} connected. Total clients: {self.active_clients}')
                thread = threading.Thread(target=self.client_handler, args=(conn, addr))
                thread.start()
            except Exception as e:
                print(f"Critical server error: {e}, restarting server...")
                self.close_resources()
                time.sleep(5)  # Wait before restarting
                self.initialize_server_socket()  # Reinitialize socket

# Запуск сервера
server = GazeTrackingServer()
server.run()

