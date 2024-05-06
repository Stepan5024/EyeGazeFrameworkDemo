import logging
import pathlib
import traceback
from typing import List
import cv2
import socket
import sys
import os
import json
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
        self.setDevice()
        self.initCamera()
        self.initLogger()
        self.host = self.configs['host']
        self.port = self.configs['port']
        self.active_clients = 0
        self.inactivity_timer = None
        self.is_active = True
        self.initialize_server_socket()
        self.initGazeModel()
        self.initEmotionModel()

    def initLogger(self):
        rev_path =  os.path.join("logs", "server")
        abs_path = pathlib.Path(self.resource_path(rev_path))
        self.logger = create_logger("EyeGazeServer", abs_path, 'server_log.txt')
    
    def initCamera(self):
        self.video_stream = VideoStream()
        calibration_matrix_path = os.path.join("resources", "calib", "calibration_matrix.yaml")
        abs_calib_path = self.resource_path(calibration_matrix_path)
        self.camera_matrix, self.dist_coefficients = get_camera_matrix(abs_calib_path)
        print(f"self.camera_matrix {self.camera_matrix}, self.dist_coefficients  {self.dist_coefficients}")
    
    def initEmotionModel(self):
        self.recognizer = EmotionRecognizer()

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
    
    def initialize_server_socket(self):
        """Инициализация сокета сервера."""
        self.logger.info("Application start")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f'Server listening on {self.host}:{self.port}')
        self.logger.info(f'Server listening on {self.host}:{self.port}')
    
    def shutdown_server(self):
        if self.active_clients == 0:
            print("No active clients. Shutting down the server.")
            self.logger.info("No active clients. Shutting down the server.")
            self.close_resources()
            self.is_active = False
            threading.current_thread()._delete()
        else:
            self.inactivity_timer.cancel()


    def reset_inactivity_timer(self):
        if self.active_clients == 0:
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
            self.inactivity_timer = Timer(10.0, self.shutdown_server)
            self.inactivity_timer.start()
        else:
            self.inactivity_timer.cancel()
        
    def close_resources(self):
        """Close resources before restarting or shutting down."""
        self.server_socket.close()
        self.video_stream.release()
    
    def send_data(self, conn, data):
        message = json.dumps(data).encode('utf-8')
        message_length = len(message)
        conn.sendall(message_length.to_bytes(4, 'big'))
        conn.sendall(message)
    
    def process_frame(self, conn):
        img = self.video_stream.read_frame()
        if self.is_active and img is not None and img.size != 0 and self.gaze_pipeline_CNN is not None:
            self.handle_image_processing_and_sending(img, conn)
            return True
        else:
            logging.warning("Received an empty frame.")
            return False
    
    def handle_image_processing_and_sending(self, img, conn):
        self.gaze_pipeline_CNN.calculate_gaze_point(img)
        x_value = self.gaze_pipeline_CNN.get_x()
        y_value = self.gaze_pipeline_CNN.get_y()
        top_emotions = self.recognizer.predict_emotions(img)
        try:
            img_encoded = self.encode_image(img)
            data = self.prepare_data("2", x_value, y_value, img_encoded, top_emotions)
            self.send_data(conn, data)
        except cv2.error as e:
            logging.error(f"Error encoding image: {e}")

    def prepare_data(self, status: str, x_value: int, y_value: int, img_base64: str, emotion: List[List[str]]):
        return {
            'status': status,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'coordinates': [x_value, y_value],
            'datetime': datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
            'emotions': emotion
        }
    
    def encode_image(self, img):
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def client_handler(self, conn, addr):
        self.active_clients += 1
        try:
            while True:
                if not self.process_frame(conn):
                    break
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
        while self.is_active:
            try:
                conn, addr = self.server_socket.accept()
                self.reset_inactivity_timer()
                self.logger.info(f"Client {addr} connected. Total clients: {self.active_clients}")
                thread = threading.Thread(target=self.client_handler, args=(conn, addr))
                thread.start()
            except Exception as e:
                print(f"server error: {e}")
                #self.logger(f"server error: {e}")
                self.close_resources()
                time.sleep(5)

def main():
    server = GazeTrackingServer()
    server.run()

if __name__ == '__main__':
    main()