import logging
import os
import pathlib
import sys
import cv2
import torch
from torchvision import transforms
import yaml

from logger import create_logger
from server.models.emotionModel import EmotionModel

class EmotionRecognizer:
    def __init__(self):
        self.readConfig(os.path.join('configs', 'server.yaml'))
        self.initLogger()
        # Load the model and set it to evaluation mode
        self.model = EmotionModel()
        relative_path_emotion_model: str = os.path.join("resources",  "emotion", "model.pth")
        abs_path: str = self.resource_path(relative_path_emotion_model)
        self.model.load_state_dict(torch.load(abs_path))
        self.model.eval()
        
        # Emotion dictionary
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        # Haar Cascade for face detection
        relative_path_haar_model: str = os.path.join("resources", "emotion", "haarcascade_frontalface_default.xml")
        abs_path: str = self.resource_path(relative_path_haar_model)
        self.face_cascade = cv2.CascadeClassifier(abs_path)
    def readConfig(self, path: str) -> None:
        path_to_config: str = self.resource_path(path)
        self.configs: dict = self.load_config(path_to_config)

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def initLogger(self) -> None:
        """Инициализация логера"""
        rev_path: str = os.path.join("logs", "server")
        abs_path: pathlib.Path = pathlib.Path(self.resource_path(rev_path))
        self.logger: logging.Logger = create_logger("EyeGazeServer", abs_path, 'server_log.txt')
    
    def resource_path(self, relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        try:
            # временная папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    def predict_emotions(self, image):
        """Take a single image as input and return the top 2 most likely emotions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        top_emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            tensor = transforms.ToTensor()(roi_gray).unsqueeze(0)

            # Get predictions
            predictions = self.model(tensor)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            top2_prob, top2_idx = torch.topk(probabilities, 2)

            # Extracting the top 2 emotions
            emotions = [self.emotion_dict[idx.item()] for idx in top2_idx[0]]
            top_emotions.append(emotions)

        return top_emotions
