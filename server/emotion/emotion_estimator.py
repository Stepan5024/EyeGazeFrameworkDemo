import cv2
import torch
from torchvision import transforms
from server.models.emotionModel import EmotionModel

class EmotionRecognizer:
    def __init__(self):
        # Load the model and set it to evaluation mode
        self.model = EmotionModel()
        self.model.load_state_dict(torch.load(r'C:\Users\bokar\Documents\EyeGazeFrameworkDemo\resources\emotion\model.pth'))
        self.model.eval()
        
        # Emotion dictionary
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        # Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(r'C:\Users\bokar\Documents\EyeGazeFrameworkDemo\resources\emotion\haarcascade_frontalface_default.xml')

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
