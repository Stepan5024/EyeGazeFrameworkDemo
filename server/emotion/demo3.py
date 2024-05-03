import cv2
import torch
import numpy as np
import torch.nn.functional as F

from server.emotion.deep.deep_emotion import Deep_Emotion

# Предполагается, что класс модели Deep_Emotion уже определен в другом месте вашего кода
 # Импортируйте класс модели

emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_frame(frame, size=(48, 48)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, size)
    gray_frame = gray_frame.astype(np.float32) / 255.0
    gray_frame = np.expand_dims(gray_frame, axis=0)  # Add batch dimension
    gray_frame = np.expand_dims(gray_frame, axis=0)  # Add channel dimension to make it [1, 1, height, width]
    return gray_frame

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели PyTorch
    model_path = 'deep_emotion-100-128-0.005.pt'
    model = Deep_Emotion()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Захват видео с веб-камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Предобработка кадра
            processed_frame = preprocess_frame(frame)
            tensor_frame = torch.from_numpy(processed_frame).to(device)

            # Выполнение модели
            with torch.no_grad():
                pred = model(tensor_frame)
                pred = F.softmax(pred, dim=1)  # Примените softmax для получения вероятностей
            emotion_prediction = torch.argmax(pred).item()
            emotion_text = emotions[emotion_prediction]

            # Отображение результата
            cv2.putText(frame, f'Emotion: {emotion_text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
