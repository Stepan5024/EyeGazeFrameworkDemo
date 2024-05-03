import cv2
import onnxruntime as ort
import numpy as np

emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_frame(frame, size=(48, 48)):
    # Convert RGB to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize and normalize the grayscale image
    gray_frame = cv2.resize(gray_frame, size)
    gray_frame = gray_frame.astype(np.float32) / 255.0

    # Expand dimensions to add channel and batch dimensions
    gray_frame = np.expand_dims(gray_frame, axis=0)  # Add batch dimension
    gray_frame = np.expand_dims(gray_frame, axis=0)  # Add channel dimension to make it [1, 1, height, width]
    
    return gray_frame

def main():
    # Загрузка модели ONNX
    model_path = 'C:\\Users\\bokar\\Documents\\EyeGazeFrameworkDemo\\resources\\models\\emotion\\emotion_model_video.onnx'
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

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

            # Выполнение модели
            pred = session.run(None, {input_name: processed_frame})

            emotion_prediction = np.argmax(pred)
            emotion_text = f'Emotion: {emotions[emotion_prediction]}'
            print("Predicted emotion index:", emotion_prediction)
            print("Predicted emotion:", emotions[emotion_prediction])
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
