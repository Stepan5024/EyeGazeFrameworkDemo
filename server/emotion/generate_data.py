import cv2 as cv
import os
import time
from mtcnn.mtcnn import MTCNN

# saving emotion images
DIR = 'C:\\Users\\bokar\\Documents\\EyeGazeFrameworkDemo\\Resources\\Faces\\Labels'


EMOTIONS = {
    'a': 'Anger',
    'd': 'Disgust',
    'f': 'Fear',
    'h': 'Happiness',
    's': 'Sadness',
    'u': 'Surprise',
    'n': 'Neutrality'
}


def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")

def main():
    # Create directories for all emotions
    for emotion_key, emotion_name in EMOTIONS.items():
        emotion_path = os.path.join(DIR, emotion_name)
        createDir(emotion_path)
    
    face_detector = MTCNN()
    capture = cv.VideoCapture(0)

    while True:
        key = cv.waitKey(20) & 0xFF
        isTrue, img = capture.read()
        if not isTrue:
            break

        # Detect faces
        faces = face_detector.detect_faces(img)
        for face in faces:
            x, y, w, h = face['box']
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)

        # Save image on key press
        for emotion_key, emotion_name in EMOTIONS.items():
            if key == ord(emotion_key):
                timestamp = "{:.1f}".format(time.time())
                file_path = os.path.join(DIR, emotion_name, f'{timestamp}.jpg')
                cv.imwrite(file_path, img)
                print(f"IMAGE SAVED => {file_path}")

        cv.imshow("VIDEO", img)

        # Exit loop if 'q' is pressed
        if key == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
