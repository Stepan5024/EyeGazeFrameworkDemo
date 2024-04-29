import os
import cv2
import numpy as np
import scipy.io
import yaml
import mediapipe as mp

# Функция для загрузки калибровочных параметров из файла YAML
def load_calibration_parameters(filepath):
    with open(filepath, 'r') as file:
        calibration_data = yaml.safe_load(file)
        camera_matrix = np.array(calibration_data['camera_matrix'])
        dist_coeffs = np.array(calibration_data['dist_coeffs']).reshape(-1, 1)
    return camera_matrix, dist_coeffs

curFolder = os.path.dirname(os.path.abspath(__file__))
calibration_file_path = os.path.join(curFolder, 'calibration.yaml')


# Загрузите сохраненные калибровочные параметры
camera_matrix, dist_coeffs = load_calibration_parameters(calibration_file_path)

# Выведите загруженные параметры для проверки
print("Loaded camera matrix:\n", camera_matrix)
print("Loaded distortion coefficients:\n", dist_coeffs)

# Загрузка модели лица
face_model_mat = scipy.io.loadmat('F:\\EyeGazeDataset\\MPIIGaze_original\\6 points-based face model.mat')
face_model = face_model_mat['model']  # Убедитесь, что ключ правильный

# Функция для оценки позы головы
def estimate_head_pose(image_path, face_model, camera_matrix, dist_coeffs):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return None, None

    # Инициализация MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Обработка изображения и получение результатов
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Если лица обнаружены, продолжаем
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Извлечение координат ключевых точек лица (здесь для примера используются точки номер 33, 263, 61, 291, 199 и 0)
        image_points = np.array([
            (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y),  # Nose tip
            (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y),  # Chin
            (face_landmarks.landmark[61].x, face_landmarks.landmark[61].y),  # Left eye left corner
            (face_landmarks.landmark[291].x, face_landmarks.landmark[291].y),  # Right eye right corner
            (face_landmarks.landmark[199].x, face_landmarks.landmark[199].y),  # Left Mouth corner
            (face_landmarks.landmark[0].x, face_landmarks.landmark[0].y)   # Right mouth corner
        ], dtype="double")

        # Преобразование координат точек из относительных в абсолютные пиксели
        image_points *= np.array([image.shape[1], image.shape[0]])

        print(f"\nface_model {face_model}\n")
        print(f"\ncamera_matrix {camera_matrix}\n")
        print(f"\nimage_points {image_points}\n")
        
        print(f"\ndist_coeffs {dist_coeffs}\n")
        # Решение PnP-задачи для оценки позы головы
        success, rotation_vector, translation_vector = cv2.solvePnP(face_model, image_points, camera_matrix, dist_coeffs)
        # Опционально, преобразование вектора вращения в матрицу вращения
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        print("Rotation Matrix:\n", rotation_matrix)
        # Если оценка прошла успешно, возвращаем векторы вращения и перемещения
        if success:
            return rotation_vector, translation_vector
        else:
            print("Head pose estimation was not successful.")
            return None, None
    else:
        print("No faces found in the image.")
        return None, None



# Пример использования
image_path = r'F:\EyeGazeDataset\MPIIFaceGaze_by_author\p00\day01\0005.jpg'
# Вызов функции оценки позы головы
rotation_vector, translation_vector = estimate_head_pose(image_path, face_model, camera_matrix, dist_coeffs)

if rotation_vector is not None and translation_vector is not None:
    print(f"rotation_vector {rotation_vector} translation_vector {translation_vector}\n")
else:
    print(f"Empty rotation_vector or translation_vector")