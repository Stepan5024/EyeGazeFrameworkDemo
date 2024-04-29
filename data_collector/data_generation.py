import os
import cv2
import numpy as np
import scipy.io

# Предполагается, что camera_matrix и dist_coeffs загружаются или определяются где-то ранее
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Замените fx, fy, cx, cy на реальные значения
dist_coeffs = np.zeros((4,1))  # Если у камеры нет значительных искажений

# Загрузка модели лица
face_model_mat = scipy.io.loadmat('F:\\EyeGazeDataset\\MPIIGaze_original\\6 points-based face model.mat')
face_model = face_model_mat['model']  # Убедитесь, что ключ правильный

def estimate_head_pose(image_path, face_model, camera_matrix, dist_coeffs):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # Детектирование лицевых точек
    # Предполагается, что вы используете какой-то метод для обнаружения ключевых точек лица, например dlib или другой
    landmarks = detect_landmarks(image)
    if landmarks is None:
        print("Landmarks not detected.")
        return

    # Используемые точки из модели лица
    image_points = np.array([
        landmarks[0],  # Левый угол левого глаза
        landmarks[1],  # Правый угол левого глаза
        landmarks[2],  # Левый угол правого глаза
        landmarks[3],  # Правый угол правого глаза
        landmarks[4],  # Левый угол рта
        landmarks[5]   # Правый угол рта
    ], dtype="double")

    # Получаем матрицу вращения и вектор трансляции
    success, rotation_vector, translation_vector = cv2.solvePnP(face_model, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        print("Head pose could not be estimated.")
        return

    # Печать результатов
    print("Rotation Vector:\n", rotation_vector)
    print("Translation Vector:\n", translation_vector)

    # Опционально, преобразование вектора вращения в матрицу вращения
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("Rotation Matrix:\n", rotation_matrix)

# Функция для детектирования ключевых точек (заглушка)
def detect_landmarks(image):
    # Здесь должен быть ваш код для детектирования ключевых точек
    return np.array([
        [100, 200],  # Примерные координаты, замените на реальные значения
        [150, 200],
        [200, 250],
        [250, 250],
        [120, 300],
        [220, 300]
    ])

# Пример использования
image_path = 'path_to_image.jpg'
estimate_head_pose(image_path, face_model, camera_matrix, dist_coeffs)
