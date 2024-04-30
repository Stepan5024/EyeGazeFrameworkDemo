import collections
import os
import cv2
import numpy as np
import scipy.io
import yaml
import mediapipe as mp

from server.models.face_model import get_face_model

# Функция для загрузки калибровочных параметров из файла YAML
def load_calibration_parameters(filepath):
    with open(filepath, 'r') as file:
        calibration_data = yaml.safe_load(file)
        camera_matrix = np.array(calibration_data['camera_matrix'])
        dist_coeffs = np.array(calibration_data['dist_coeffs']).reshape(-1, 1)
        rvecs = np.array(calibration_data['rvecs'])
        tvecs = np.array(calibration_data['tvecs'])
    return camera_matrix, dist_coeffs, rvecs, tvecs

curFolder = os.path.dirname(os.path.abspath(__file__))
calibration_file_path = os.path.join(curFolder, 'calibration.yaml')


# Загрузите сохраненные калибровочные параметры
#camera_matrix, dist_coeffs, rvecs, tvecs = # Объявление матрицы камеры
camera_matrix = np.array([[994.73532636,   0.        , 624.66344095],
                          [  0.        , 998.16646784, 364.08742557],
                          [  0.        ,   0.        ,   1.        ]])

# Объявление коэффициентов дисторсии
dist_coeffs = np.array([[-0.16321888,  0.66783406, -0.00121854, -0.00303158, -1.02159927]])

#load_calibration_parameters(calibration_file_path)

# Выведите загруженные параметры для проверки
print("Loaded camera matrix:\n", camera_matrix)
print("Loaded distortion coefficients:\n", dist_coeffs)

# Загрузка модели лица
face_model_mat = scipy.io.loadmat('F:\\EyeGazeDataset\\MPIIGaze_original\\6 points-based face model.mat')
face_model = face_model_mat['model']  # Убедитесь, что ключ правильный
face_model = face_model.astype(np.float32)

face_model = np.array(face_model, dtype=np.float32)
print(f"face mode {face_model}\n")
face_model_all = get_face_model()
landmarks_ids = [33, 133, 362, 263, 61, 291]  # reye, leye, mouth
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

smoothing_buffer = collections.deque(maxlen=3)

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

        height, width, _ = image.shape

        face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
        face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])
        
        camera_matrix = np.array(camera_matrix, dtype=np.float32)
        dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)
        
        print(f"\nface_model {face_model}\n")
        print(f"\ncamera_matrix {camera_matrix}\n")
        print(f"\nface_landmarks {face_landmarks}\n")

        print(f"\ndist_coeffs {dist_coeffs}\n")
        # Решение PnP-задачи для оценки позы головы
        success, rotation_vector, translation_vector = cv2.solvePnP(face_model,
                                                                    face_landmarks, 
                                                                    camera_matrix, 
                                                                    dist_coeffs)
        if success:
            print("\nRotation Vector:\n", rotation_vector)
            print("\nTranslation Vector:\n", translation_vector)
        else:
            print("\nsolvePnP failed to find a solution.")
        # Опционально, преобразование вектора вращения в матрицу вращения
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        print("\nRotation Matrix:\n", rotation_matrix)
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
    print(f"\nrotation_vector {rotation_vector} \ntranslation_vector {translation_vector}\n")
else:
    print(f"Empty rotation_vector or translation_vector")