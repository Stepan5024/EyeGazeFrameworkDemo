import cv2
import numpy as np
import mediapipe as mp
import scipy

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Камерные параметры
# Объявление матрицы камеры
camera_matrix = np.array([[994.73532636,   0.        , 624.66344095],
                          [  0.        , 998.16646784, 364.08742557],
                          [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

# Объявление коэффициентов дисторсии
dist_coeffs = np.array([[-0.16321888,  0.66783406, -0.00121854, -0.00303158, -1.02159927]], dtype=np.float32).reshape(-1, 1)

face_model_mat = scipy.io.loadmat('F:\\EyeGazeDataset\\MPIIGaze_original\\6 points-based face model.mat')
face_model = face_model_mat['model']
face_model = face_model.astype(np.float32)
# Загрузите вашу 3D модель лица или определите координаты вручную
# Например, точки могут быть определены в метрической системе (в миллиметрах или сантиметрах)
face_model_3d = face_model.astype(np.float32)

face_model_3d = np.array([
    [-45.096767, -21.312859,  21.312859,  45.096767, -26.299578,  26.299578],
    [-0.483773,   0.483773,   0.483773,  -0.483773,  68.595032,  68.595032],
    [ 2.397030,  -2.397030,  -2.397030,   2.397030,  -0.000000,  -0.000000]
], dtype=np.float32).T  # Транспонируем для соответствия формату [n, 3]

def estimate_head_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        image_points = np.array([
            (landmarks[33].x * w, landmarks[33].y * h),  # Левый угол левого глаза
            (landmarks[133].x * w, landmarks[133].y * h),  # Правый угол левого глаза
            (landmarks[362].x * w, landmarks[362].y * h),  # Правый угол правого глаза
            (landmarks[263].x * w, landmarks[263].y * h),  # Правый угол правого глаза
            (landmarks[61].x * w, landmarks[61].y * h),  # Левый угол рта
            (landmarks[291].x * w, landmarks[291].y * h),  # Правый угол рта
        ], dtype=np.float32)
        print(f"face_model_3d {face_model_3d}\n")
        print(f"image_points {image_points}\n")
        
        success, rotation_vector, translation_vector = cv2.solvePnP(face_model_3d, 
                                                                    image_points, 
                                                                    camera_matrix, 
                                                                    dist_coeffs)
        
        if success:
            head_translation = translation_vector
            head_rotation = rotation_vector
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            return rotation_vector, rotation_matrix, translation_vector
        else:
            return None, None, None
    else:
        return None, None, None


image = cv2.imread(r'F:\EyeGazeDataset\MPIIFaceGaze_by_author\p00\day01\0005.jpg')
rotation_vector, rotation_matrix, translation_vector = estimate_head_pose(image)
if rotation_vector is not None:
    print("Rotation Vector:\n", rotation_vector)
    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", translation_vector)
else:
    print("No faces detected or solvePnP failed.")
