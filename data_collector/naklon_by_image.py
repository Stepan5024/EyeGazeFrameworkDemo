from typing import Tuple
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np
import time

from server.models.face_model import get_face_model
from mpl_toolkits.mplot3d import Axes3D


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Загрузка изображения из файла
image = cv2.imread(r'F:\EyeGazeDataset\MPIIFaceGaze_by_author\p00\day01\0005.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if image is None:
    print("Image not found.")
    exit(0)

face_model_all = get_face_model()
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])  # fix axis
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])
results = face_mesh.process(image)
image.flags.writeable = True

img_h, img_w, img_c = image.shape
face_3d = []
face_2d = []
target_point_on_screen = (476, 758)
monitor_pixels = (1920, 1080)
screen_width_pixel = monitor_pixels[0]
screen_height_pixel = monitor_pixels[1]
monitor_mm = (400, 250)
screen_height_mm_offset: int = monitor_mm[1]


# Камерные параметры
# Объявление матрицы камеры
camera_matrix = np.array([[994.73532636,   0.        , 624.66344095],
                          [  0.        , 998.16646784, 364.08742557],
                          [  0.        ,   0.        ,   1.        ]], dtype=np.float32)

# Объявление коэффициентов дисторсии
dist_coeffs = np.array([[-0.16321888,  0.66783406, -0.00121854, -0.00303158, -1.02159927]], dtype=np.float32).reshape(-1, 1)



def setup_figure() -> Tuple:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-400, 400)
    ax.set_ylim(-100, 700)
    ax.set_zlim(-10, 800 - 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax

def plot_screen(ax, screen_width_mm, screen_height_mm, screen_height_mm_offset) -> None:
    ax.plot(0, 0, 0, linestyle="", marker="o", color='#1f77b4', label='webcam')

    screen_x = [-screen_width_mm / 2, screen_width_mm / 2]
    screen_y = [screen_height_mm_offset, screen_height_mm + screen_height_mm_offset]
    ax.plot(
        [screen_x[0], screen_x[1], screen_x[1], screen_x[0], screen_x[0]],
        [screen_y[0], screen_y[0], screen_y[1], screen_y[1], screen_y[0]],
        [0, 0, 0, 0, 0],
        color='#ff7f0e',
        label='screen'
    )


def plot_target_on_screen(ax, point_on_screen_px, monitor_mm, monitor_pixels, screen_height_mm_offset):
    screen_width_ratio = monitor_mm[0] / monitor_pixels[0]
    screen_height_ratio = monitor_mm[1] / monitor_pixels[1]

    point_on_screen_mm = (monitor_mm[0] / 2 - point_on_screen_px[0] * screen_width_ratio, point_on_screen_px[1] * screen_height_ratio + screen_height_mm_offset)
    ax.plot(point_on_screen_mm[0], point_on_screen_mm[1], 0, linestyle="", marker="X", color='#9467bd', label='target on screen')
    return point_on_screen_mm[0], point_on_screen_mm[1], 0

def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, image, results):
    """
    Fit `face_model` onto `face_landmarks` using `solvePnP`.

    :param camera_matrix: camera intrinsic matrix
    :param dist_coefficients: distortion coefficients
    :param shape: image shape
    :param results: output of MediaPipe FaceMesh
    :return: full face model in the camera coordinate system
    """
    height, width, _ = image
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks



def plot_face_landmarks(ax, face_model_all_transformed):
    ax.plot(face_model_all_transformed[0, :], face_model_all_transformed[1, :], face_model_all_transformed[2, :], linestyle="", marker="o", color='#7f7f7f', markersize=1, label='face landmarks')


def plot_eye_to_target_on_screen_line(ax, face_model_all_transformed, point_on_screen_3d):
    eye_center = (face_model_all_transformed[:, 33] + face_model_all_transformed[:, 133]) / 2
    ax.plot([point_on_screen_3d[0], eye_center[0]], [point_on_screen_3d[1], eye_center[1]], [point_on_screen_3d[2], eye_center[2]], color='#2ca02c', label='right eye gaze vector')

    eye_center = (face_model_all_transformed[:, 263] + face_model_all_transformed[:, 362]) / 2
    ax.plot([point_on_screen_3d[0], eye_center[0]], [point_on_screen_3d[1], eye_center[1]], [point_on_screen_3d[2], eye_center[2]], color='#d62728', label='left eye gaze vector')


def fix_qt_cv_mismatch():
    import os
    for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]

start = time.time()
def calculate_eye_centers(rotation_vector, translation_vector):
    face_model_3d = face_model_all #get_face_model()
    eye_indices = [33, 263]  # Indices for eyes in the face model
    eye_points_3d = face_model_3d[eye_indices]
    eye_centers = eye_points_3d.mean(axis=0)
    return eye_centers

def estimate_distance(known_width, focal_length_x, focal_length_y, observed_width_pixels):
    """
    Оценка расстояния до объекта с использованием метода треугольников.

    :param known_width: известная ширина объекта в тех же единицах, что и расстояние, которое нужно оценить.
    :param focal_length_x: фокусное расстояние камеры по оси X.
    :param focal_length_y: фокусное расстояние камеры по оси Y.
    :param observed_width_pixels: ширина объекта в пикселях на изображении.
    :return: расстояние до объекта в тех же единицах, что и known_width.
    """
    # Выбор фокусного расстояния
    focal_length = (focal_length_x + focal_length_y) / 2
    
    # Расчет расстояния
    distance = (known_width * focal_length) / observed_width_pixels
    return distance


if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Выделение координат ключевых точек лица
        landmark_coords = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        landmark_coords = [(x * img_w, y * img_h) for x, y in landmark_coords]
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x,y])
                face_3d.append([x, y, lm.z])
            
        # Определение ограничивающего прямоугольника
        x_coords, y_coords = zip(*landmark_coords)
        bbox_x_min = min(x_coords)
        bbox_x_max = max(x_coords)
        bbox_y_min = min(y_coords)
        bbox_y_max = max(y_coords)

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        local_lenght = 1 * img_w

        
        cam_matrix =  np.array([ [local_lenght, 0, img_h / 2],
                                 [0, local_lenght, img_w / 2],
                                 [0, 0 , 1]])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        succes, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        head_rotation = rot_vec
        head_translation = trans_vec
        head_pose_3D = (head_rotation, head_translation)
        rmat, jac = cv2.Rodrigues(rot_vec) # rot_mat
        center_of_eyes = calculate_eye_centers(rot_vec, trans_vec)
        print(f"face_3d {face_3d}\n")
        print(f"center_of_eyes {center_of_eyes}\n")
        # Проекция 2D точки обратно в 3D пространство
        #screen_point_3D = cv2.undistortPoints(np.array([[[target_point_on_screen[0], target_point_on_screen[1]]]], dtype=np.float32), camera_matrix, dist_coeffs)
        #screen_point_3D = np.dot(np.linalg.inv(camera_matrix), np.concatenate((screen_point_3D[0][0], [1])))
        # Нормализация направления взгляда
        #gaze_direction_3D = screen_point_3D - np.dot(rmat, center_of_eyes) - trans_vec
        # Использование undistortPoints для получения нормализованных координат точки взгляда на экране
        screen_point_2D = np.array([[[target_point_on_screen[0], target_point_on_screen[1]]]], dtype=np.float32)
        screen_point_3D = cv2.undistortPoints(screen_point_2D, camera_matrix, dist_coeffs, P=None, R=None)
        
        # Преобразование из нормализованных координат в мировые координаты
        screen_point_3D_homogeneous = np.dot(np.linalg.inv(camera_matrix), np.concatenate((screen_point_3D[0][0], [1])))
        screen_point_3D_world = np.dot(rmat.T, (screen_point_3D_homogeneous - trans_vec.reshape(3)))  # Убедитесь, что trans_vec имеет форму (3,)

        #screen_point_3D = cv2.undistortPoints(np.array([[[target_point_on_screen[0], target_point_on_screen[1]]]], dtype=np.float32), camera_matrix, dist_coeffs, P=None, R=None)
        #screen_point_3D_homogeneous = np.dot(np.linalg.inv(camera_matrix), np.concatenate((screen_point_3D[0][0], [1])))
        #screen_point_3D_world = np.dot(rmat.T, (screen_point_3D_homogeneous - trans_vec))  # Преобразование в мировые координаты


        gaze_direction_3D = screen_point_3D_world.reshape(3) - center_of_eyes.reshape(3)
        print(f"gaze_direction_3D dim {gaze_direction_3D.ndim}")
        print("3D Gaze Target Position:", gaze_direction_3D)
        gaze_direction_3D_normalized = gaze_direction_3D / np.linalg.norm(gaze_direction_3D)

        # Известная ширина лица в мм
        known_face_width = 190.0  # реальная ширина лица в миллиметрах
        face_width_in_pixels = bbox_y_max - bbox_y_min

        focal_length_x = camera_matrix[0, 0]
        focal_length_y = camera_matrix[1, 1]

        distance_to_target = estimate_distance(known_face_width, focal_length_x, focal_length_y, face_width_in_pixels)  # Примерное расстояние до объекта взгляда в миллиметрах
        print("Estimated distance to the target (mm):", distance_to_target)
        # Проверяем, является ли center_of_eyes одномерным массивом
        if center_of_eyes.ndim != 1:
            raise ValueError("center_of_eyes должен быть одномерным массивом (1D)")

        # Проверяем, является ли gaze_direction_3D_normalized одномерным массивом
        if gaze_direction_3D_normalized.ndim != 1:
            raise ValueError("gaze_direction_3D_normalized должен быть одномерным массивом (1D)")

        #gaze_target_3D = center_of_eyes + gaze_direction_3D_normalized  * distance_to_target
        #gaze_target_3D = center_of_eyes + gaze_direction_3D_normalized
        #print("3D Gaze Target Position:", gaze_target_3D)
        
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif y > 10:
            text = "Looking Up"
        else:
            text = "Forward"
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
        cv2.putText(image, "x: " + str(np.round(x,2)), (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.putText(image, "y: " + str(np.round(y,2)), (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.putText(image, "z: " + str(np.round(z,2)), (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

end = time.time()
totalTime = end - start 
fps = 1 / totalTime
cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
cv2.imshow('Head Pose Estimation', image)
cv2.waitKey(0)
    
cv2.destroyAllWindows()
