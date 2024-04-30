import glob
import os
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import skimage.io
from tqdm import tqdm



def get_matrices(camera_matrix: np.ndarray, distance_norm: int, center_point: np.ndarray, focal_norm: int, head_rotation_matrix: np.ndarray, image_output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rotation, scaling and transformation matrix.

    :param camera_matrix: intrinsic camera matrix
    :param distance_norm: normalized distance of the camera
    :param center_point: position of the center in the image
    :param focal_norm: normalized focal length
    :param head_rotation_matrix: rotation of the head
    :param image_output_size: output size of the output image
    :return: rotation, scaling and transformation matrix
    """
    # normalize image
    distance = np.linalg.norm(center_point)  # actual distance between center point and original camera
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [focal_norm, 0, image_output_size[0] / 2],
        [0, focal_norm, image_output_size[1] / 2],
        [0, 0, 1.0],
    ])

    scaling_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    forward = (center_point / distance).reshape(3)
    down = np.cross(forward, head_rotation_matrix[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    rotation_matrix = np.asarray([right, down, forward])
    transformation_matrix = np.dot(np.dot(cam_norm, scaling_matrix), np.dot(rotation_matrix, np.linalg.inv(camera_matrix)))

    return rotation_matrix, scaling_matrix, transformation_matrix


def equalize_hist_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of a RGB image.

    :param rgb_img: RGB image
    :return: equalized RGB image
    """
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)  # convert from RGB color-space to YCrCb
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])  # equalize the histogram of the Y channel
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)  # convert back to RGB color-space from YCrCb
    return equalized_img


def normalize_single_image(image: np.ndarray, head_rotation, gaze_target: np.ndarray, center_point: np.ndarray, camera_matrix: np.ndarray, is_eye: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The normalization process of a single image, creates a normalized eye image or a face image, depending on `is_eye`.

    :param image: original image
    :param head_rotation: rotation of the head
    :param gaze_target: 3D target of the gaze
    :param center_point: 3D point on the face to focus on
    :param camera_matrix: intrinsic camera matrix
    :param is_eye: if true the `distance_norm` and `image_output_size` values for the eye are used
    :return: normalized image, normalized gaze and rotation matrix
    """
    # normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 500 if is_eye else 1600  # normalized distance between eye and camera
    image_output_size = (96, 64) if is_eye else (96, 96)  # size of cropped eye image

    # compute estimated 3D positions of the landmarks
    if gaze_target is not None:
        gaze_target = gaze_target.reshape((3, 1))

    head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
    rotation_matrix, scaling_matrix, transformation_matrix = get_matrices(camera_matrix, distance_norm, center_point, focal_norm, head_rotation_matrix, image_output_size)

    img_warped = cv2.warpPerspective(image, transformation_matrix, image_output_size)  # image normalization
    img_warped = equalize_hist_rgb(img_warped)  # equalizes the histogram (normalization)

    if gaze_target is not None:
        # normalize gaze vector
        gaze_normalized = gaze_target - center_point  # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        gaze_normalized = np.dot(rotation_matrix, gaze_normalized)
        gaze_normalized = gaze_normalized / np.linalg.norm(gaze_normalized)
    else:
        gaze_normalized = np.zeros(3)

    return img_warped, gaze_normalized.reshape(3), rotation_matrix

def main(input_path: str, output_path: str):
    face_model = scipy.io.loadmat(f'{input_path}/6 points-based face model.mat')['model']
    person = "p00"
    day = "day01"
    person_path = os.path.join(input_path, person)
    day_path = os.path.join(person_path, day)
    image_path = os.path.join(day_path, "0005.jpg")

    # Загрузка аннотаций
    annotations = pd.read_csv(os.path.join(person_path, 'p00.txt'), sep=' ', header=None, index_col=0)
    annotation = annotations.loc['day01/0005.jpg']

    camera_matrix = scipy.io.loadmat(f'{person_path}/Calibration/Camera.mat')['cameraMatrix']
    print(f"camera_matrix {camera_matrix}")
    screen_size = scipy.io.loadmat(f'{person_path}/Calibration/screenSize.mat')
    screen_width_pixel = screen_size["width_pixel"].item()
    screen_height_pixel = screen_size["height_pixel"].item()
        
    # Загрузка изображения
    img = skimage.io.imread(image_path)

    # Печать аннотации для проверки
    #print(f"Annotation: \n{annotation}\n")
    first_row_series = annotations.iloc[0]
    for i, value in first_row_series.items():
        print(f"{i} {value}")
    height, width, _ = img.shape
    data = defaultdict(list)
    """В аннотации датасета MPIIFaceGaze следующие позиции означают:

Размер 0: путь и имя файла изображения. 3
Размер 1–2: координаты взгляда на экране в пикселях. Фактический размер экрана можно найти в папке «Калибровка». 3
Размер 3–14: положение (x, y) шести ориентиров лица — четырёх углов глаз и двух углов рта. 3
Размер 15–20: оценённая 3D-позиция головы в системе координат камеры на основе 3D-модели лица из шести точек, вращение и перевод. 3
Размер 21–23 (fc): центр лица в системе координат камеры, который представляет собой усреднённое 3D-положение шести фокальных ориентиров лица. 3
Размер 24–26 (gt): 3D-положение цели взгляда в системе координат камеры. Направление взгляда можно рассчитать как gt — fc. 3
Размер 27: какой глаз (левый или правый) используется для подмножества оценки. 3
"""
    print(f"annotation len {len(annotation)}\n")
    print(f"annotation[15:17] \n{annotation[14:17]}\n\nend\n")
    head_rotation = annotation[14:17].to_numpy().reshape(-1).astype(float)  # 3D head rotation based on 6 points-based 3D face model
    print(f"annotation[18:20] \n{annotation[17:20]}\n\nend\n")
    head_translation = annotation[17:20].to_numpy().reshape(-1).astype(float)  # 3D head translation based on 6 points-based 3D face model
    print(f"annotation[24:26] \n{annotation[23:26]}\n\nend\n")
    gaze_target_3d = annotation[23:26].to_numpy().reshape(-1).astype(float)  # 3D gaze target position related to camera (on the screen)
    print(f"head_rotation \n{head_rotation}\n")
    print(f"head_translation \n{head_translation}\n")
    print(f"gaze_target_3d \n{gaze_target_3d}\n")
    head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
    face_landmarks = np.dot(head_rotation_matrix, face_model) + head_translation.reshape((3, 1))  # 3D positions of facial landmarks
    left_eye_center = 0.5 * (face_landmarks[:, 2] + face_landmarks[:, 3]).reshape((3, 1))  # center eye
    right_eye_center = 0.5 * (face_landmarks[:, 0] + face_landmarks[:, 1]).reshape((3, 1))  # center eye
    face_center = face_landmarks.mean(axis=1).reshape((3, 1))
    
    img_warped_left_eye, _, _ = normalize_single_image(img, head_rotation, None, left_eye_center, camera_matrix)
    img_warped_right_eye, _, _ = normalize_single_image(img, head_rotation, None, right_eye_center, camera_matrix)
    img_warped_face, gaze_normalized, rotation_matrix = normalize_single_image(img, head_rotation, gaze_target_3d, face_center, camera_matrix, is_eye=False)
    print(f"gaze location {list(annotation[:2])}\n")
    gaze_pitch = np.arcsin(-gaze_normalized[1])
    gaze_yaw = np.arctan2(-gaze_normalized[0], -gaze_normalized[2])
    base_file_name = f'{person}/{day}/'
    #pathlib.Path(f"{output_path}/{base_file_name}").mkdir(parents=True, exist_ok=True)
    base_file_name += f'{image_path.split("/")[-1][:-4]}'
    #skimage.io.imsave(f"{output_path}/{base_file_name}-left_eye.png", img_warped_left_eye.astype(np.uint8), check_contrast=False)
    #skimage.io.imsave(f"{output_path}/{base_file_name}-right_eye.png", img_warped_right_eye.astype(np.uint8), check_contrast=False)
    #skimage.io.imsave(f"{output_path}/{base_file_name}-full_face.png", img_warped_face.astype(np.uint8), check_contrast=Fals
    data['file_name_base'].append(base_file_name)
    data['gaze_pitch'].append(gaze_pitch)
    data['gaze_yaw'].append(gaze_yaw)
    data['gaze_location'].append(list(annotation[:2]))
    data['screen_size'].append([screen_width_pixel, screen_height_pixel])



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default='F:\\EyeGazeDataset\\MPIIFaceGaze_by_author')
    parser.add_argument("--output_path", type=str, default='./data')
    args = parser.parse_args()

    main(args.input_path, args.output_path)
