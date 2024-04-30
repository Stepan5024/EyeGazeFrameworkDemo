import collections
import cv2
from server.processors.mpii_face_gaze_preprocessing import normalize_single_image
from server.utils.camera_utils import gaze_2d_to_3d, get_face_landmarks_in_ccs, get_point_on_screen, plane_equation, ray_plane_intersection
import numpy as np
from server.models.face_model import face_model_all
import mediapipe as mp
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2


class GazeCNN():

    def __init__(self, model, camera_matrix, dist_coefficients, device):
        self.model = model
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coefficients
        self.device = device

        plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
        self.plane_w = plane[0:3]
        self.plane_b = plane[3]
        self.monitor_mm = (400, 250)
        self.monitor_pixels = (1700, 800)

        self.smoothing_buffer = collections.deque(maxlen=3)
        self.rvec_buffer = collections.deque(maxlen=3)
        self.tvec_buffer = collections.deque(maxlen=3)
        self.gaze_vector_buffer = collections.deque(maxlen=10)
        self.gaze_points = collections.deque(maxlen=64)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

        self.landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
        self.face_model = np.asarray([face_model_all[i] for i in self.landmarks_ids])

        self.plot_3d_scene = None
        self.x = 0
        self.y = 0
    
    def get_x(self):
        """Метод для получения значения x."""
        return int(self.x)

    def get_y(self):
        """Метод для получения значения y."""
        return int(self.y)
    
    def draw_laser_pointer(self, point_on_screen, monitor_pixels, 
                           face_model_all_transformed, gaze_points, frame_idx, plot_3d_scene,
                            face_center, gaze_vector, result, 
                           visualize_laser_pointer=True, visualize_3d=True):
        if visualize_laser_pointer:
                    display = np.ones((monitor_pixels[1], monitor_pixels[0], 3), np.float32)
                    gaze_points.appendleft(point_on_screen)
                    """for idx in range(1, len(gaze_points)):
                        thickness = round((len(gaze_points) - idx) / len(gaze_points) * 5) + 1
                        #cv2.line(display, gaze_points[idx - 1], gaze_points[idx], (0, 0, 255), thickness)
                    if frame_idx % 2 == 0:
                        #cv2.imshow(self.WINDOW_NAME, display)"""
        if visualize_3d and plot_3d_scene is not None:
            plot_3d_scene.plot_face_landmarks(face_model_all_transformed)
            plot_3d_scene.plot_center_point(face_center, gaze_vector)
            plot_3d_scene.plot_point_on_screen(result)
            plot_3d_scene.update_canvas()


    def calculate_gaze_point(self, frame):
        default_return = (None, None, None, None, None)
        rvec, tvec = None, None
        if frame is None:
            return default_return
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            # head pose estimation
            #landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
    
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] 
                                         for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in self.landmarks_ids])
            self.smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(self.smoothing_buffer).mean(axis=0)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(self.face_model, 
                                                              face_landmarks,
                                                               self.camera_matrix, 
                                                               self.dist_coefficients, 
                                                               rvec=rvec, tvec=tvec, 
                                                               useExtrinsicGuess=True, 
                                                               flags=cv2.SOLVEPNP_EPNP)  # Initial fit
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(self.face_model, 
                                                   face_landmarks, 
                                                   self.camera_matrix, 
                                                   self.dist_coefficients, 
                                                   rvec=rvec, tvec=tvec, 
                                                   useExtrinsicGuess=True, 
                                                   flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy
            self.rvec_buffer.append(rvec)
            rvec = np.asarray(self.rvec_buffer).mean(axis=0)
            self.tvec_buffer.append(tvec)
            tvec = np.asarray(self.tvec_buffer).mean(axis=0)
            # data preprocessing
            (face_model_transformed, 
             face_model_all_transformed) = get_face_landmarks_in_ccs(self.camera_matrix,
                self.dist_coefficients, frame.shape, results, self.face_model, 
                face_model_all, self.landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # center eye
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # center eye
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))
            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, 
                                                               rvec, None, 
                                                               left_eye_center, self.camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb,
                                                                 rvec, None, 
                                                                 right_eye_center, self.camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, 
                                                                         rvec, None, 
                                                                         face_center, self.camera_matrix, is_eye=False)
  
            transform = A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(self.device)
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(self.device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(self.device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(self.device)
            # prediction
            output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
            print(f"output {output}")
            print(f"любая модель которая предсказывает pitch and yaw")
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)
            self.gaze_vector_buffer.append(gaze_vector)
            gaze_vector = np.asarray(self.gaze_vector_buffer).mean(axis=0)
            print(f"gaze_vector {gaze_vector}\n")
            print(f"gaze_vector_3d_normalized {gaze_vector_3d_normalized}\n")
            # gaze vector to screen
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, self.plane_w, self.plane_b)
            point_on_screen = get_point_on_screen(self.monitor_mm, self.monitor_pixels, result)
            #print(f"point_on_screen {point_on_screen}")
            self.x = point_on_screen[0]
            self.y = point_on_screen[1]
            return point_on_screen, face_model_all_transformed, face_center, gaze_vector, result

        return default_return

    