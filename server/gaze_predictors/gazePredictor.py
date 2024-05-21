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


class GazePredictor():

    def __init__(self, model, camera_matrix, dist_coefficients):
        self.model = model
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coefficients

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
    
    def get_pitch(self):
        """Метод для получения значения pitch."""
        return self.gaze_pitch
    
    def get_yaw(self):
        """Метод для получения значения yaw."""
        return self.gaze_yaw

    def draw_laser_pointer(self, point_on_screen, monitor_pixels, 
                           face_model_all_transformed, gaze_points, frame_idx, plot_3d_scene,
                            face_center, gaze_vector, result, 
                           visualize_laser_pointer=True, visualize_3d=True):
        if visualize_laser_pointer:
                    display = np.ones((monitor_pixels[1], monitor_pixels[0], 3), np.float32)
                    gaze_points.appendleft(point_on_screen)
        if visualize_3d and plot_3d_scene is not None:
            plot_3d_scene.plot_face_landmarks(face_model_all_transformed)
            plot_3d_scene.plot_center_point(face_center, gaze_vector)
            plot_3d_scene.plot_point_on_screen(result)
            plot_3d_scene.update_canvas()

    def check_frame_validity(self, frame):
        if frame is None:
            return None, (None, None, None, None, None)
        else:
            return frame, None
    def preprocess_image(self, frame):
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        return image_rgb, height, width

    def detect_face_landmarks(self, face_mesh, image_rgb):
        return face_mesh.process(image_rgb)

    def extract_landmarks(self, results, width, height, landmarks_ids):
        if results.multi_face_landmarks:
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] 
                                         for landmark in results.multi_face_landmarks[0].landmark])
            return np.asarray([face_landmarks[i] for i in landmarks_ids])
        else:
            return None

    def smooth_landmarks(self, face_landmarks, smoothing_buffer):
        smoothing_buffer.append(face_landmarks)
        return np.mean(smoothing_buffer, axis=0)

    def estimate_head_pose(self, face_model, face_landmarks, camera_matrix, dist_coefficients, rvec, tvec):
        _, rvec, tvec, _ = cv2.solvePnPRansac(face_model, face_landmarks,
                                              camera_matrix, dist_coefficients,
                                              rvec=rvec, tvec=tvec,
                                              useExtrinsicGuess=True,
                                              flags=cv2.SOLVEPNP_EPNP)
        for _ in range(10):
            _, rvec, tvec = cv2.solvePnP(face_model, face_landmarks,
                                         camera_matrix, dist_coefficients,
                                         rvec=rvec, tvec=tvec,
                                         useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
        return rvec, tvec

    def normalize_and_transform_images(self, face_center, image_rgb, rvec, camera_matrix):
        img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)
        return img_warped_face, rotation_matrix

    def prepare_images_for_model(self, image, transform, device):
        transformed_image = transform(image=image)["image"].unsqueeze(0).float().to(device)
        return transformed_image

    def predict_gaze(self, model, person_idx, images):
        return model(person_idx, *images).squeeze(0).detach().cpu().numpy()

    def convert_gaze_to_vector(self, gaze_output, rotation_matrix):
        gaze_vector_3d_normalized = gaze_2d_to_3d(gaze_output)
        return np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

    def calculate_screen_coordinates(self, face_center, gaze_vector, plane_w, plane_b):
        result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
        return result
    
    def calculate_gaze_point(self, frame):
        frame, default_return = self.check_frame_validity(frame)
        if default_return:
            return default_return

        image_rgb, height, width = self.preprocess_image(frame)
        results = self.detect_face_landmarks(self.face_mesh, image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = self.extract_landmarks(results, width, height, self.landmarks_ids)
            face_landmarks = self.smooth_landmarks(face_landmarks, self.smoothing_buffer)
            rvec, tvec = self.estimate_head_pose(self.face_model, face_landmarks, self.camera_matrix, self.dist_coefficients, None, None)

            # data preprocessing
            (face_model_transformed, 
             face_model_all_transformed) = get_face_landmarks_in_ccs(self.camera_matrix,
                self.dist_coefficients, frame.shape, results, self.face_model, 
                face_model_all, self.landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # center eye
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # center eye
            
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))  # Assuming face_center calculation as mean of landmarks
            img_warped_face, rotation_matrix = self.normalize_and_transform_images(face_center, image_rgb, rvec, self.camera_matrix)

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, 
                                                               rvec, None, 
                                                               left_eye_center, self.camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb,
                                                                 rvec, None, 
                                                                 right_eye_center, self.camera_matrix)
            

            transform = A.Compose([A.Normalize(), ToTensorV2()])
            device = next(self.model.parameters()).device
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(device)

            full_face_image = self.prepare_images_for_model(img_warped_face, transform, device)
            left_eye_image = self.prepare_images_for_model(img_warped_left_eye, transform, device)
            right_eye_image = self.prepare_images_for_model(img_warped_right_eye, transform, device)

            output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
            self.gaze_pitch, self.gaze_yaw = output[:2]

            gaze_vector = self.convert_gaze_to_vector(output, rotation_matrix)
            result = self.calculate_screen_coordinates(face_center, gaze_vector, self.plane_w, self.plane_b)
            point_on_screen = get_point_on_screen(self.monitor_mm, self.monitor_pixels, result)

            self.x, self.y = point_on_screen[:2]
            return point_on_screen, None, face_center, gaze_vector, result  # Adjust second return value as needed

        return default_return
