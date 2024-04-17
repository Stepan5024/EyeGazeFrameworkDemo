import cv2
import mediapipe as mp
print(mp.__file__)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from server.models.eye import Eye
from server.models.point import Point
from server.video_stream.video_stream import VideoStream
from server.calibration.calibration import Calibration
import os
import sys

class MediaPipeProcessor:
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    

    def __init__(self):
        # Determine if the application is frozen (packaged by PyInstaller)
        if getattr(sys, 'frozen', False):
            # The application is frozen
            # Use the temporary folder PyInstaller unpacks to
            bundle_dir = sys._MEIPASS  # Use the correct attribute here
            models_dir = os.path.join(bundle_dir, 'mediapipe', 'models')
            #bundle_dir = sys._MEIPASS
        else:
            # The application is not frozen
            # Directly use the development path relative to this file
            models_dir = os.path.join(os.path.dirname(__file__), 'mediapipe', 'models')
        
        print(f"models_dir {models_dir}")
        
        # Initialize paths to the model files
        face_mesh_model_path = os.path.join(models_dir, 'face_landmark.tflite')
        face_detection_model_path = os.path.join(models_dir, 'face_detection_short_range.tflite')

        # Initialize FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize FaceDetection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        """self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            #model_path="./models/face_landmark.tflite"
        )
        self.mp_face_detection = mp.solutions.face_detection
        #Create an FaceDetector object.
        model_path="./models/face_detection_short_range.tflite"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        detector = vision.FaceDetector.create_from_options(options)

        self.face_detection = detector#self.mp_face_detection.FaceDetection(model_selection=1, 
                               #                                    min_detection_confidence=0.5,
                               #                                    model_path="./models/face_detection_short_range.tflite")
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.eye_left = None # Объект, представляющий левый глаз.
        self.eye_right = None  # Объект, представляющий правый глаз.
        self.calibration = Calibration() # Объект, представляющий данные калибровки для определения размера глаза.


    def process_head_pose(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Get the shape of the frame for future calculations
        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []
        x, y, z = None, None, None
        text = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Specific landmark indices
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1: # Nose tip
                        
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

            # Convert to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix and distance matrix for solvePnP
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0 , 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # SolvePnP to find rotation vector and translation vector
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Convert the angles to degrees
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Determine the direction the person is looking
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Project the 3D nose point to 2D
            """nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            # Draw line from nose to projected point
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            """
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            # Put the direction text and angles on the frame
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, (x, y, z), text
    
    def _analyze(self, frame):
       
         # Преобразует кадр в оттенки серого для использования в детекции лиц.
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Инициализация MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        results = face_detection.process(frame)
        # Проверка наличия лиц в результате
        if results.detections:
            faces_mp = []  # Список для хранения координат лиц
            for detection in results.detections:
                # Получение координат ограничивающего прямоугольника (bounding box) лица
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape  # Размеры изображения
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                faces_mp.append(bbox)

            # Теперь в переменной faces_mp хранится список кортежей с координатами ограничивающих прямоугольников лиц
            # Каждый кортеж представляет собой (x, y, width, height)
            else:
                faces_mp = None  # Лица не обнаружены
        try:
            # Использует предиктор для получения точек landmarks для первого обнаруженного лица.
            # Обработка изображения с помощью Face Mesh
            #face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

            results_face = face_mesh.process(frame)
            # Переменная для хранения ключевых точек лица в формате, аналогичном dlib
            landmarks_mp = []
            # Извлечение ключевых точек лица, если они обнаружены
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        landmarks_mp.append(Point(x, y))
            # Итерация по всем точкам и вывод их координат
            #for i in range(landmarks.num_parts):
            #    point = landmarks.part(i)
            #    print(f"Точка {i}: (x={point.x}, y={point.y})")
             # Инициализирует объекты Eye для левого и правого глаза с использованием landmarks.
            self.eye_left = Eye(frame, landmarks_mp, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks_mp, 1, self.calibration)
            #print(f"eye_left {self.eye_left.pupil.x} eye_right {self.eye_right.pupil.x}")
        except IndexError:
             # В случае отсутствия обнаруженных лиц, устанавливает значения eye_left и eye_right в None.
            print(f"Лица не обнаружены eye_left eye_right {None}")
            self.eye_left = None
            self.eye_right = None

    def draw_eye_contours(self, frame, left_eye_points, right_eye_points):
        # Рисуем контур вокруг левого глаза
        cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=1)
        # Рисуем контур вокруг правого глаза
        cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=1)
    @property
    def pupils_located(self):
        """Проверка расположения зрачков"""
        try:
            # Проверяет, что координаты зрачков обоих глаз являются целыми числами.
       
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            print(f"pupils_located {False}")
             # Возвращает False, если хотя бы для одного зрачка координаты не являются целыми числами
        
            return False
        
    def get_eye_points(self, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, frame_shape):
        # Собираем координаты точек для левого и правого глаз
        left_eye_points = [(int(face_landmarks.landmark[index].x * frame_shape[1]), 
                            int(face_landmarks.landmark[index].y * frame_shape[0])) 
                           for index in LEFT_EYE_INDICES]
        right_eye_points = [(int(face_landmarks.landmark[index].x * frame_shape[1]), 
                             int(face_landmarks.landmark[index].y * frame_shape[0])) 
                            for index in RIGHT_EYE_INDICES]
        return left_eye_points, right_eye_points
    
    def pupil_left_coords(self):
        """Возвращает координаты левого зрачка
         Возвращает координаты левого зрачка относительно исходной точки (origin) левого глаза.
    
        Возвращаемые значения:
        tuple or None: Кортеж с координатами (x, y) левого зрачка,
        или None, если координаты зрачка не были обнаружены.
                       """
        # Проверяет, были ли обнаружены координаты зрачков обоих глаз.
        if self.pupils_located:
             # Рассчитывает абсолютные координаты левого зрачка относительно исходной точки левого глаза.
        
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        else:
        # Возвращает None, если координаты зрачка не были обнаружены.
            return None

    def pupil_right_coords(self):
        """Возвращает координаты правого зрачка
        Возвращает координаты правого зрачка относительно исходной точки (origin) правого глаза.
    
        Возвращаемые значения:
        tuple or None: Кортеж с координатами (x, y) правого зрачка,
                       или None, если координаты зрачка не были обнаружены.
        """
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        
    def process_eyes(self, frame):
        # Преобразование кадра из BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обработка кадра для получения ключевых точек глаз
        results = self.face_mesh.process(rgb_frame)
        annotated_frame = frame.copy()
        self._analyze(annotated_frame)
        if self.pupils_located:
            # Задает цвет линий для выделения зрачков (зеленый).
            color = (0, 255, 0)
             # Получает координаты левого и правого зрачков.
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
             # Рисует линии, обозначающие положение левого и правого зрачков.
            cv2.line(annotated_frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(annotated_frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(annotated_frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(annotated_frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        #

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Получение координат для контуров левого и правого глаза
                # Получаем координаты точек для глаз
 
                left_eye_points, right_eye_points = self.get_eye_points(face_landmarks, 
                                                                        self.LEFT_EYE_INDICES, 
                                                                        self.RIGHT_EYE_INDICES, 
                                                                        frame.shape)
                
                # Рисование контуров глаз
                self.draw_eye_contours(annotated_frame, left_eye_points, right_eye_points)



        return annotated_frame
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face_mesh = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_frame = frame.copy()

        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec
                )
        results_face_detection = self.face_detection.process(rgb_frame)
        detected_faces = []
        # Отрисовка прямоугольников вокруг каждого лица
        if results_face_detection.detections:
            for detection in results_face_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                int(bboxC.width * iw), int(bboxC.height * ih)
                x2, y2 = x1 + w, y1 + h

                # Добавляем координаты прямоугольника в список
                detected_faces.append((x1, y1, x2, y2))

                # Отрисовка прямоугольника вокруг лица
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return annotated_frame, detected_faces
    
    def get_landmarks(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks.append(face_landmarks.landmark)

        return landmarks
    
    def release(self):
        self.face_mesh.close()
        
