import cv2
import numpy as np
from scipy.io import loadmat

# Load the face model
face_model_data = loadmat('../MPIIGaze/Data/6 points-based face model.mat')
face_model = face_model_data['model']

# Load the image, annotation, and camera parameters.
img = cv2.imread('../MPIIGaze/Data/Original/p00/day01/0001.jpg')
annotation = np.loadtxt('../MPIIGaze/Data/Original/p00/day01/annotation.txt')
camera_calib_data = loadmat('../MPIIGaze/Data/Original/p00/Calibration/Camera.mat')
camera_matrix = camera_calib_data['cameraMatrix']

# Get head pose
headpose_hr = annotation[0, 29:32]
headpose_ht = annotation[0, 32:35]
hR, _ = cv2.Rodrigues(np.array(headpose_hr))
Fc = hR @ face_model + headpose_ht  # Rotate and translate the face model

# Get the eye centers in the original camera coordinate system
right_eye_center = 0.5 * (Fc[:, 0] + Fc[:, 1])
left_eye_center = 0.5 * (Fc[:, 2] + Fc[:, 3])

# Get the gaze target
gaze_target = annotation[0, 26:29]

# Set the size of the normalized eye image
eye_image_width = 60
eye_image_height = 36

# Normalize the image for the right eye
eye_img, headpose, gaze = normalize_img(img, right_eye_center, hR, gaze_target,
                                        [eye_image_width, eye_image_height],
                                        camera_matrix)

# Display the normalized eye image
cv2.imshow('Normalized Eye Image', eye_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the gaze direction in the camera coordinate system to angles in the polar coordinate system
gaze_theta = np.arcsin(-gaze[1])  # Vertical gaze angle
gaze_phi = np.arctan2(-gaze[0], -gaze[2])  # Horizontal gaze angle

# Convert the head pose to angles in the polar coordinate system
M = cv2.Rodrigues(headpose)[0]
Zv = M[:, 2]
headpose_theta = np.arcsin(Zv[1])  # Vertical head pose angle
headpose_phi = np.arctan2(Zv[0], Zv[2])  # Horizontal head pose angle

# Note: You'll need to implement the 'normalize_img' function in Python as shown in previous examples.
