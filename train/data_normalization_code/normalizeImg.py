import numpy as np
import cv2

def normalize_img(input_img, target_3d, hR, gc, roi_size, camera_matrix, focal_new=960, distance_new=600):
    # Calculate distance and scaling factor
    distance = np.linalg.norm(target_3d)
    z_scale = distance_new / distance
    
    # Create new camera matrix for the virtual camera
    cam_new = np.array([
        [focal_new, 0, roi_size[0] / 2],
        [0, focal_new, roi_size[1] / 2],
        [0, 0, 1.0]
    ])
    
    # Scale matrix
    scale_mat = np.diag([1.0, 1.0, z_scale])
    
    # Calculate the new rotation matrix
    hRx = hR[:, 0]
    forward = target_3d / distance
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    rot_mat = np.column_stack((right, down, forward)).T
    
    # Warp matrix calculation
    warp_mat = cam_new.dot(scale_mat).dot(rot_mat.dot(np.linalg.inv(camera_matrix)))
    
    # Perform the image warping
    img_warped = cv2.warpPerspective(input_img, warp_mat, dsize=(roi_size[0], roi_size[1]))
    
    # Normalize the rotation
    cnv_mat = scale_mat.dot(rot_mat)
    hR_new = cnv_mat.dot(hR)
    hr_new = cv2.Rodrigues(hR_new)[0]  # Convert rotation matrix to rotation vector
    
    # Normalize the gaze vector
    ht_new = cnv_mat.dot(target_3d[:, None])  # Ensure target_3d is a column vector
    gc_new = cnv_mat.dot(gc[:, None])  # Ensure gc is a column vector
    gv_new = (gc_new - ht_new).flatten()  # Subtract and flatten to get the gaze vector
    gv_new /= np.linalg.norm(gv_new)  # Normalize the gaze vector
    
    return img_warped, hr_new, gv_new

# Example of using the function
# Assuming all required inputs are defined: input_img, target_3d, hR, gc, roi_size, camera_matrix
# img_warped, hr_new, gv_new = normalize_img(input_img, target_3d, hR, gc, roi_size, camera_matrix)
