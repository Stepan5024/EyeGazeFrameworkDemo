import numpy as np
import cv2

def rodrigues(input_val):
    if isinstance(input_val, np.ndarray) and input_val.size in {3, 9}:
        # Convert rotation matrix to rotation vector (or vice versa)
        output_val, jacobian = cv2.Rodrigues(input_val)
        return output_val, jacobian
    else:
        raise ValueError("Input should be a 1x3 or 3x1 rotation vector or a 3x3 rotation matrix.")
    
# Example usage:
# Converting a rotation matrix to a rotation vector
rot_matrix = np.array([[0.5, -0.866, 0],
                       [0.866, 0.5, 0],
                       [0, 0, 1]])
rot_vector, _ = rodrigues(rot_matrix)
print("Rotation Vector:\n", rot_vector)

# Converting a rotation vector to a rotation matrix
rot_vector = np.array([[1.2], [0.2], [0.3]])
rot_matrix, _ = rodrigues(rot_vector)
print("Rotation Matrix:\n", rot_matrix)
