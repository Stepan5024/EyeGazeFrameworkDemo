import scipy.io
import numpy as np
import h5py

# Load the MAT file

mat = scipy.io.loadmat("F:\\EyeGazeDataset\\MPIIGaze_original\\Data\\Normalized\\p00\\day01.mat")
#mat = scipy.io.loadmat('/media/stepan/Expansion/EyeGazeDataset/MPIIGaze_original/Data/Normalized/p00/day01.mat')
# Display the keys and variables in the file
print("Keys in the MAT file:", mat.keys())

# Access the 'data' key
data = mat['data']

# 'data' is likely a structured array with multiple fields. Let's see what's inside.
print("Fields in 'data':", data.dtype.names)

# Access and print details for 'right' and 'left'
for side in ['right', 'left']:
    eye_data = data[side][0, 0]
    print(f"\nData for {side} eye:")
    print("Fields in eye data:", eye_data.dtype.names)
    
    # For each type of data in 'right' and 'left' (e.g., gaze, image, pose)
    for field in ['gaze', 'pose']:
        if field in eye_data.dtype.names:
            field_data = eye_data[field][0, 0]
            print(f"Shape of {field} data for {side} eye:", field_data.shape)
            print(f"Sample {field} data (first few entries):", field_data[:3])

            # Calculate mean, max, and min for each column in gaze and pose data
            mean_values = np.mean(field_data, axis=0)
            max_values = np.max(field_data, axis=0)
            min_values = np.min(field_data, axis=0)

            print(f"Mean {field} values for {side} eye:", mean_values)
            print(f"Max {field} values for {side} eye:", max_values)
            print(f"Min {field} values for {side} eye:", min_values)

# Assuming 'filenames' is a separate key or nested within right/left eye data
filenames = mat['filenames']

print(filenames[:10])

# Load the MAT file
mat = scipy.io.loadmat('F:\\EyeGazeDataset\\MPIIGaze_original\\6 points-based face model.mat')
#mat = scipy.io.loadmat('/media/stepan/Expansion/EyeGazeDataset/MPIIGaze_original/6 points-based face model.mat')
# Display the keys and variables in the file
print("6 points-based face model Keys in the MAT file:", mat.keys())
# Access the 'data' key
model = mat['model']
# 'data' is likely a structured array with multiple fields. Let's see what's inside.
print("Fields in 'model':", model.dtype.names)
print(model)

"""file_path = '/media/stepan/Expansion/EyeGazeDataset/MPIIFaceGaze_normalized/p00.mat'

# Настройка форматирования вывода для numpy
np.set_printoptions(suppress=True, precision=4)

with h5py.File(file_path, 'r') as file:
    print("Keys in the file:")
    for key in file.keys():
        item = file[key]
        if isinstance(item, h5py.Dataset):
            # It's a Dataset, print its shape, dtype, and sample data
            print(f" - {key}: shape={item.shape}, dtype={item.dtype}")
            # To safely print a sample without loading too much data into memory
            print("Sample data:", item[:1])  # Change ":1" to slice what you want
        elif isinstance(item, h5py.Group):
            # It's a Group, print its keys
            print(f" - {key}: (Group)")
            for subkey in item.keys():
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    # Subitem is a Dataset, print its shape, dtype, and sample data
                    print(f"   - {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
                    # To safely print a sample without loading too much data into memory
                    print("   Sample data:", subitem[:1])  # Change ":1" to slice what you want
                else:
                    # Subitem is a nested Group
                    print(f"   - {subkey}: (Nested Group)")

                    """