import os

# Initialize the path to the root folder where the dataset resides and the path to the train and test directories
DATASET_FOLDER = 'F:\\EyeGazeDataset\\emotion\\'
train_directory = os.path.join(DATASET_FOLDER, "train")
test_directory = os.path.join(DATASET_FOLDER, "test")

# Initialize the amount of samples to use for training and validation
TRAIN_SIZE = 0.90
VAL_SIZE = 0.10

# Specify the batch size, total number of epochs, and the learning rate
BATCH_SIZE = 16
NUM_OF_EPOCHS = 60
LR = 1e-1
