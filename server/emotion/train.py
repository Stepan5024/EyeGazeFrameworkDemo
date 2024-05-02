
import argparse
import math
import torch
from torchvision import transforms
from server.models.emotionNet import EmotionNet
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from . import config as cfg
from .util import EarlyStopping
from .util import LRScheduler
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import pandas as pd

# Initialize the argument parser and establish the arguments required
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Path to save the trained model')
parser.add_argument('-p', '--plot', type=str, help='Path to save the loss/accuracy plot')
args = vars(parser.parse_args())

# Configure the device to use for training the model, either GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Current training device: {device}")

# Initialize a list of preprocessing steps to apply on each image during training

train_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])

# Define the test transformations and the data loading
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Load all the images within the specified folder and apply different augmentation
train_data = datasets.ImageFolder(cfg.train_directory, transform=train_transform)
test_data = datasets.ImageFolder(cfg.test_directory, transform=test_transform)

# Extract the class labels and the total number of classes
classes = train_data.classes
num_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")

# Use train samples to generate train/validation set
num_train_samples = len(train_data)
train_size = math.floor(num_train_samples * cfg.TRAIN_SIZE)
val_size = math.ceil(num_train_samples * cfg.VAL_SIZE)
print(f"[INFO] Train samples: {train_size} \t Validation samples: {val_size}")

# Randomly split the training dataset into train and validation set
train_data, val_data = random_split(train_data, [train_size, val_size])

# Modify the data transform applied towards the validation set
val_data.dataset.transforms = test_transform

# Get the labels within the training set
train_classes = [label for _, label in train_data]

# Count each label within each class
class_count = Counter(train_classes)
print(f"[INFO] Total sample: {class_count}")

# Compute and determine the weights to be applied on each category depending on the number of samples available
class_weight = torch.Tensor([len(train_classes) / c for c in pd.Series(class_count).sort_index().values])

sample_weight = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data):
    weight = class_weight[label]
    sample_weight[idx] = weight

# Define a sampler which randomly samples labels from the train dataset
sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_data), replacement=True)

# Load our own dataset and store each sample with their corresponding labels
train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, sampler=sampler)
val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

# Initialize the model and send it to device
model = EmotionNet(num_of_channels=1, num_of_classes=num_of_classes)
model = model.to(device)

optimizer = SGD(params=model.parameters(), lr=cfg.LR)
criterion = nn.CrossEntropyLoss()

# Initialize the learning rate scheduler and early stopping mechanism
lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

# Calculate the steps per epoch for training and validation set
train_steps = len(train_dataloader.dataset) // cfg.BATCH_SIZE
val_steps = len(val_dataloader.dataset) // cfg.BATCH_SIZE

# Initialize a dictionary to save the training history
history = {
    "train_acc": [],
    "train_loss": [],
    "val_acc": [],
    "val_loss": []
}

# Start training the model
print("[INFO] Training the model...")
start_time = datetime.now()

for epoch in range(0, cfg.NUM_OF_EPOCHS):
    print(f"[INFO] epoch: {epoch + 1}/{cfg.NUM_OF_EPOCHS}")

    """
    Training the model
    """
    # Set the model to training mode
    model.train()

    # Initialize the total training and validation loss and 
    # the total number of correct predictions in both steps
    total_train_loss = 0
    total_val_loss = 0

    train_correct = 0
    val_correct = 0

    # Iterate through the training set
    for (data, target) in train_dataloader:
        # Move the data into the device used for training
        data, target = data.to(device), target.to(device)

        # Perform a forward pass and calculate the training loss
        predictions = model(data)
        loss = criterion(predictions, target)

        # Zero the gradients accumulated from the previous operation,
        # perform a backward pass, and then update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add the training loss and keep track of the number of correct predictions
        total_train_loss += loss.item()
        train_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()
    
    model.eval()  # Disable dropout and dropout layers

    # Prevents PyTorch from calculating the gradients, reducing memory usage and speeding up computation (no backprop)
    with torch.set_grad_enabled(False):
        # Iterate through the validation set
        for (data, target) in val_dataloader:
            # Move the data into the device used for testing
            data, target = data.to(device), target.to(device)

            # Perform a forward pass and calculate the training loss
            predictions = model(data)
            loss = criterion(predictions, target)

            # Add the validation loss and keep track of the number of correct predictions
            total_val_loss += loss.item()
            val_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    # Calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    # Calculate the training and validation accuracy
    train_accuracy = train_correct / len(train_dataloader.dataset)
    val_accuracy = val_correct / len(val_dataloader.dataset)

    # Print model training and validation records
    print(f"train loss: {avg_train_loss:.3f} - train accuracy: {train_accuracy:.3f}")
    print(f"val loss: {avg_val_loss:.3f} - val accuracy: {val_accuracy:.3f}", end='\n\n')

    # Update the training and validation results in history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_accuracy)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_accuracy)

    # Execute the learning rate scheduler and early stopping
    lr_scheduler(avg_val_loss)
    early_stopping(avg_val_loss)
    
    # Stop the training procedure if early stopping is triggered
    if early_stopping.early_stop_enabled:
        break

    print(f"[INFO] Total training time: {datetime.now() - start_time}...")

    # Move model back to CPU and save the trained model to disk
    if device == "cuda":
        model = model.to("cpu")
    torch.save(model.state_dict(), './resources/models/emotion/emotion_model.pth')

    # Plot the training loss and accuracy over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('# of Epochs')
    plt.title('Training Loss and Accuracy on FER2013')
    plt.legend(loc='upper right')
    plt.savefig('./resources/models/emotion/emotion_model.jpg')

    # Evaluate the model based on the test set
    model = model.to(device)
    with torch.set_grad_enabled(False):
        model.eval()  # Set the evaluation mode

        # Initialize a list to keep track of our predictions
        predictions = []

        # Iterate through the test set
        for (data, _) in test_dataloader:
            # Move the data into the device used for testing
            data = data.to(device)

            # Perform a forward pass and calculate the output
            output = model(data)
            output = output.argmax(axis=1).cpu().numpy()
            predictions.extend(output)

    # Evaluate the network
    print("[INFO] Evaluating network...")
    actual = [label for _, label in test_data]
    print(classification_report(actual, predictions, target_names=test_data.classes))
