import argparse
import torch
import cv2

# Initialize the argument parser and establish the arguments required
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video", type=str, required=True, help="path to the video file/webcam")
parser.add_argument("-m", "--model", type=str, required=True, help="path to the trained model")
parser.add_argument("-p", "--prototxt", type=str, required=True, help="Path to deployed prototxt.txt model architecture file")
parser.add_argument("-c", "--caffemodel", type=str, required=True, help="Path to Caffe model containing the weights")
parser.add_argument("-conf", "--confidence", type=int, default=0.5, help="the minimum probability to filter out weak detection")
args = vars(parser.parse_args())

# Load our serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['caffemodel'])

# Check if GPU is available or not
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dictionary mapping for different outputs
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

# Load the EmotionNet weights
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model_weights = torch.load(args["model"])
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Initialize a list of preprocessing steps to apply on each image during runtime
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Initialize the video stream