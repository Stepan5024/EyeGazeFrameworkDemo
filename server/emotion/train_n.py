import torch
from torch.utils import data
import pandas as pd
import albumentations
from albumentations import pytorch as AT
from tqdm import tqdm
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch.nn.functional as F
import torchvision.models as models
from torch import nn
#from torchsummary import summary
from collections import OrderedDict
import torch.optim as optim

from .data_loader_n import Dataset
from server.models.EmotionNet_n  import EmotionNet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"use_cuda {use_cuda} {device}")

model=EmotionNet(1.0)
model.to(device)
epochs=100

input_size=(3,48,48)
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []

class AlbumentationWrapper(object):
    def __init__(self,split):
        self.split=split
        self.aug=albumentations.Compose([                                         
    albumentations.Normalize((0.5), (0.5)),
    AT.ToTensorV2()
    ])
	
        if self.split=='train':
            self.aug=albumentations.Compose([
                                             
            #albumentations.Resize(48,48),
    albumentations.HorizontalFlip(),
    albumentations.CoarseDropout(max_holes=2, max_height=2, max_width=2, min_holes=1, min_height=1, min_width=1, fill_value=0.5, p=0.5),
    albumentations.GaussNoise(),
    #albumentations.ElasticTransform(),    
    albumentations.Normalize((0.5), (0.5)),
    AT.ToTensorV2()    
    ])
            
    def __call__(self,img):
        #img = np.array(img)
        img = self.aug(image=img)['image']
        return img
    
def main():


    params = {'batch_size': 64,'shuffle': True,'num_workers': 10}

    df=pd.read_csv('F:\\EyeGazeDataset\\emotion\\fer2013.csv')
    df['pixelss']=[[int(y) for y in x.split()] for x in df['pixels']]

    df_train=df[df['Usage']=='Training']
    df_valid=df[df['Usage']=='PrivateTest']
    df_test=df[df['Usage']=='PublicTest']

    part={}
    part['train']= list(range(0,len(df_train)))
    part['valid']= list(range(0,len(df_valid)))
    part['test']= list(range(0,len(df_test)))
    train_labels=df_train['emotion'].tolist()
    valid_labels=df_valid['emotion'].tolist()
    test_labels=df_test['emotion'].tolist()

    train_transforms , validation_transforms=AlbumentationWrapper('train'), AlbumentationWrapper('test')
    
    training_set = Dataset(df_train, train_transforms)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(df_valid, validation_transforms)
    validation_generator = data.DataLoader(validation_set, **params)

    test_set = Dataset(df_test, validation_transforms)
    test_generator = data.DataLoader(test_set, **params)
    
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=9e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=0.02, 
                                                    steps_per_epoch=len(training_generator), 
                                                    pct_start=0.2, div_factor=10, 
                                                    cycle_momentum=False, 
                                                    epochs=epochs)

    for epoch in range(epochs):
        print("EPOCH: %s LR: %s " % (epoch, get_lr(optimizer)))
        train(model, training_generator, optimizer,scheduler)
        test(model, validation_generator)
        #scheduler.step()
    plot(train_loss,train_acc, valid_loss, valid_acc, 'Loss & Accuracy')
    # Save the trained model to ONNX
    save_model_onnx(model, input_size)

def save_model_onnx(model, input_size, file_name="./resources/models/emotion/emotion_model.onnx"):
    model.eval()
    # Create a dummy input tensor with the correct size
    dummy_input = torch.randn(1, *input_size, device=device)
    # Export the model
    torch.onnx.export(model, dummy_input, file_name, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model saved to {file_name}")

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            new_target=target.view_as(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    valid_loss.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    valid_acc.append(100. * correct / len(test_loader.dataset))

def plot(train_losses,train_acc,test_losses,test_acc, label):
  fig, axs = plt.subplots(1,2,figsize=(20,8))
  axs[0].plot(test_losses, label=label)
  axs[0].set_title("Test Loss")
  axs[1].plot(test_acc, label=label)
  axs[1].set_title("Test Accuracy")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(model, train_loader, optimizer,scheduler):
  model.train()
  pbar = tqdm(train_loader)
  running_loss = 0.0
  correct = 0
  processed = 0
  criterion = nn.CrossEntropyLoss()

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    loss = criterion(y_pred, target)
    running_loss += loss.item()
    train_loss.append(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} running_loss={running_loss} threshold={best_loss*(0.996)}')
    train_acc.append(100*correct/processed)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} le={get_lr(optimizer)} Accuracy={100*correct/processed:0.2f}')


if __name__ == '__main__':
    main()
