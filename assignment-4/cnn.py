import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import datetime
from torch.utils.tensorboard import SummaryWriter
import re
from train_model import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
IMAGE_PATH = "../datasets/UCMerced_LandUse/Images"
OUTPUT_PATH = "output/adadelta"

img_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

os.makedirs("./cache", exist_ok=True)
os.makedirs("./output", exist_ok=True)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

dataset = datasets.ImageFolder(root=IMAGE_PATH, 
                               transform=img_transform,
                               target_transform=None)

len_dataset = len(dataset)
train_split, test_split = int(len_dataset*0.7), int(len_dataset*0.2)
val_split = len_dataset - train_split - test_split
train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_split, test_split, val_split])

class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

train_loader, test_loader, val_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True), DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model = CNN(num_classes=len(class_to_idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

model = train(model, train_loader, val_loader, optimizer, loss_fn, BATCH_SIZE, 10, OUTPUT_PATH, resume=False, print_freq=10, timestamp=timestamp)

models = os.listdir(Path(OUTPUT_PATH) / 'models')
pattern = re.compile(rf"^model_{timestamp}_\d+\.pth$")
models = [m for m in models if pattern.match(m)]
models.sort()
model_path = models[-1]
model = CNN(num_classes=len(class_to_idx)).to(device)
model.load_state_dict(torch.load(Path(OUTPUT_PATH) / 'models' / model_path))

test(model, test_loader, loss_fn, BATCH_SIZE, OUTPUT_PATH, timestamp)