import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

IMAGE_DIR = "../datasets/UCMerced_LandUse/Images"
BATCH_SIZE = 32

# Load the dataset
input_img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
dataset = datasets.ImageFolder(
    root=IMAGE_DIR,
    transform=input_img_transform,
    target_transform=None,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)


# Define the main CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer10 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(7 * 7 * 512, 4096), nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

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
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


model = CNN(num_classes=len(dataset.classes))
if os.path.exists("output_cnn/models/model_20250309_153723_9.pth"):
    print("Loading model from file")
    # Load the trained model
    model.load_state_dict(torch.load("output_cnn/models/model_20250309_153723_9.pth"))
else:
    print("No pre-trained model found!")
    exit(1)


# Define the extension of the CNN model to be used for CAM
class CNNCAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 512, 7 * 7)
        x = x.mean(2)
        x = self.fc(x)
        return F.softmax(x, dim=1)


# Swap the last FC layers of the model with the CAM layers
# Remove the last 3 FC layers
model_with_only_cnn_part = nn.Sequential(*list(model.children())[:-3])
model = nn.Sequential(
    model_with_only_cnn_part,
    CNNCAM(num_classes=len(dataset.classes)),
)

# Select only the new FC layers for training
trainable_params = []
for name, param in model.named_parameters():
    if "fc" in name:
        trainable_params.append(param)

optimizer = torch.optim.Adam(
    trainable_params,
    lr=0.0001,
    weight_decay=0.0001,
)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, len(dataloader), loss.item()
                )
            )

# Save the model
torch.save(model.state_dict(), "output_cnn/models/model_cnn_cam.pth")

# Set the model to evaluation mode
model.eval()


def get_cam_img(feature_blobs, fc_weights, class_idx):
    bz, nc, h, w = feature_blobs.shape
    beforeDot = feature_blobs.reshape((nc, h * w))
    cam = np.matmul(fc_weights[class_idx], beforeDot)
    cam = cam.reshape(h, w)
    min = torch.min(cam)
    cam = cam - 1
    cam_img = cam / torch.max(cam)
    cam_img = (255 * cam_img).to(torch.uint8)
    cam_img = cam_img.cpu().numpy()  # Convert to NumPy
    cam_img = cv2.resize(cam_img, (img.shape[1], img.shape[0]))  # Resize
    return cam_img


# Make the predictions and CAM for each image in the dataset
for img_path, _ in dataset.imgs:
    img = cv2.imread(img_path)
    input_tensor: torch.Tensor = input_img_transform(img)  # type: ignore
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
    predicted_idx = F.softmax(prediction, dim=1).argmax(dim=1).cpu().numpy()[0]

    print(f"Image: {img_path}, Prediction: {dataset.classes[predicted_idx]}")

    feature_blobs = model_with_only_cnn_part(input_tensor)
    feature_blobs = feature_blobs.cpu().detach().numpy()

    # Get the weights of the last FC layer
    fc_weights = list(model.modules())[-1].weight.data

    # Compute the CAM
    cam_img = get_cam_img(feature_blobs, fc_weights, predicted_idx)

    # Resize the CAM to the original image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Blend the CAM with the original image
    blended_img = cv2.addWeighted(img, 0.5, cam, 0.5, 0)

    # Save the blended image
    cv2.imwrite(f"output_cnn/cam_images/{img_path.split('/')[-1]}", blended_img)
