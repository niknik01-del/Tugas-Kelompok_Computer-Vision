# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 08:08:13 2025

@author: BMKGPC
"""
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
import random
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Tambahkan import untuk metrik baru
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import json # Untuk menyimpan metrik ke JSON
import seaborn as sns # Untuk visualisasi confusion matrix (heatmap)

# Import necessary libraries............................
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#from torchvision.transforms import v2
from torchvision import transforms, models
from torch.optim import lr_scheduler
import torch.nn.functional as F # activation function
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from tempfile import TemporaryDirectory
from torchsummary import summary
import time
# %matplotlib inline
if __name__ == "__main__":
    # Pengecekan GPU hanya berjalan jika skrip dijalankan sebagai main
    if torch.cuda.is_available():
        print("Number of GPU: ", torch.cuda.device_count())
        print("GPU Name: ", torch.cuda.get_device_name(0))
    else:
        print("GPU not available. Using CPU.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

"""# Cloud Image Dataset imported"""

# Perhatian: Jalur ini harus disesuaikan dengan lingkungan Anda
dataset_path_train = r"D:\NIKEN\Personal\Tugas_Kelompok_ACV_Awan\clouds_train"

# Analyze dataset structure to understand what files are available
def analyze_dataset(root_dir):
    structure = {}
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found at {root_dir}")
        return {}

    for root, dirs, files in os.walk(root_dir):
        rel_dir = os.path.relpath(root, root_dir)
        if rel_dir == '.':
            continue

        # Count files by extension
        file_counts = {}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in file_counts:
                file_counts[ext] += 1
            else:
                file_counts[ext] = 1

        structure[rel_dir] = file_counts

    return structure

dataset_structure = analyze_dataset(dataset_path_train)
print("Dataset structure (Train):")
for dir_path, file_types in dataset_structure.items():
    print(f"{dir_path}: {file_types}")

dataset_path_test = r"D:\NIKEN\Personal\Tugas_Kelompok_ACV_Awan\clouds_test"

dataset_structure = analyze_dataset(dataset_path_test)
print("Dataset structure (Test/Validation):")
for dir_path, file_types in dataset_structure.items():
    print(f"{dir_path}: {file_types}")

"""# Data Loading and Preprocessing"""

Root_dir = dataset_path_train

test_dir = dataset_path_test

clouds_classes = os.listdir(Root_dir)

print('Total Clouds Class ', len(os.listdir(Root_dir)))

class_names = ['high cumuliform clouds','cumulus clouds','cirriform clouds','stratiform clouds','stratocumulus clouds','cumulonimbus clouds','clear sky']

train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=18),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the data
])

val_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the data
])

train = datasets.ImageFolder(
    root = Root_dir,
    transform = train_transform
)

valid = datasets.ImageFolder(
    root=test_dir,
    transform=val_transform
)


print(f'Train dataset : {len(train)}, {type(train)}')

print(f'Validation dataset : {len(valid)}')

"""# Create a custom dataset"""

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)

"""# Train and valid Dataloader"""

batch_size = 32
# DataLoaders for training and validation
train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid, batch_size, num_workers=0, pin_memory=True)

# Moving data into GPU, WrappedDataLoader
train_dataloader = DeviceDataLoader(train_loader, device)
valid_dataloader = DeviceDataLoader(valid_loader, device)

"""# Sample Image show after Augmentation"""

# Function to show an image
def imshow(img):
    """Display unnormalized image from a torch Tensor."""
    img = img.clone().detach()  # detach from graph
    
    # Denormalize the image (undoing the transforms.Normalize)
    # The normalization applied was: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # x' = (x - mean) / std  => x = x' * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1) # Clip to [0, 1] range

    npimg = img.numpy()

    # Check if single channel (grayscale) or 3-channel (RGB)
    if npimg.shape[0] == 1:
        npimg = npimg.squeeze()  # remove channel dimension
        plt.imshow(npimg, cmap='gray')
    else:
        npimg = np.transpose(npimg, (1, 2, 0))  # CxHxW → HxWxC
        plt.imshow(npimg)

    plt.axis('off')
    plt.show()


# # # Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)

# # Show one augmented image
print('Augmented Image:')
imshow(images[0])
print('Label:', train.classes[labels[0]])

# Get some random test images (without augmentation)
valid_dataiter = iter(valid_loader)
valid_images, valid_labels = next(valid_dataiter)

# Show one original test image
print('Original Valid Image:')
imshow(valid_images[0])
print('Label:', valid.classes[valid_labels[0]])

# 3. Visualizing Augmented Images

# Function to display a batch of images..........................
def imshow_batch(img_batch, labels_batch, title, data):
    # Denormalize the batch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    img_batch = img_batch * std + mean
    img_batch = torch.clamp(img_batch, 0, 1) # Clip to [0, 1] range

    npimg = torchvision.utils.make_grid(img_batch, nrow=8)
    npimg = npimg.numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze())
    plt.title(title)
    plt.axis('off')
    plt.show()
    # Print labels
    print('Labels:', ' '.join(f'{data.classes[labels_batch[j]]}' for j in range(len(labels_batch))))


# Display a batch of augmented images
print('Batch of Augmented Images:')
imshow_batch(images[:24], labels[:24], 'Augmented Training Images', train)

# # Display a batch of original test images
# print('Batch of Original Valid Images:')
# imshow_batch(valid_images[:24], valid_labels[:24], 'Original Valid Images', valid)

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        #images, labels = images.to(DEVICE), labels.to(DEVICE) # move to GPU
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        #images, labels = images.to(DEVICE), labels.to(DEVICE) # move to GPU
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

"""# Custom resnet architecture"""

# resnet architecture
class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        #self.conv5 = ConvBlock(256, 256, pool=True)
        #self.conv6 = ConvBlock(256, 512, pool=True)
        #self.conv7 = ConvBlock(512, 512, pool=True)

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        # self.classifier = nn.Sequential(nn.MaxPool2d(4),
        #                                nn.Flatten(),
        #                                nn.Linear(512, num_diseases))

        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Safe replacement
                nn.Flatten(),
                nn.Linear(512, num_diseases)
        )

    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        #out = self.conv5(out)
        #out = self.conv6(out)
        #out = self.conv7(out)
        out = self.res2(out) + out
        out = self.classifier(out)

        return out

# defining the model and moving it to the GPU
# 3 is number of channels RGB, len(train.classes()) is number of diseases.
model = to_device(CNN_NeuralNet(3, len(class_names)), device)
#model = model.to(DEVICE)
model

# for training
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    if optimizer is None:
        raise ValueError("Optimizer is not defined.")
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no param groups.")
    else:
        for param_group in optimizer.param_groups:
            return param_group['lr']

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = [evaluate(model, valid_dataloader)]
history

"""# Hyperparameters Function:

Now it's time to create a function that get epochs, learning rate, train and validation loader and optim function..

Clear GPU memory after PyTorch model training without restarting kernel with torch.cuda.empty_cache()
"""

def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []  # Untuk mengumpulkan hasil

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                                epochs=epochs, steps_per_epoch=len(train_loader))


    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            sched.step()
             # validation

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        # >>> FIX: Menambahkan nomor epoch ke history <<<
        result['epoch'] = epoch + 1
        # >>> END FIX <<<
        model.epoch_end(epoch, result)
        history.append(result)

    torch.save(model.state_dict(), 'resnet_Model.pth')
    return history

"""# Training Model:

Evaluate function added to history of model.

Then we can define our hyperparameters like number of epochs, learning rate and ... .

Now we can update history with fit_OneCycle function (Adding two function together). Attention to history = [] in the second function. Now we have model evaluation.
"""

num_epoch = 40
lr_rate = 0.001
grad_clip = 0.15
weight_decay = 1e-4
optims = torch.optim.AdamW

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = fit_OneCycle(num_epoch, lr_rate, model, train_dataloader, valid_dataloader,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=optims)

val_acc = []
val_loss = []
train_loss = []

for i in history:
    val_acc.append(i['val_acc'])
    val_loss.append(i['val_loss'])
    train_loss.append(i.get('train_loss'))

"""# Loss per Epochs curve"""

epoch_count = range(1, len(train_loss) + 1)

plt.figure(figsize=(10,5), dpi=200)
plt.plot(epoch_count, train_loss, 'r--', color= 'orangered')
plt.plot(epoch_count, val_loss, '--bo',color= 'green', linewidth = '2.5', label='line with marker')
plt.legend(['Training Loss', 'Val Loss'])
plt.title('Number of epochs & Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(1, len(train_loss) + 1, 2))
plt.show();

epoch_count = range(1, len(val_acc) + 1)
plt.figure(figsize=(10,5), dpi=200)
plt.plot(epoch_count, val_acc, '--bo',color= 'green', linewidth = '2.5', label='line with marker')
plt.legend(['Val Acc'])
plt.title('Number of epochs & Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.xticks(np.arange(1, len(val_acc) + 1, 2))
plt.show();

"""# =====================================================================
FINAL EVALUATION AND METRICS SAVING (Perbaikan dan Penambahan Metrik)
====================================================================="""

# Load the best model weights
model.load_state_dict(torch.load("resnet_Model.pth"))
model.eval()
print("\n--- Final Evaluation and Metrics Saving ---")

all_labels = []
all_preds = []

# Collect predictions and true labels from the validation set
with torch.no_grad():
    for inputs, targets in tqdm(valid_dataloader, desc="Collecting Predictions"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)

        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# --- 1. Classification Metrics Calculation ---
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
conf_mat = confusion_matrix(all_labels, all_preds)

# Persiapan data metrik untuk saving
metrics_data = {
    "accuracy": report['accuracy'],
    "macro_avg_f1_score": report['macro avg']['f1-score'],
    "per_class_metrics": {
        name: {
            "precision": report[name]['precision'],
            "recall": report[name]['recall'],
            "f1-score": report[name]['f1-score'],
            "support": report[name]['support']
        }
        for name in class_names
    },
    "confusion_matrix": conf_mat.tolist(), # Convert to list for JSON serialization
    "class_names": class_names
}

# --- SETUP SAVE DIRECTORY ---
save_dir = "models_web_h5"
os.makedirs(save_dir, exist_ok=True)


# --- 2. Simpan Data History (Loss/Acc per epoch) ---
history_path = os.path.join(save_dir, "training_history.json")
with open(history_path, "w", encoding="utf-8") as f:
    # Simpan variabel 'history' yang sudah berisi train_loss, val_loss, val_acc, dan epoch
    json.dump(history, f, ensure_ascii=False, indent=4)
print(f"[INFO] Training History (Loss/Acc) saved to: {history_path}")


# --- 3. Simpan Metrics File (cloud_metrics.json) ---
metrics_path = os.path.join(save_dir, "cloud_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, ensure_ascii=False, indent=4)
print(f"[INFO] Classification metrics and CM saved to: {metrics_path}")


# --- 4. Simpan Model Weights dan Labels ---

deploy_model = model.to("cpu")
deploy_model.eval()

weights_path = os.path.join(save_dir, "cloud_resnet_state_dict.h5")
torch.save(deploy_model.state_dict(), weights_path)

labels_path = os.path.join(save_dir, "cloud_labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("\n[INFO] All deployment assets saved in folder:", save_dir)
print(f"  Weights: {weights_path}")
print(f"  Labels: {labels_path}")