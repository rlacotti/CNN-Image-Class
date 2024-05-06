# Imports
import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Loaddata import create_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create cnn class
class CNN(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv2d(input_channel, out_channels=32, kernel_size=5, stride=2, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(p=0.1)
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define folder path for checkpoints
checkpoint_folder = "checkpoints"

# Create folder if it does not exist
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

def save_checkpoint(state, filename="Checkpoint20.pth.tar"):
    logging.info("--> Saving checkpoint")
    torch.save(state, os.path.join(checkpoint_folder, filename))

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    logging.info("--> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return epoch, loss

def save_metrics(train_accuracies, test_accuracies, train_losses, test_losses, 
                 train_precision, test_precision, train_recall, test_recall, epoch):
    # Create file to save metrics
    filename = os.path.join(checkpoint_folder, f"metrics_epoch_1.txt")
    with open(filename, "a") as file:
        file.write(f"Iteration {epoch}:\n")
        file.write("Train accuracies:\n")
        file.write("\n".join(map(str, train_accuracies)) + "\n")
        file.write("Test accuracies:\n")
        file.write("\n".join(map(str, test_accuracies)) + "\n")
        file.write("Train losses:\n")
        file.write("\n".join(map(str, train_losses)) + "\n")
        file.write("Test losses:\n")
        file.write("\n".join(map(str, test_losses)) + "\n")
        file.write("Train precision:\n")
        file.write("\n".join(map(str, train_precision)) + "\n")
        file.write("Test precision:\n")
        file.write("\n".join(map(str, test_precision)) + "\n")
        file.write("Train recall:\n")
        file.write("\n".join(map(str, train_recall)) + "\n")
        file.write("Test recall:\n")
        file.write("\n".join(map(str, test_recall)) + "\n\n")

# Accuracy eval
def acc_check(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0
    losses = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            _, preds = scores.max(1)
            num_correct += (preds == labels).sum().item()
            num_samples += preds.size(0)
            losses.append(criterion(scores, labels).item())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    model.train()
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    return num_correct / num_samples, losses, predictions, true_labels

# Calculate precision and recall
def calculate_precision_recall(predictions, labels):
    num_classes = len(set(labels))
    precision_list = []
    recall_list = []
    for class_id in range(num_classes):
        TP = ((predictions == class_id) & (labels == class_id)).sum()
        FP = ((predictions == class_id) & (labels != class_id)).sum()
        FN = ((predictions != class_id) & (labels == class_id)).sum()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
    return precision_list, recall_list

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_channel = 3
num_classes = 10
batch_size = 32
lr = 0.0001
num_epochs = 5
load_model = True
data_root = r'C:\Users\ryanl\OneDrive\Documents\CNN Image Class\DATA'

# Load data
train_loader, test_loader = create_dataloader(data_root, batch_size, random_seed=42)

# Initialize the model
model = CNN(input_channel, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# Check if the checkpoint file exists
checkpoint_path = os.path.join(checkpoint_folder, "Checkpoint19.pth.tar")
if os.path.exists(checkpoint_path) and load_model:
    load_checkpoint(torch.load(checkpoint_path), model, optimizer, scheduler)

# Initialize lists to store metrics
train_accuracies_all = []
test_accuracies_all = []
train_losses_all = []
test_losses_all = []
train_precision_all = []
test_precision_all = []
train_recall_all = []
test_recall_all = []



# Training loop
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        scores = model(images)
        loss = criterion(scores, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Calculate accuracy and losses on training and test sets
    train_accuracy, train_losses, train_predictions, train_labels = acc_check(train_loader, model)
    test_accuracy, test_losses, test_predictions, test_labels = acc_check(test_loader, model)

    # Append accuracies and losses to lists
    train_accuracies_all.append(train_accuracy * 100)
    test_accuracies_all.append(test_accuracy * 100)
    train_losses_all.append(sum(train_losses) / len(train_losses))
    test_losses_all.append(sum(test_losses) / len(test_losses))

    # Calculate precision and recall
    train_precision, train_recall = calculate_precision_recall(train_predictions, train_labels)
    test_precision, test_recall = calculate_precision_recall(test_predictions, test_labels)

    # Append precision and recall to lists
    train_precision_all.append(train_precision)
    test_precision_all.append(test_precision)
    train_recall_all.append(train_recall)
    test_recall_all.append(test_recall)

    # Save metrics after every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_metrics(train_accuracies_all, test_accuracies_all, train_losses_all, test_losses_all, 
                     train_precision_all, test_precision_all, train_recall_all, test_recall_all, epoch)

    # Save checkpoint after every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        save_checkpoint(checkpoint)

    # Step the scheduler
    scheduler.step(test_accuracy)

if load_model:
    load_checkpoint(torch.load(os.path.join(checkpoint_folder, "Checkpoint20.pth.tar")), model, optimizer, scheduler)

# Print accuracy on training and test sets
print(f"Accuracy on training set: {train_accuracies_all[-1]:.2f}%")
print(f"Accuracy on test set: {test_accuracies_all[-1]:.2f}%")














































