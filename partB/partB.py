import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

wandb.login()
# Initialize wandb
wandb.init(project="DA6401-A2-V4", name="finet-tunning-using-resnet50")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
data_dir = r"/kaggle/input/inatural-12k/inaturalist_12K"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")  # We'll use the original val as test

# Data augmentation and normalization
# Using ImageNet mean and std since we're using a pre-trained model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the full training dataset
full_train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])

# Split the training data into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create a proper validation dataset with validation transforms
class TransformedSubset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        # Apply the proper validation transform
        if self.transform:
            # Get the original image before any transforms
            original_img_path = self.subset.dataset.samples[self.subset.indices[idx]][0]
            from PIL import Image
            img = Image.open(original_img_path).convert('RGB')
            x = self.transform(img)
        return x, y
    
    def __len__(self):
        return len(self.subset)

# Apply validation transforms to validation split
val_dataset = TransformedSubset(val_dataset, data_transforms['val'])

# Load the test dataset
test_dataset = ImageFolder(test_dir, transform=data_transforms['test'])

# Create data loaders
batch_size = 32
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
}

# Get dataset sizes
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

# Get class names
class_names = full_train_dataset.classes
num_classes = len(class_names)

print(f"Number of training samples: {dataset_sizes['train']}")
print(f"Number of validation samples: {dataset_sizes['val']}")
print(f"Number of test samples: {dataset_sizes['test']}")
print(f"Number of classes: {num_classes}")

# Log configuration to wandb
wandb.config.update({
    "num_classes": num_classes,
    "train_samples": dataset_sizes['train'],
    "val_samples": dataset_sizes['val'],
    "test_samples": dataset_sizes['test'],
    "model": "ResNet50",
    "fine_tuning_strategy": "freeze_except_last_layer",
    "batch_size": batch_size
})

# Function to visualize sample images
def visualize_samples(dataloader, num_images=5):
    batch = next(iter(dataloader))
    images, labels = batch
    
    plt.figure(figsize=(15, 6))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        # Convert tensor to numpy and transpose to (H,W,C)
        img = images[i].numpy().transpose((1, 2, 0))
        # De-normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"Class: {class_names[labels[i]]}")
    plt.tight_layout()
    plt.savefig("sample_images.png")
    wandb.log({"sample_images": wandb.Image("sample_images.png")})

# Visualize some training images
visualize_samples(dataloaders['train'])
# Load pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Strategy 1: Freeze all layers except the last layer
# First, freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
# ResNet50's fc layer has shape (2048, 1000) where 1000 is the number of ImageNet classes
# We need to modify it for our number of classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move model to the device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Only optimize the parameters of the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log metrics to wandb
            wandb.log({
                f"{phase}_loss": epoch_loss,
                f"{phase}_accuracy": epoch_acc,
                "epoch": epoch
            })
            
            # Save the best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')
                
    print(f'Best val Acc: {best_acc:.4f}')
    return model

# Train the model
num_epochs = 10
model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    # Store all predictions and true labels
    all_preds = []
    all_labels = []
    
    # Confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Update statistics
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for wandb
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate accuracy
    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Calculate per-class accuracy
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    for i, acc in enumerate(per_class_acc):
        print(f'Accuracy of {class_names[i]}: {acc:.4f}')
    
    # Log metrics to wandb
    wandb.log({
        "test_accuracy": test_acc.item(),
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,  # Using collected predictions
            y_true=all_labels,  # Using collected true labels
            class_names=class_names
        )
    })
    
    return test_acc
# Evaluate on the test set
test_acc = evaluate_model(model, dataloaders['test'])

# Finish wandb run
wandb.finish()
