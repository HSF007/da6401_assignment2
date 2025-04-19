import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
import random
from Q1 import CNNModel  # Import the PyTorch model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Data paths
DATASET_PATH = r"E:\IITM\2nd sem\inaturalist_12K"  # Update with your path
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "val")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Hyperparameter sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'base_filters': {'values': [32, 64]},
        'conv_activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
        'dense_activation': {'values':['relu', 'gelu', 'silu', 'leakyrelu']},
        'filter_organization': {'values': ['same', 'doubling', 'halving']},
        'data_augmentation': {'values': [True, False]},
        'use_batch_norm': {'values': [True, False]},
        'dropout_rate': {'values': [0, 0.2, 0.3, 0.5]},
        'dense_neurons': {'values': [128, 256, 512, 1024]},
        'learning_rate': {'values': [0.0001, 0.001, 0.01]},
        'epochs': {'value': 10},
        'batch_size': {'value': 32},
        'image_size': {'value': 224},
        'validation_split': {'value': 0.2}
    }
}

class iNaturalistDataset(Dataset):
    """Custom Dataset for iNaturalist images."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with class subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get class directories and create class-to-idx mapping
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_and_split_data(config):
    """
    Load data and split into train and validation sets,
    ensuring equal class representation in validation set.
    """
    # Base transforms
    base_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data augmentation transform
    augment_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Choose transform based on config
    train_transform = augment_transform if config.data_augmentation else base_transform
    
    # Load dataset
    full_dataset = iNaturalistDataset(root_dir=TRAIN_DIR, transform=train_transform)
    val_dataset = iNaturalistDataset(root_dir=TRAIN_DIR, transform=base_transform)
    test_datase = iNaturalistDataset(root_dir=TEST_DIR, transform=base_transform)
    
    # Get class counts for stratified split
    class_counts = {}
    for label in full_dataset.labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Create stratified split
    train_indices = []
    val_indices = []
    
    for class_idx in range(len(full_dataset.classes)):
        # Get indices for this class
        class_indices = [i for i, label in enumerate(full_dataset.labels) if label == class_idx]
        np.random.shuffle(class_indices)
        
        # Split indices
        val_count = int(len(class_indices) * config.validation_split)
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_datase,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(full_dataset.classes)


def train_model(config):
    """Train the model with current hyperparameter configuration."""
    # Load and split data
    train_loader, val_loader, test_loader, num_classes = load_and_split_data(config)
    
    # Create model based on hyperparameters
    if config.filter_organization == 'same':
        filters = [config.base_filters] * 5
    elif config.filter_organization == 'doubling':
        filters = [config.base_filters * (2**i) for i in range(5)]
    elif config.filter_organization == 'halving':
        filters = [config.base_filters * (2**(4-i)) for i in range(5)]
    else:
        filters = [32, 64, 128, 256, 512]  # Default
    
    model = CNNModel(
        input_channels=3,
        num_classes=num_classes,
        filters_per_layer=filters,
        kernel_size=3,
        conv_activation=config.conv_activation,
        dense_units=config.dense_neurons,
        dense_activation = config.dense_activation,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm
    )
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize WandB for tracking
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_acc
        })
        
        print(f'Epoch: {epoch + 1}, Val Loss: {val_loss / len(val_loader):.3f}, Val Acc: {100 * val_acc:.2f}%')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model
            torch.save(model, f"best_model_{wandb.run.id}.pth")
            # Log the model to wandb
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(f"best_model_{wandb.run.id}.pth")
            wandb.log_artifact(artifact)
    
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f'Test Accuracy: {100 * test_acc:.2f}%')
    wandb.log({"test_accuracy": test_acc})
    
    # Save final model
    torch.save(model, f"final_model_{wandb.run.id}.pth")

    # Log the model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f"final_model_{wandb.run.id}.pth")
    wandb.log_artifact(artifact)


def sweep_train():
    """Configure and run hyperparameter sweep."""
    # Initialize wandb
    wandb.init()

    # Configuration parameters
    config = wandb.config

    # Set run name based on hyperparameters
    run_name = f'bf_{config.base_filters}_fo_{config.filter_organization}_dn_{config.dense_neurons}_ca_{config.conv_activation}_da_{config.dense_activation}_v5'
    wandb.run.name = run_name

    # Call training function with current hyperparameters
    train_model(config)


if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="Deep-learning-A2-v3")
    
    # Start the sweep
    wandb.agent(sweep_id, sweep_train, count=1)