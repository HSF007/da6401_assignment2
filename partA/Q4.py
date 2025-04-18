import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Q1 import CNNModel
from Q2 import iNaturalistDataset

# Path to the test data
DATASET_PATH = r"E:\IITM\2nd sem\inaturalist_12K"  # Update with your path
TEST_DIR = os.path.join(DATASET_PATH, "val")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image size
IMAGE_SIZE = 224

def get_best_run_from_sweep(sweep_id, entity, project):
    api = wandb.Api()
    
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    
    best_val_acc = -1
    best_run = None
    
    for run in sweep.runs:
        val_acc = run.summary.get("val_accuracy")
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run = run
    
    if best_run is None:
        raise ValueError("No runs with 'val_accuracy' found in the sweep")
    
    print(f"Best run: {best_run.name}, val_accuracy: {best_val_acc:.4f}")
    return best_run

def download_best_model(run, artifact_name="model", output_dir="downloaded_model"):
    """
    Download the best model file from wandb.
    
    Args:
        best_run: The wandb run object for the best run
    
    Returns:
        model_path: Local path to the downloaded model
        config: Configuration of the best model
    """
    api = wandb.Api()

    # List all artifacts with this name in the project
    artifact_versions = api.artifacts(name=f"{run.project}/{artifact_name}", type_name='model')
    
    for artifact in artifact_versions:
        # Match the artifact to the run that created it
        if artifact.logged_by and artifact.logged_by().id == run.id:
            print(f"Found artifact version: {artifact.version} from run: {run.name}")
            artifact_dir = artifact.download(root=output_dir)
            for file_name in os.listdir(artifact_dir):
                print(file_name)
                if file_name.startswith("final_model") and file_name.endswith(".pth"):
                    model_path = os.path.join(artifact_dir, file_name)
                    print(f"Downloaded model file: {model_path}")
    
    print(f"Downloaded model file: {model_path}")
    
    # Get the model configuration from the run
    config = {
        'base_filters': run.config.get('base_filters', 32),
        'dense_activation': run.config.get('dense_activation', 'relu'),
        'filter_organization': run.config.get('filter_organization', 'doubling'),
        'dense_neurons': run.config.get('dense_neurons', 512),
        'dropout_rate': run.config.get('dropout_rate', 0.3),
        'use_batch_norm': run.config.get('use_batch_norm', True),
        'conv_activation': run.config.get('conv_activation', 'mish')
    }
    
    return model_path, config

def load_test_data():
    """Load the test dataset"""
    # Define transforms for test data
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = iNaturalistDataset(root_dir=TEST_DIR, transform=test_transform)
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_dataset

def evaluate_model(sweep_id='uf2dfd5t', entity='da24m008-iit-madras', project='DA6401-A2-V4'):
    """Evaluate the best model on the test set"""
    # Initialize wandb
    wandb.init(project=project, job_type="evaluation")
    
    try:
        # Get the best run and download its model
        best_run = get_best_run_from_sweep(
            sweep_id=sweep_id,
            entity=entity,
            project=project
        )

        model_path, best_config = download_best_model(best_run)
        
        # Log the best run information
        wandb.log({"best_run_id": best_run.id, "best_run_name": best_run.name})
        
        # Load test data
        test_loader, test_dataset = load_test_data()
        
        # Get class names and count
        class_names = test_dataset.classes
        num_classes = len(class_names)
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        
        # Update config with the correct number of classes
        best_config['num_classes'] = num_classes
        
        # Create model with the best configuration
        if best_config['filter_organization'] == 'same':
            filters = [best_config['base_filters']] * 5
        elif best_config['filter_organization'] == 'doubling':
            filters = [best_config['base_filters'] * (2**i) for i in range(5)]
        elif best_config['filter_organization'] == 'halving':
            filters = [best_config['base_filters'] * (2**(4-i)) for i in range(5)]
        else:
            filters = [32, 64, 128, 256, 512]  # Default
        
        model = CNNModel(
            input_channels=3,
            num_classes=num_classes,
            filters_per_layer=filters,
            kernel_size=3,
            conv_activation=best_config['conv_activation'],
            dense_units=best_config['dense_neurons'],
            dropout_rate=best_config['dropout_rate'],
            use_batch_norm=best_config['use_batch_norm'],
            dense_activation=best_config['dense_activation']
        )
        
        # Load the best model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Evaluate model
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Log to wandb
        wandb.log({
            "best_model_test_accuracy": test_accuracy
        })
        
        return model, test_loader, test_dataset, all_labels, all_predictions, class_names
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise e

def create_prediction_grid(model, test_dataset, class_names):
    """Create a 10x3 grid of test images with predictions"""
    # Set model to evaluation mode
    model.eval()
    
    # Define transform for visualization
    vis_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Sample indices for the grid
    num_samples = min(30, len(test_dataset))  # 10x3 grid needs 30 images
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Create figure for the grid
    plt.figure(figsize=(15, 25))
    
    # Create lists to store images and captions for wandb
    wandb_images = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and label
            original_image, label = test_dataset[idx]
            
            # For visualization, we need the unnormalized image
            img_path = test_dataset.image_paths[idx]
            vis_image = Image.open(img_path).convert('RGB')
            vis_tensor = vis_transform(vis_image)
            
            # Move to device and add batch dimension
            input_tensor = original_image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_tensor)
            _, prediction = output.max(1)
            prediction = prediction.item()
            
            # Plot
            plt.subplot(10, 3, i+1)
            # Convert tensor to numpy for plotting
            img_array = vis_tensor.permute(1, 2, 0).numpy()
            plt.imshow(img_array)
            
            true_class = class_names[label]
            pred_class = class_names[prediction]
            
            if label == prediction:
                color = 'green'
                caption = f"True: {true_class} | Pred: {pred_class} ✓"
            else:
                color = 'red'
                caption = f"True: {true_class} | Pred: {pred_class} ✗"
            
            plt.title(caption, color=color)
            plt.axis('off')
            
            # Add to wandb images list
            wandb_images.append(wandb.Image(img_array, caption=caption))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('prediction_grid.png')
    
    # Log the figure to wandb
    wandb.log({"prediction_grid": wandb.Image('prediction_grid.png')})
    
    # Also log the individual images with captions
    wandb.log({"test_predictions": wandb_images})

def create_confusion_matrix(all_labels, all_predictions, class_names):
    """Create and log a confusion matrix"""
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('confusion_matrix.png')
    
    # Log the figure to wandb
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    
    # Also log a summary of class-wise accuracies
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, (class_name, accuracy) in enumerate(zip(class_names, class_accuracy)):
        wandb.log({f"class_accuracy/{class_name}": accuracy})

def generate_classification_report(all_labels, all_predictions, class_names):
    """Generate and log classification report"""
    from sklearn.metrics import classification_report
    
    # Generate report
    report = classification_report(all_labels, all_predictions, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Log to wandb
    for class_name in class_names:
        if class_name in report:
            wandb.log({
                f"metrics/{class_name}/precision": report[class_name]['precision'],
                f"metrics/{class_name}/recall": report[class_name]['recall'],
                f"metrics/{class_name}/f1-score": report[class_name]['f1-score']
            })
    
    # Log overall metrics
    wandb.log({
        "metrics/accuracy": report['accuracy'],
        "metrics/macro_avg_precision": report['macro avg']['precision'],
        "metrics/macro_avg_recall": report['macro avg']['recall'],
        "metrics/macro_avg_f1": report['macro avg']['f1-score'],
        "metrics/weighted_avg_precision": report['weighted avg']['precision'],
        "metrics/weighted_avg_recall": report['weighted avg']['recall'],
        "metrics/weighted_avg_f1": report['weighted avg']['f1-score']
    })

if __name__ == "__main__":
    # Evaluate model
    model, test_loader, test_dataset, all_labels, all_predictions, class_names = evaluate_model(
        sweep_id='uf2dfd5t',
        entity='da24m008-iit-madras',
        project='DA6401-A2-V4'
    )
    
    # Create prediction grid
    create_prediction_grid(model, test_dataset, class_names)
    
    # Create confusion matrix
    create_confusion_matrix(all_labels, all_predictions, class_names)
    
    # Generate classification report
    generate_classification_report(all_labels, all_predictions, class_names)
    
    print("Evaluation complete. Results logged to Weights & Biases.")