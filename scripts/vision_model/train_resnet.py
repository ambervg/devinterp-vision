"""
    Code to train ResNet18 without pre-trained weights on ImageNet data.
    First, you need to download ImageNet data and place into dir: data/imagenet
"""

import os
import wandb
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


def training_loop(model_architecture, dataset_selected, resume_from_checkpoint=None, wandb_logging=False):
    """
    Main function to train a ResNet model on a selected dataset.

    Args:
    - model_architecture (str): The architecture of the ResNet model ('resnet18' or 'resnet50').
    - dataset_selected (str): The dataset to use ('imagenet1k' or 'cifar10').
    """

    print(f"Start training {model_architecture} on {dataset_selected}.")

    # Check if CUDA is available and set the device to GPU if it is, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # start a new wandb run to track this script
    if wandb_logging:
        wandb.init(
            # set the wandb project where this run will be logged
            project="vision",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.001,
            "architecture": model_architecture,
            "dataset": dataset_selected,
            }
        )

    # Optional: Set a seed for reproducibility
    torch.manual_seed(46)
    print(f"Random seed set to {torch.random.initial_seed()}")

    # Instantiate a ResNet model without pre-trained weights
    if model_architecture == "resnet18":
        model = models.resnet18(weights=None)
    elif model_architecture == "resnet50":
        model = models.resnet50(weights=None)
    model = model.to(device)

    # Define transformations for pre-processing images. ResNet likes 224x224 pixels as input
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the entire dataset
    if dataset_selected == "imagenet1k":
        dataset = ImageFolder(root=os.path.join(os.getcwd(), 'data', 'imagenet1k'), transform=transform)
    elif dataset_selected == "cifar10":
        dataset = ImageFolder(root=os.path.join(os.getcwd(), 'data', 'cifar-10-python', 'cifar-10-batches-py'), transform=transform)

    # Split the dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # e.g. 80% train and 20% val
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Logarithmic checkpoint taking
    num_epochs = 40  # guesstimate
    total_iterations = num_epochs * len(train_loader)
    log_indices = np.unique(np.logspace(0, np.log10(total_iterations), num=100))  # dtype=np.int

    # Define Loss Function and Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train(model, loader, optimizer, criterion, epoch):
        """
        Train the model for one epoch.

        Args:
        - model (torch.nn.Module): The model to train.
        - loader (DataLoader): DataLoader for the training data.
        - optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        - criterion (nn.Module): Loss function.
        - epoch (int): The current epoch number.

        Returns:
        - average_loss (float): Average loss for the epoch.
        - accuracy (float): Training accuracy for the epoch.
        """

        model.train()  # Set model to training mode
        running_loss = 0.0
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, (inputs, labels) in enumerate(loader):
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Check in every 200 steps
            if (i % 200 == 199) or (epoch * len(train_loader) + i + 1 in log_indices):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'Batch: {i+1}, Loss: {running_loss/100:.4f}, Accuracy: {correct_predictions/total_predictions*100:.2f}%, Timestamp: {current_time}')
                running_loss = 0.0  # Reset running loss after printing
            
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'phase_type': 'train',
                    'batch_iteration': i+1,
                    'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'random_seed': torch.random.initial_seed(),
                    # Include other information as needed
                }, filename=f"checkpoint_{datetime.now().strftime('%Y-%m-%d-%Hh%M')}_epoch_{epoch}_train_{i+1}.pth.tar")

        
        average_loss = total_loss / len(loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy

    def validate(model, loader, criterion, epoch):
        """
        Validate the model for one epoch.

        Args:
        - model (torch.nn.Module): The model to validate.
        - loader (DataLoader): DataLoader for the validation data.
        - criterion (nn.Module): Loss function.
        - epoch (int): The current epoch number.

        Returns:
        - average_loss (float): Average loss for the epoch.
        - accuracy (float): Validation accuracy for the epoch.
        """

        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():  # Disable gradient computation
            for i, (inputs, labels) in enumerate(loader):
                # Move data to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Check in every 200 steps
                if (i % 200 == 199) or (epoch * len(train_loader) + i + 1 in log_indices):
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f'Validation Batch: {i+1}, Loss: {running_loss/100:.4f}, Accuracy: {correct_predictions/total_predictions*100:.2f}%, , Timestamp: {current_time}')
                    running_loss = 0.0  # Reset running loss after printing
                
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'phase_type': 'val',
                        'batch_iteration': i+1,
                        'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'random_seed': torch.random.initial_seed(),
                        # Include other information as needed
                    }, filename=f"checkpoint_{datetime.now().strftime('%Y-%m-%d-%Hh%M')}_epoch_{epoch}_val_{i+1}.pth.tar")
        
        average_loss = total_loss / len(loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy

    def load_checkpoint(checkpoint_path, model, optimizer=None):
        """
        Load model and optimizer state from a checkpoint file.

        Args:
        - checkpoint_path (str): Path to the checkpoint file.
        - model (torch.nn.Module): The model to load the state into.
        - optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into. Defaults to None.

        Returns:
        - int: The epoch number if available, otherwise None.
        """

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # If optimizer state is saved and an optimizer is provided, load its state
            if 'optimizer_state_dict' in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Return the epoch number if it's included in the checkpoint, otherwise return None
            return checkpoint.get('epoch')
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")
            return None

    def save_checkpoint(state, filename="checkpoint.pth.tar"):
        """
        Save the current state of the model and optimizer to a checkpoint file.

        Args:
        - state (dict): State to save, including model and optimizer state dictionaries.
        - filename (str, optional): Name of the checkpoint file. Defaults to "checkpoint.pth.tar".
        """

        filepath = os.path.join(os.getcwd(), 'data', 'checkpoints', filename)
        try: 
            torch.save(state, filepath)
            print(f"Checkpoint saved to '{filepath}'")
        except Exception as e:
            print(f"Unable to save checkpoint: {e}")


    # Optional resume from checkpoint
    if resume_from_checkpoint is not None:
        # checkpoint_filename = 'checkpoint_2024-03-25-11h29_epoch_13_train_200.pth.tar' # Enter the desired checkpoint file
        checkpoint_path = os.path.join(os.getcwd(), 'data', 'checkpoints', resume_from_checkpoint) 
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
        if start_epoch is not None:
            start_epoch += 1  # Start from the next epoch
        else:
            start_epoch = 0  # Start from scratch
    else:
        start_epoch = 0  # Start from scratch

    # Train the model
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 5
    early_stop = False
    num_epochs = 60

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Mark the start of a new epoch
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start epoch [{epoch}/{start_epoch + num_epochs - 1}] at {current_time}")

        # Train and validate your model
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_accuracy = validate(model, val_loader, criterion, epoch)
        
        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the model if you want to keep the best one
            torch.save(model.state_dict(), 'best_model_state_dict.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs")
        
        # Early stopping
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            early_stop = True
            break  # Break out of the loop

        # Update the learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # for logging
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, \
                Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Learning Rate: {current_lr}')
        
        # Log metrics to wandb
        if wandb_logging:
            wandb.log({"epoch": epoch+1, "acc_train": train_accuracy, "loss_train": train_loss,
                        "acc_val": val_accuracy, "loss_val": val_loss, "lr": current_lr})

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'phase_type': 'end_epoch',
            'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'random_seed': torch.random.initial_seed(),
            # Include other information as needed
        }, filename=f"checkpoint_{datetime.now().strftime('%Y-%m-%d-%Hh%M')}_epoch_{epoch}_end.pth.tar")

    if not early_stop:
        print("Finished training without triggering early stopping")


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Train a ResNet model on a selected dataset.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet50"], help="Model architecture to use: 'resnet18' or 'resnet50'")
    parser.add_argument("--dataset", type=str, required=True, choices=["imagenet1k", "cifar10"], help="Dataset to use: 'imagenet1k' or 'cifar10'")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training from")
    parser.add_argument("--wandb_logging", type=bool, default=False, help="Bool to log training process to wandb")

    # Parse the arguments
    args = parser.parse_args()

    # Begin training
    training_loop(args.model, args.dataset, args.resume_from_checkpoint, args.wandb_logging)
