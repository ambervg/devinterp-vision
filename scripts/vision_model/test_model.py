import os
import torch
import argparse
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader


def testing_loop(model_checkpoint_path, dataset_selected):
    """
    Evaluates the performance of a pre-trained ResNet model on a selected dataset.

    Args:
        model_checkpoint_path (str): The path to the model checkpoint file. If not None, the function loads the model weights from this checkpoint.
        dataset_selected (str): The dataset to be used for evaluation. Supported values are 'imagenet1k' and 'cifar10'.

    Returns:
        float: The accuracy of the model on the validation set.

    Steps:
        1. Initializes the device to use GPU if available, otherwise CPU.
        2. Instantiates a ResNet-18 model.
        3. Loads the model weights from the checkpoint if provided.
        4. Defines image transformations for preprocessing.
        5. Loads the selected dataset (ImageNet or CIFAR-10).
        6. Splits the dataset into training and validation sets.
        7. Prints and saves the filenames of the images in the validation set.
        8. Creates data loaders for training and validation datasets.
        9. Defines a nested function to evaluate the model on the validation set.
        10. Returns the accuracy of the model on the validation set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)  # Instantiate the model here
    
    # Load the checkpoint if a model checkpoint path is provided
    if model_checkpoint_path is not None:
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Move the model to the device

    # Define transformations for pre-processing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    if dataset_selected == "imagenet1k":
        dataset_path = os.path.join(os.getcwd(), 'vision', 'imagenet', 'imagenet1k')
    elif dataset_selected == "cifar10":
        dataset_path = os.path.join(os.getcwd(), 'vision', 'cifar-10-python', 'cifar-10-batches-py')
    else:
        raise ValueError("Unsupported dataset selected.")
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # Split the dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Print the filenames of the images in the validation set
    for i in val_dataset.indices:
            print(dataset.imgs[i][0])
    with open("val_dataset_filenames.txt", "a") as f:
        for i in val_dataset.indices:
            f.write(dataset.imgs[i][0])


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    def test_model(model, dataloader):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    # Return the result of test_model
    return test_model(model, val_loader)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Train a ResNet model on a selected dataset.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, choices=["resnet18", "resnet50"], help="Path to the checkpoint file to resume training from")
    parser.add_argument("--dataset_selected", type=str, required=True, choices=["imagenet1k", "cifar10"], help="Dataset to use: 'imagenet1k' or 'cifar10'")

    # Parse the arguments
    args = parser.parse_args()

    # Begin testing
    torch.manual_seed(45)  # For reproducibility
    accuracy = testing_loop(args.model_checkpoint_path, args.dataset_selected)  # Get accuracy
    print(f"Accuracy: {accuracy}")
