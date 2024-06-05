import torch
import os
# import pandas as pd
# import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

def testing_loop(model_checkpoint_path, dataset_selected):
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
    results = []  # List to hold seed and accuracy tuples

    torch.manual_seed(45)  # For reproducibility
    model_checkpoint_path = os.path.join("vision", "checkpoints", "checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar")
    dataset_selected = "imagenet1k"
    
    accuracy = testing_loop(model_checkpoint_path, dataset_selected)  # Get accuracy
    print(accuracy)
    # results.append((i, accuracy))  # Append seed and accuracy

    # # Create DataFrame
    # df = pd.DataFrame(results, columns=['Seed', 'Accuracy'])
    
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Seed'], df['Accuracy'], marker='o')
    # plt.title('Model Accuracy by Seed')
    # plt.xlabel('Seed')
    # plt.ylabel('Accuracy (%)')
    # plt.grid(True)
    # plt.show()
