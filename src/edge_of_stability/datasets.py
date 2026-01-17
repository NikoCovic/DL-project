import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch.nn.functional as F
from typing import Union, Literal

class CIFAR10Dataset(Dataset):
    def __init__(
        self, 
        n_classes: int, 
        n_samples_per_class: int, 
        loss: Union[Literal["mse", "ce"]], 
        device: str = "cpu"
    ):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.49, 0.48, 0.45), (0.24, 0.24, 0.26))
        ])
        cifar10 = CIFAR10("./data/", download=True, train=True, transform=transform)
        
        images = []
        targets = []
        class_counts = {i: 0 for i in range(n_classes)}
        
        if n_classes == -1: n_classes = 10

        # Filter images and labels
        for img, target in cifar10:
            if target < n_classes and (class_counts[target] < n_samples_per_class or n_samples_per_class == -1):
                images.append(img)
                targets.append(target)
                class_counts[target] += 1
            
            if all(n_samples_per_class != -1 and count >= n_samples_per_class for count in class_counts.values()):
                break

        self.images = torch.stack(images).to(device)
        self.targets = torch.tensor(targets, dtype=torch.long).to(device)
        
        self.input_shape = (3, 32, 32)
        self.output_dim = n_classes

        if loss == "mse":
            self.targets = F.one_hot(self.targets, num_classes=n_classes).float()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Data is already on the target device
        return self.images[idx], self.targets[idx]