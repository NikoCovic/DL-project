from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import numpy as np
from typing import Union, Literal


class CIFAR10Dataset(Dataset):
    def __init__(self, n_classes:int, n_samples_per_class:int, loss=Union[Literal["mse", "ce"]]):
        # Create the CIFAR-10 dataset and extract n_classes
        cifar10 = CIFAR10("./data/", download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.45), (0.24703233, 0.24348505, 0.26158768))]))
        images = []
        targets = []
        class_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        for i in range(len(cifar10)):
            img, target = cifar10[i]
            if target < n_classes and class_counts[target] < n_samples_per_class:
                images.append(img)
                targets.append(target)
                class_counts[target] += 1

        self.input_shape = (3, 32, 32)

        self.images = np.array(images).astype(np.float32)
        self.targets = np.array(targets).astype(np.float32)
        self.output_dim = np.max(targets)+1
        # Transform the targets
        if loss == "mse":
            new_targets = np.zeros((len(targets), self.output_dim))
            new_targets[np.arange(len(targets)), targets] = 1
            self.targets = new_targets.astype(np.float32)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]