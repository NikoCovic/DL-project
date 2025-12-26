from src.networks import MLP
from torch.optim import SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from src.hessian import Hessian


def main():

    torch.manual_seed(42)

    # Use CUDA if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # How many samples are used per class and how many classes are used
    n_samples_per_class = 100
    n_classes = 2

    # Create an MLP
    model = MLP(input_shape=(3, 32, 32), n_hidden=1, width=24, output_dim=n_classes)

    # The learning rate
    lr = 2/100

    # The sharpness threshold SGD should oscillate around
    thresh_sgd = 2/lr

    # SGD optimiuer
    optim = SGD(model.parameters(), lr=lr)

    # If a different optimizer is used, specify the preconditioner here
    # NOTE: This currently does not work as intended
    #preconditioner_muon = MuonPreconditioner(optim, list(model.parameters()))
    #preconditioner_rmsprop = RMSpropPreconditioner(optim, list(model.parameters()))
    #preconditioner_sgd = SGDLRPreconditioner(optim, list(model.parameters()), lr=lr)

    # Create the CIFAR-10 dataset and extract n_classes
    cifar10 = CIFAR10("./data/", download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.45), (0.24703233, 0.24348505, 0.26158768))]))
    imgs = []
    targets = []
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for i in range(len(cifar10)):
        img, target = cifar10[i]
        if target < n_classes and class_counts[target] < n_samples_per_class:
            imgs.append(img)
            targets.append(target)
            class_counts[target] += 1

    class C10D(Dataset):
        def __init__(self, images, targets, loss_type="mse"):
            self.images = np.array(images).astype(np.float32)
            self.targets = np.array(targets).astype(np.float32)
            self.output_dim = np.max(targets)+1
            # Transform the targets
            if loss_type == "mse":
                new_targets = np.zeros((len(targets), self.output_dim))
                new_targets[np.arange(len(targets)), targets] = 1
                self.targets = new_targets.astype(np.float32)

        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.targets[idx]
        
    # Construct dataset and full-batch data loader
    dataset = C10D(imgs, targets)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Use the MSE loss
    loss_fn = nn.MSELoss()

    # Construct the Hessian class, to be used to compute the sharpness
    # This class works pretty much exactly the same as pyhessian.hessian 
    # (different inputs, but functions are the same and do the same thing)
    # The difference is that this class can also work with a Preconditioner
    hessian = Hessian(model, next(iter(data_loader)), loss_fn, device=device)

    # Number of epochs to train for
    n_epochs = 1000

    # List of all eigenvalues
    eigenvalues = []

    # Train the model
    for epoch in tqdm(range(n_epochs)):

        for i, (inputs, targets) in enumerate(data_loader):
            optim.zero_grad()

            y_pred = model(inputs.to(device))
            loss = loss_fn(targets.to(device), y_pred)

            loss.backward()

            optim.step()

        # This part compues the eigenvalues (by default top_n=1, specifying just the sharpness)
        eigvecs, eigvals = hessian.eigenvalues(preconditioner=None, method="power_iteration")

        # Store the eigenvalues
        eigenvalues.append(eigvals)

    
    # Plot the eigenvalues throughout training
    plt.plot(eigenvalues)
    plt.hlines([thresh_sgd], xmin=0, xmax=n_epochs, colors="black", linestyles="--")
    plt.show()

        
if __name__ == "__main__":
    main()
