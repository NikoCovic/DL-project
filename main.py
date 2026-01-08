from src.networks import MLP
from torch.optim import SGD, RMSprop, Muon, Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from src.hessian import Hessian
from src import *
import pyhessian as hes
import time


def main():

    torch.manual_seed(41)

    # Use CUDA if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # How many samples are used per class and how many classes are used
    n_samples_per_class = 250
    n_classes = 4

    # Create an MLP
    model = MLP(input_shape=(3, 32, 32), n_hidden=1, width=24, output_dim=n_classes, bias=True)

    momentum = 0.99

    # The learning rate
    lr_sgd = 2/200
    lr_rmsprop = 2e-5
    lr_muon = 2e-3
    lr_adam = 2e-3
    #lr = lr_muon

    # The sharpness threshold SGD should oscillate around
    thresh_sgd = 2/lr_sgd
    thresh_sgd_lr = 2
    thresh_rmsprop = 2/lr_rmsprop
    thresh_muon = (2 + 2*momentum)/lr_muon
    thresh_adam = (2 + 2*momentum)/lr_adam
    thresh = thresh_rmsprop

    # Optimizers
    #optim = SGD(model.parameters(), lr=lr_sgd)
    optim = RMSprop(model.parameters(), lr=lr_rmsprop)
    #optim = Adam(model.parameters(), lr=lr_adam)
    #optim = Muon(model.parameters(), lr=lr_muon, nesterov=False, weight_decay=0, momentum=momentum)

    # If a different optimizer is used, specify the preconditioner here
    # NOTE: This currently does not work as intended
    #preconditioner = None
    #preconditioner = MuonPreconditioner(optim, model)
    preconditioner = RMSpropPreconditioner(optim, model)
    #preconditioner = SGDLRPreconditioner(optim, list(model.parameters()), lr=lr)

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
    hessian_computer = Hessian(model, next(iter(data_loader)), loss_fn, device=device)
    #hessian_computer = hes.hessian(model, loss_fn, dataloader=data_loader, cuda=False)

    # Number of epochs to train for
    n_epochs = 500

    # Number of epochs to warm up for
    n_warmup = 5

    # List of all eigenvalues
    eigenvalues = []
    eigenvalues2 = []
    singularvalues = []
    commutativity_measures = []

    # Tracking the loss
    losses = []

    pbar = tqdm(range(n_epochs), desc="Loss: - | Sharpness: -")

    # Train the model
    for epoch in pbar:

        #print(f"Epoch: {epoch+1}")

        losses_b = []

        if preconditioner is not None:
            preconditioner.update_params()
        hessian_computer.update_params()

        #if epoch >= n_warmup:
        #    eigvals, eigvecs = hessian_computer.eigenvalues(top_n=1, maxIter=200, tol=1e-6)

        for i, (inputs, targets) in enumerate(data_loader):
            optim.zero_grad()

            y_pred = model(inputs.to(device))
            loss = loss_fn(targets.to(device), y_pred)

            loss.backward()

            losses_b.append(loss.item())

            optim.step()

        if epoch >= n_warmup:
            #preconditioner.prepare()
            # This part compues the eigenvalues (by default top_n=1, specifying just the sharpness)
            if preconditioner is not None:
                preconditioner.update()
            #es = hessian_computer.update_eigenvalues(preconditioner=preconditioner)
            #s = hessian_computer.update_spectral_norm(preconditioner=preconditioner)
            #_, es = hessian_computer.eigenvalues()
            c = hessian_computer.commutativity_measure(preconditioner)
            #e = abs(es[0])

            # Store the eigenvalues and singular values
            #eigenvalues.append(e)
            #singularvalues.append(s)
            commutativity_measures.append(c)

            # Store the loss
            losses.append(losses_b)

            pbar.set_description(f"Loss: {loss.item():.2e} | Commutativity: {c:.2e}")
        else: 
            pbar.set_description(f"Loss: {loss.item():.2e} | Commutativity: -")
            pass

    
    # Plot the eigenvalues throughout training
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(losses)
    #ax[1].plot(eigenvalues, label="Sharpness")
    #ax[1].plot(eigenvalues, label="Update Sharpness")
    #ax[1].plot(singularvalues, label="Spectral Norm")
    #ax[1].plot(singularvalues, label="Update Spectral Norm")
    ax[1].plot(commutativity_measures, label="Commutativity")
    #ax[1].hlines([thresh], xmin=0, xmax=n_epochs, colors="black", linestyles="--")
    ax[1].legend()
    plt.savefig(f"experiments/experiment-{time.time_ns()}.png")
    #plt.show()

        
if __name__ == "__main__":
    main()
