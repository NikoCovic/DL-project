import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


class EigenValueComputer():
    def __init__(self,
                 model,
                 loss_fn,
                 data,
                 k=1):
        self.model = model
        self.loss_fn = loss_fn
        self.inputs, self.targets = data
        self.v0 = None
        self.k=k
        self.eigenvalue_history = []

    def flatten_params(self, params):
        return torch.cat([p.view(-1) for p in params])
    
    def hvp(self, params, loss, v):
        # Reset the model gradients
        self.model.zero_grad()
        # Compute the gradient of the loss w.r.t the model parameters
        grads = torch.autograd.grad(loss, params, create_graph=True)
        # Flatten the gradient to a 1-D vector, and set the NaN gradients to 0
        flat_grad = self.flatten_params(grads)
        # Convert v (numpy array) into a PyTorch Tensor
        v = torch.from_numpy(v).to(flat_grad.device).float()
        # Compute the Hessian-vector product by differentiating again
        Hv = torch.autograd.grad(torch.dot(flat_grad, v), params, retain_graph=True)
        # Flatten the product
        Hv_flatten = self.flatten_params([h if h is not None else torch.zeros_like(p) for h,p in zip(Hv, params)])
        return Hv_flatten.detach().cpu().numpy()

    def make_hvp_operator(self, loss):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.model.zero_grad()
        n = sum(p.numel() for p in params)
        def mv(v):
            v = np.asarray(v, dtype=np.float32)
            return self.hvp(params, loss, v)
        return LinearOperator((n, n), matvec=mv), n

    def step(self):
        # Compute the loss
        loss = self.loss_fn(self.model(self.inputs), self.targets)
        # Create the hvp operator
        op, n = self.make_hvp_operator(loss)
        # If there is no initial guess, initialize it randomly
        if self.v0 is None:
            self.v0 = np.random.randn(n)
            self.v0 = self.v0 / np.sqrt(np.sum(self.v0**2))
        # compute largest k eigenvalues
        eigvals, eigvecs = eigsh(op, k=self.k, which='LA', v0=self.v0, tol=1e-3, maxiter=None)
        eigvals = eigvals[::-1]  # descending
        # Update v0
        self.v0 = eigvecs[:,-1]
        self.eigenvalue_history.append(eigvals)

        return eigvals
    

if __name__ == '__main__':
    from networks import MLP
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
    import torch.nn as nn
    from torch.optim import SGD, Muon, Adam, RMSprop
    import matplotlib.pyplot as plt
    from pyhessian import hessian
    from tqdm import tqdm

    device = "cuda"

    n_samples_per_class = 250
    n_classes = 4

    loss_type = "mse"

    #255 * np.array([0.49139968, 0.48215827, 0.44653124])

    cifar10 = CIFAR10("./data/", download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.45), (0.24703233, 0.24348505, 0.26158768))]))
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
        
    
    class MockDataset(Dataset):
        def __init__(self, n_samples):
            self.X = torch.rand((n_samples, 2))
            self.y = torch.cat([torch.sin(self.X[:,0:1]), torch.cos(self.X[:,1:])], dim=1) + torch.rand_like(self.X)*0.1
            self.input_dim = 2
            self.output_dim = 2

        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    cifar10_custom = C10D(imgs, targets, loss_type=loss_type)
    mock_dataset = MockDataset(n_samples=256*4)

    batch_size = n_classes*n_samples_per_class
    cifar10_loader = DataLoader(cifar10_custom, batch_size=batch_size, shuffle=False)
    batch_size_mock = len(mock_dataset)
    mock_dataset_loader = DataLoader(mock_dataset, batch_size=batch_size_mock, shuffle=False)

    mlp = MLP((3, 32, 32), cifar10_custom.output_dim).to(device)
    mlp_mock = MLP((2), output_dim=mock_dataset.output_dim, activation='tanh').to(device)

    n_epochs = 1000
    lr_gd = 2/100
    lr_muon = 2/100
    lr_adam = 0.001#2/100
    lr_rmsprop = 2e-5
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss()

    h_input, h_target = next(iter(mock_dataset_loader))
    hessian_comp = hessian(mlp, loss_fn, dataloader=cifar10_loader)#EigenValueComputer(mlp_mock, loss_fn, (h_input.to(device), h_target.to(device)))#

    #optimizers = []
    params_square = [p for p in mlp.parameters() if len(p.shape) == 2]
    params_non_square = [p for p in mlp.parameters() if len(p.shape) != 2]
    optimizer_sgd = SGD(mlp_mock.parameters(), lr=lr_gd, momentum=0.0)
    optimizer_muon = Muon(params_square, lr=lr_muon, weight_decay=0, nesterov=False)
    optimizer_adam = Adam(mlp.parameters(), lr=lr_adam)
    optimizer_rmsprop = RMSprop(mlp.parameters(), lr=lr_rmsprop)
    optimizers = [optimizer_sgd]

    eigenvalues = []
    loss_history = []

    pbar = tqdm(range(n_epochs), desc="Loss: ")
    for epoch in pbar:
        #print(f"=================== EPOCH {epoch} ===================")
        avg_loss = 0
        n = 0
        for i, (inputs, targets) in enumerate(mock_dataset_loader):
            for optimizer in optimizers:
                optimizer.zero_grad()

            #pred = mlp(inputs.to(device))
            pred = mlp_mock(inputs.to(device))
            loss = loss_fn(pred, targets.to(device))

            avg_loss += loss.item()
            n += 1

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

        evs, _ = hessian_comp.eigenvalues(top_n=3)
        #vs = hessian_comp.step()
        #eigenvalues.append(_eigenvalues[0])
        eigenvalues += [evs]

        #print(f"New Computed Eigenvalues: {evs}")
        loss_history.append(avg_loss/n)
            
        #print(f"Loss: {avg_loss/n}")
        pbar.set_description(f"Loss: {avg_loss/n}")

    fig, ax = plt.subplots(1,2)
    ax[0].plot(loss_history)
    ax[1].hlines([2/lr_gd], xmin=0, xmax=n_epochs, linestyles="--", colors="black", label='2/$\eta$ GD')
    #ax[1].hlines([2/lr_muon], xmin=0, xmax=n_epochs, linestyles="--", colors="black", label='2/$\eta$ Muon')
    #ax[1].hlines([2/lr_adam], xmin=0, xmax=n_epochs, linestyles="--", colors="black", label='2/$\eta$ Adam')
    #ax[1].hlines([2/lr_rmsprop], xmin=0, xmax=n_epochs, linestyles="--", colors="black", label='2/$\eta$ RMSprop')
    ax[1].plot(eigenvalues, label='sharpness')#eigenvalues)
    ax[1].legend()
    plt.show()