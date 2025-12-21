import torch
from typing import Iterable
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.optim import Muon, RMSprop, SGD
from scipy.sparse.linalg import eigsh, LinearOperator
import numpy as np


class ParameterVector():
    def __init__(self, params:Iterable[Parameter]):
        self.params = [p.clone() for p in params]
        self.dim = sum([p.numel() for p in self.params])
    
    def dot(self, v:"ParameterVector") -> float:
        prod = 0
        for p1, p2 in zip(self.params, v.params):
            prod += torch.sum(p1 * p2)
        return prod
    
    def sum(self, v:"ParameterVector", alpha:float=1, inplace:bool=False) -> "ParameterVector":
        v_sum = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v_sum.params[i] += p*alpha
        return v_sum
    
    def copy(self) -> "ParameterVector":
        params_copy = [p.clone() for p in self.params]
        return ParameterVector(params_copy)

    def norm(self) -> float:
        sum_sq = 0
        for p in self.params:
            sum_sq += torch.sum(p**2)
        return torch.sqrt(sum_sq)
    
    def mult(self, alpha:float, inplace:bool=False) ->"ParameterVector":
        v = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v.params[i] *= alpha
        return v
    
    def orthonormal(self, vectors:Iterable["ParameterVector"], inplace:bool=False) -> "ParameterVector":
        v_orthonormal = self if inplace else self.copy()
        for v in vectors:
            v_orthonormal.sum(v, alpha=-v_orthonormal.dot(v), inplace=True)
        return v_orthonormal.mult(1/v_orthonormal.norm())
    
    def random_like(params:Iterable[Parameter]):
        params_rand = [Parameter(torch.rand_like(p), requires_grad=p.requires_grad) for p in params]
        return ParameterVector(params_rand)
    

class Preconditioner:
    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def prepare(self):
        pass

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        pass


class MuonPreconditioner(Preconditioner):
    def __init__(self, muon_optim:Muon, params:Iterable[Parameter], p:float=1, power_method:str="svd"):
        #super().__init__()
        self.optim = muon_optim
        self.params = params
        self.power_method = power_method
        self.p = p
        self.P_dict = None

    def copy(self) -> "MuonPreconditioner":
        P = MuonPreconditioner(self.optim, self.params, p=self.p, power_method=self.power_method)
        P.P_dict = {}
        for p in self.P_dict:
            P.P_dict[p] = {}
            P.P_dict[p]["U"] = self.P_dict[p]["U"].clone()
            P.P_dict[p]["S"] = self.P_dict[p]["S"].clone()
            P.P_dict[p]["P_pow_p"] = self.P_dict[p]["P_pow_p"].clone()
        return P
    
    def prepare(self):
        self.P_dict = {}
        # Loop through all of the optimizers parameters
        for p in self.params:
            # Extract the momentum
            M = self.optim.state[p]['momentum_buffer'].clone().detach()
            
            if self.power_method == "svd":
                # Compute the SVD of M = USV^T
                U, S, V = torch.svd(M)
                # Make sure eigen-values are at least 1e-12
                S = torch.clamp_min(S, 1e-12)
                # Store the U and S matrices
                self.P_dict[p] = {}
                self.P_dict[p]["U"] = U
                self.P_dict[p]["S"] = S
                # Precompute the power
                # P = (MM^T)^{1/2} = (USV^TVSU^T)^{1/2} = (US^2U^T)^{1/2}  = USU^T
                # P^p = US^pU^T
                self.P_dict[p]["P_pow_p"] = U @ torch.diag(S.pow(-self.p)) @ U.T

    def pow(self, p:float, inplace:bool=False) -> "MuonPreconditioner":
        # Compute the power
        P_pow_p = self if inplace else self.copy()
        P_pow_p.p = self.p * p
        for param in self.P_dict:
            U = P_pow_p.P_dict[param]["U"]
            S = P_pow_p.P_dict[param]["S"]
            P_pow_p.P_dict[param]["P_pow_p"] = U @ torch.diag(S.pow(self.p)) @ U.T
        return P_pow_p

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        v = v if inplace else v.copy()
        for p, p_v in zip(self.P_dict.keys(), v.params):
            #print("before", p_v)
            p_v.data = self.P_dict[p]["P_pow_p"] @ p_v
            #print("dot", p_v)
            #v.params[i] = p_dot
        return v
    
class RMSpropPreconditioner(Preconditioner):
    def __init__(self, rmsprop_optim:RMSprop, params:Iterable[Parameter], p:float=1):
        self.optim = rmsprop_optim
        self.p = p
        self.params = params
        self.P_dict = None

    def prepare(self):
        self.P_dict = {}
        for p in self.params:
            # Fetch the square gradients
            M = self.optim.state[p]["square_avg"].clone().detach()

            self.P_dict[p] = {}
            # Store P^p = diag(1/(M + e))^p
            self.P_dict[p]["P_pow_p"] = torch.pow(1/(torch.sqrt(M) + 1e-8), self.p)

    def copy(self) -> "RMSpropPreconditioner":
        P_copy = RMSpropPreconditioner(self.optim, self.params, p=self.p)
        P_copy.P_dict = {}
        for p in self.P_dict:
            P_copy.P_dict[p] = {}
            P_copy.P_dict[p]["P_pow_p"] = self.P_dict[p]["P_pow_p"].clone()
        return P_copy

    def pow(self, p:float, inplace:bool=False) -> "RMSpropPreconditioner":
        P = self if inplace else self.copy()
        P.p = p*self.p
        for param in P.P_dict:
            P.P_dict[param]["P_pow_p"] = torch.pow(self.P_dict[param]["P_pow_p"], P.p)
        return P

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        v = v if inplace else v.copy()
        for p, p_v in zip(self.P_dict, v.params):
            p_v.data = self.P_dict[p]["P_pow_p"] * p_v
        return v
    
class SGDLRPreconditioner(Preconditioner):
    def __init__(self, sgd_optim:SGD, params:Iterable[Parameter], lr:float, p:float=1):
        self.optim = sgd_optim
        self.p = p
        self.lr = lr
        self.params = params

    def copy(self) -> "SGDLRPreconditioner":
        P_copy = SGDLRPreconditioner(self.optim, self.params, self.lr, p=self.p)
        return P_copy

    def pow(self, p:float, inplace:bool=False) -> "SGDLRPreconditioner":
        P = self if inplace else self.copy()
        P.p = p*self.p
        return P

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        v = v if inplace else v.copy()
        for p_v in v.params:
            p_v.data = p_v * (self.lr**self.p)
        return v


class Hessian:
    def __init__(self, model:nn.Module, data, loss_fn, device:str="cpu"):
        self.model = model
        self.inputs, self.targets = data
        self.loss_fn = loss_fn
        self.model = model
        self.dim = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.device = device

    def eigenvalues(self, max_iter:int=200, tol:float=1e-12, top_n:int=1, preconditioner:Preconditioner=None, method:str="power_iteration"):
        if preconditioner is not None:
            preconditioner.prepare()
            preconditioner_inv_sqrt = preconditioner.pow(0.5)

        eigenvectors:list[ParameterVector] = []
        eigenvalues:list[float] = []

        while len(eigenvectors) < top_n:   

            # Initialize the first eigenvector to a random normalized vector
            eigenvector:ParameterVector = ParameterVector.random_like([p for p in self.model.parameters() if p.requires_grad])
            eigenvector.mult(1/eigenvector.norm(), inplace=True)

            #print(preconditioner.P_dict)

            if method == "power_iteration":
                eigenvalue:float = None

                for i in range(max_iter):
                    # Orthonormalize the eigenvector w.r.t the already computed eigenvectors
                    eigenvector.orthonormal(eigenvectors, inplace=True)

                    # Compute P^{-1/2}HP^{-1/2}v or just Hv
                    v = eigenvector.copy()
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)
                    # Hv
                    v = self.hessian_vector_product(v)
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)

                    # Compute the eigenvalue
                    temp_eigenvalue = v.dot(eigenvector).cpu().item()

                    # Compute the eigenvector by normalizing
                    norm = v.norm().cpu().item()
                    eigenvector = v.mult(1/norm, inplace=True)

                    if eigenvalue is None:
                        eigenvalue = temp_eigenvalue
                    else:
                        if abs(eigenvalue - temp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = temp_eigenvalue
                    #print(eigenvalue)

                eigenvalues.append(eigenvalue)
                eigenvectors.append(eigenvector)
            elif method == "LA":
                # Convert the eigenvector to a flat numpy array
                v0 = []
                for p in eigenvector.params:
                    v0 += p.flatten().tolist()
                v0 = np.array(v0)

                # Create the operator
                def mv(v:np.ndarray):
                    # Reshape to match the eigenvector and create ParameterVector
                    v_torch = []
                    i = 0
                    v = np.astype(v, np.float32)
                    for p in eigenvector.params:
                        _v_p = v[i:i+p.numel()]
                        _v_p = _v_p.reshape(p.shape)
                        _v_p = Parameter(torch.tensor(_v_p).to(self.device), requires_grad=True)
                        v_torch.append(_v_p)
                        i = i+p.numel()
                    v_torch = ParameterVector(v_torch)
                    if preconditioner is not None:
                        # Compute P^{-1/2}v
                        v_torch = preconditioner.pow(-0.5).dot(v_torch)
                    # Compute Hv
                    v_torch = self.hessian_vector_product(v_torch)
                    if preconditioner is not None:
                        v_torch = preconditioner.pow(-0.5).dot(v_torch)
                    # Convert back to numpy array
                    _v = []
                    for p in v_torch.params:
                        _v += p.flatten().tolist()
                    _v = np.array(_v)
                    return _v
                
                # Create the linear operator
                n = len(v0)
                op = LinearOperator((n,n), matvec=mv)
                eigenvalues, eigenvectors = eigsh(op, k=top_n, which='LA', v0=v0, tol=tol, maxiter=None)
        
        return eigenvectors, eigenvalues

    def hessian_vector_product(self, v:ParameterVector, inplace:bool=True) -> ParameterVector:
        # Reset model gradients
        self.model.zero_grad()

        # Compute loss
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))

        # Fetch the parameters which require a gradient
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute gradient
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad = ParameterVector(grad)
        
        # Compute the Hessian-vector product by differentiating again
        Hv = torch.autograd.grad(grad.dot(v), params, retain_graph=True)
        v = v if inplace else v.copy()
        for p_v, p_Hv in zip(v.params, Hv):
            p_v.data = p_Hv.data
        return v
        

if __name__ == "__main__":
    from torch.optim import SGD
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms

    torch.manual_seed(42)

    X = torch.rand((200, 32))
    y = torch.rand((200, 10))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples_per_class = 250
    n_classes = 4

    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(3*32*32, 256, bias=False),
                          nn.GELU(),
                          nn.Linear(256, n_classes, bias=False)).to(device)#MLP((32), 10, bias=False)

    lr = 2/100

    muon_momentum=0.95

    thresh_sgd = 2/lr
    thresh_muon = (2+2*muon_momentum)/lr
    thresh_rmsprop = 2/lr

    thresh = thresh_rmsprop

    #optim = Muon(model.parameters(), lr=lr, nesterov=False, weight_decay=0, momentum=muon_momentum)
    #optim = SGD(model.parameters(), lr=lr)
    optim = RMSprop(model.parameters(), lr=lr)

    preconditioner_muon = MuonPreconditioner(optim, list(model.parameters()))
    preconditioner_rmsprop = RMSpropPreconditioner(optim, list(model.parameters()))
    preconditioner_sgd = SGDLRPreconditioner(optim, list(model.parameters()), lr=lr)
    #print(list(preconditioner_muon.params))

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
        
    dataset = C10D(imgs, targets)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    loss_fn = nn.MSELoss()

    hessian = Hessian(model, next(iter(data_loader)), loss_fn, device=device)

    n_epochs = 1000

    eigenvalues = []

    for epoch in tqdm(range(n_epochs)):

        for i, (inputs, targets) in enumerate(data_loader):
            optim.zero_grad()

            y_pred = model(inputs.to(device))
            loss = loss_fn(targets.to(device), y_pred)

            loss.backward()

            optim.step()

        eigvecs, eigvals = hessian.eigenvalues(preconditioner=preconditioner_rmsprop, method="power_iteration")

        eigenvalues.append(eigvals)

        #print(eigenvectors)
        #print(eigenvalues)

    plt.plot(eigenvalues)
    plt.hlines([thresh], xmin=0, xmax=n_epochs, colors="black", linestyles="--")
    plt.show()

        


        

    