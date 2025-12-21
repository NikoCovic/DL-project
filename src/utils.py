import torch


def aaT_matrix_pow(A, pow=1/2, method="svd"):
    # Computes (AA^T)^{pow}
    if method == "svd":
        # Compute the SVD of A
        U, S, V = torch.svd(A)
        # Make sure eigen-values are at least 1e-12
        S = torch.clamp_min(S, 1e-12)
        # Return (AA^T)^{pow} = US^{2pow}U^T
        return U @ torch.diag(S.pow(2*pow)) @ U.T


def flatten_params(params):
    return torch.cat([p.view(-1) for p in params])

def hvp(v, model, params, loss):
    # Reset the model gradients
    model.zero_grad()
    # Compute the gradient of the loss w.r.t the model parameters
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # Flatten the gradient to a 1-D vector, and set the NaN gradients to 0
    flat_grad = flatten_params(grads)
    # Convert v (numpy array) into a PyTorch Tensor
    v = torch.from_numpy(v).to(flat_grad.device).float()
    # Compute the Hessian-vector product by differentiating again
    Hv = torch.autograd.grad(torch.dot(flat_grad, v), params, retain_graph=True)
    # Flatten the product
    Hv_flatten = flatten_params([h if h is not None else torch.zeros_like(p) for h,p in zip(Hv, params)])
    return Hv_flatten.detach().cpu().numpy()

def effective_hvp(v, model, params, loss, preconditioner):
    # If preconditioner is a right-preconditioner: need HPv
    if preconditioner.is_right:
        Pv = preconditioner.vect_mult(v)
        return hvp(Pv, model, params, loss)
    # If preconditioner is a left-preconditioner, need P^{-1}Hv
    else:
        P_inv = preconditioner.pow(-1)
        return P_inv @ hvp(model, params, loss)
    

def power_iteration(matvec, n, v0=None, eps=1e-12, max_iter=200):
    if v0 is None:
        v0 = torch.rand(n)
        v0 /= torch.norm(v0, 2)
    else:
        assert len(v0) == n
    
    i = 0
    diff = torch.inf
    eigvector = v0
    eigvalue = None
    while diff > eps and i < max_iter:
        mv = matvec(eigvector)
        eigvector = mv / torch.norm(mv, 2)

        if eigvalue is None:
            eigvalue = torch.dot(mv, eigvector)
        else:
            temp_eigvalue = torch.dot(mv, eigvector)
            if abs(eigvalue - temp_eigvalue) < eps:
                break

    return eigvector, eigvalue
    

if __name__ == "__main__":
    from torch.optim import Muon, SGD, Adam, RMSprop
    import torch.nn as nn
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh
    import matplotlib.pyplot as plt
    from pyhessian import hessian
    from networks import MLP
    from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms
    from tqdm import tqdm

    device = "cuda"

    lr_gd = 2/100
    lr_muon = 2/1000

    loss_fn = nn.MSELoss()

    thresh = (2+2*0.95)/lr_muon
    thresh2 = 2/lr_gd

    all_eigvals = []
    
    n_epochs = 2000

    n_samples_per_class = 250
    n_classes = 4

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
        
    model = MLP((3, 32, 32), n_classes, activation="gelu", bias=False, width=64).to(device)

    #hessian_computer = hessian(model, loss_fn, (X, y), cuda=False)

    lr_adam = 2/500
    lr_rmsprop = 2/500

    optimizer_muon = Muon(model.parameters(), lr=lr_muon, weight_decay=0, nesterov=False)
    optimizer_gd = SGD(model.parameters(), lr=lr_gd)
    optimizer_adam = Adam(model.parameters(), lr=lr_adam)
    optimizer_rmsprop = RMSprop(model.parameters(), lr=lr_rmsprop)

    cifar_dataset = C10D(imgs, targets)
    cifar_loader = DataLoader(cifar_dataset, batch_size=len(cifar_dataset), shuffle=False)

    compute_every = 100

    avg_loss = 0
    pbar = tqdm(range(n_epochs), desc="Loss: ")
    for epoch in pbar:

        avg_loss = 0
        n = 0
        for i, (x, y) in enumerate(cifar_loader):

            optimizer_rmsprop.zero_grad()

            y_pred = model(x.to(device))
            loss = loss_fn(y.to(device), y_pred)
            

            avg_loss += loss.item()
            n += 1

            loss.backward()

            optimizer_rmsprop.step()

        if (epoch+1) % compute_every == 0:
            # Construct all the preconditioners  
            '''  
            P_inv_sqrt_dict = {}
            for p in optimizer_muon.param_groups[0]['params']:
                M = optimizer_muon.state[p]['momentum_buffer'].clone().detach()

                # P^{-1} = (MM^T)^{-1/2}
                # Compute P^{-1/2} = (MM^T)^{-1/4}
                P_inv_sqrt = aaT_matrix_pow(M, pow=-0.25)
                P_inv_sqrt_dict[p] = P_inv_sqrt
            
            #print(P_inv_sqrt_dict)
            '''
            def p_inv_sqrt_v(v):
                Pv = []
                #for p in optimizer_muon.param_groups[0]['params']:
                #    n = p.numel()
                #    _v = np.reshape(v[len(Pv):len(Pv)+n], p.shape)
                #    #print(_v)
                #    _Pv = (P_inv_sqrt_dict[p].cpu() @ _v).flatten()
                #    Pv = np.concat([Pv, _Pv])
                #    #print(Pv)
                #return Pv
                #print(optimizer_rmsprop.state[list(optimizer_rmsprop.state.keys())[0]].keys())
                for p in optimizer_muon.param_groups[0]['params']:
                    n = p.numel()
                    P = (torch.sqrt((optimizer_rmsprop.state[p]['square_avg'].clone().detach())) + 1e-12).pow(0.25)
                    _v = np.reshape(v[len(Pv):len(Pv)+n], p.shape)
                    #print(_v.shape)
                    #print(P.shape)
                    _Pv = (P.cpu() * _v).flatten()
                    Pv = np.concat([Pv, _Pv])
                return Pv
            
            #loss_history.append(avg_loss/n)
                
            #print(f"Loss: {avg_loss/n}")
            
            pbar.set_description(f"Loss: {avg_loss/n}")

            x, y = next(iter(cifar_loader))
            loss2 = loss_fn(y.to(device), model(x.to(device)))
            #print(x.shape, y.shape)
            #loss = loss_fn(y.to(device), model(x.to(device)))

            def phpv(v):
                v = p_inv_sqrt_v(v)
                v = hvp(v, model, [p for p in model.parameters() if p.requires_grad], loss2)
                v = p_inv_sqrt_v(v)
                return v

            #print(phpv(np.random.rand(10*256*3)))

            def make_phpv_operator():
                params = [p for p in model.parameters() if p.requires_grad]
                n = sum(p.numel() for p in params)
                def mv(v):
                    v = np.asarray(v, dtype=np.float32)
                    return phpv(v)
                return LinearOperator((n, n), matvec=mv), n

            op, n = make_phpv_operator()
            k = 1
            v0 = np.random.randn(n)
            v0 = v0 / np.sqrt(np.sum(v0**2))
            eigvals, eigvecs = eigsh(op, k=k, which='LA', v0=v0, tol=1e-3, maxiter=None)
            #eigvals = hessian_computer.eigenvalues(top_n=1)
            eigvals = eigvals[::-1]  # descending

            all_eigvals.append(eigvals[0])

            #print(f"Epoch {i} finished", f"Loss = {loss.item()}")
            #print(eigvals)

    plt.plot([(i+1)*compute_every for i in range(len(all_eigvals))], all_eigvals)
    #plt.hlines([thresh], 0, n_epochs, color='black', linestyles="--")
    plt.show()

