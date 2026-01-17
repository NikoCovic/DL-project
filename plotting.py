import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler


""" Given a dict of experiments create plots.
    keys: optimizer name
    values: list of experiment names corresponding to the folder name where the .json is stored
    {"adam":["experimentA", "eperimentB"], "rmsprop":[], "muon":[]}"""


class EOSVisualizer():
    def __init__(self, experiments_dict):
        self.exp_dict = experiments_dict
        self.params_plot = {"font.family": "serif",
                            "font.serif": ["Times New Roman", "DejaVu Serif"],
                            "font.size": 12,
                            "axes.linewidth": 1.5,
                            "lines.linestyle": "-",
                            "xtick.direction": "in",
                            "ytick.direction": "in",
                            "xtick.top": True,
                            "ytick.right": True,
                            "xtick.major.size": 6,
                            "ytick.major.size": 6,
                            "figure.dpi": 150,
                            "axes.prop_cycle" : cycler(color=[
                                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                                        ])        
                            }


    def plot_train_loss(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', yscale='linear', fig_size=3.0):
        plt.rcParams.update(self.params_plot)
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            print(f"\nPlotting training loss history for {optim_name} {model_size} {yscale}...")

            fig, ax = plt.subplots(figsize=(1.61*fig_size, fig_size))
            for experiment in experiments:
                results = json.load(open(f"experiments/{experiment}/results.json", "r"))

                train_loss_history = results['train_loss_history']
                n_epochs = results['n_epochs'] 
                optim_params = results[optim_name]
                lr = optim_params['lr']
                x = np.arange(0, n_epochs)
                ax.plot(x, train_loss_history, label=fr'$\eta = {lr}$')
            ax.set_xlabel(r'Epochs')
            ax.set_ylabel(r'Training loss')
            ax.set_yscale(yscale)
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            ax.set_title(f"{optim_name.capitalize()} {model_size.capitalize()} Training Loss")
            plt.tight_layout()
            filename = f"plots/{model_size}/train_loss_history_{optim_name}_{model_size}_{yscale}.pdf"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
            print(f"Figure saved under: {filename}")


    def plot_sharpness(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', yscale='linear', fig_size=3.0):
        plt.rcParams.update(self.params_plot)
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            print(f"\nPlotting sharpness for {optim_name} {model_size} {yscale}...")

            fig, ax = plt.subplots(figsize=(1.61*fig_size, fig_size))
            for experiment in experiments:
                results = json.load(open(f"experiments/{experiment}/results.json", "r"))
                sharp_dict = results['sharpness']
                sharp = sharp_dict['measurements']
                freq = sharp_dict['freq']
                n_warmup = sharp_dict['n_warmup']
                n_epochs = results['n_epochs']
                optim_params = results[optim_name]
                lr = optim_params['lr']
                x = np.arange(0 + n_warmup, n_epochs, step=freq)
                ax.plot(x, sharp, label=fr'$\eta = {lr}$')
            ax.set_xlabel(r'Epochs')
            ax.set_ylabel(r'Sharpness')
            ax.set_yscale(yscale)
            ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            ax.set_title(f" Sharpness {optim_name.capitalize()} {model_size.capitalize()}")
            plt.tight_layout()
            filename = f"plots/{model_size}/sharpness_{optim_name}_{model_size}_{yscale}.pdf"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')
            print(f"Figure saved under: {filename}")



if __name__ == "__main__":

    big_exp_dict = {}
    small_exp_dict = {}
    for exp in os.listdir('experiments'):
        if exp.endswith("small"):
            size = "small"
            base = exp[:-5]
        else:
            size = "big"
            base = exp[:-3]        
        optimizer_name = ''.join(c for c in base if c.isalpha())
        target = small_exp_dict if size == "small" else big_exp_dict
        target.setdefault(optimizer_name, []).append(exp)

    print("\nExperiment dict 'big': ", big_exp_dict)
    print("\nExperiment dict 'small': ", small_exp_dict)

    print("\nCreating plots...")
    for model_size in ['big', 'small']:
        exp_dict = big_exp_dict if model_size == 'big' else small_exp_dict
        eos_visualizer = EOSVisualizer(exp_dict)
        eos_visualizer.plot_train_loss(optim_names=list(exp_dict.keys()), model_size=model_size, yscale='linear')
        eos_visualizer.plot_train_loss(optim_names=list(exp_dict.keys()), model_size=model_size, yscale='log')
        eos_visualizer.plot_sharpness(optim_names=list(exp_dict.keys()), model_size=model_size)
    print("\nPlots complete.\n")