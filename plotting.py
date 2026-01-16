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


    def plot_train_loss(self, optim_names=["adam", "rmsprop", "muon"], fig_size=3.0):
        plt.rcParams.update(self.params_plot)
        print("Plotting training loss history...")
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            if len(experiments) >= 1: 
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
                ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                plt.tight_layout()
                filename = f"plots/train_loss_history_{optim_name}.pdf"
                directory = os.path.dirname(filename)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(filename, bbox_inches='tight')
                print(f"Figure saved under: {filename}")



if __name__ == "__main__":
    print("Creating plots")

    exp_dict = {"adam":[], "rmsprop":["experiment-1768601045502916003"], "muon":[]}
    eos_visualizer = EOSVisualizer(exp_dict)
    eos_visualizer.plot_train_loss(optim_names=["rmsprop"])