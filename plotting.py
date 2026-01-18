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


    def plot_train_loss(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', yscale='linear', fig_size=4.0):
        plt.rcParams.update(self.params_plot)
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            print(f"\nPlotting training loss history for {optim_name} {model_size} {yscale}...")

            fig, ax = plt.subplots(figsize=(1.61*fig_size, fig_size))
            for experiment in experiments:
                results = json.load(open(f"experiments/{experiment}/results.json", "r"))
                lr = results[optim_name]['lr']
                x = np.arange(0, results['n_epochs'])
                ax.plot(x, results['train_loss_history'], label=fr'$\eta = {lr}$')
            ax.set_xlabel(r'epochs')
            ax.set_ylabel(r'training loss')
            ax.set_yscale(yscale)
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            # ax.set_title(f"{optim_name.capitalize()} {model_size.capitalize()} Training Loss")
            plt.tight_layout()
            filename = f"plots/{model_size}/train_loss_history/train_loss_history_{optim_name}_{model_size}_{yscale}.pdf"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Figure saved under: {filename}")


    def plot_sharpness(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', fig_size=4.0):
        plt.rcParams.update(self.params_plot)
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            print(f"\nPlotting sharpness for {optim_name} {model_size}...")

            fig, ax = plt.subplots(figsize=(1.61*fig_size, fig_size))
            for experiment in experiments:
                results = json.load(open(f"experiments/{experiment}/results.json", "r"))
                sharp_dict = results['sharpness']
                lr = results[optim_name]['lr']
                x = np.arange(0 + sharp_dict['n_warmup'], results['n_epochs'], step=sharp_dict['freq'])
                ax.plot(x, sharp_dict['measurements'], label=fr'$\eta = {lr}$')
            ax.set_xlabel(r'epochs')
            ax.set_ylabel(r'$\lambda_{max}(H_t)$')
            ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            # ax.set_title(f"Sharpness {optim_name.capitalize()} {model_size.capitalize()}")
            plt.tight_layout()
            filename = f"plots/{model_size}/sharpness/sharpness_{optim_name}_{model_size}.pdf"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Figure saved under: {filename}")

    def plot_eff_sharpness(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', fig_size=4.0):
        plt.rcParams.update(self.params_plot)
        for optim_name in optim_names:
            assert(optim_name in self.exp_dict.keys())
            experiments = self.exp_dict[optim_name]
            print(f"\nPlotting effective sharpness for {optim_name} {model_size}...")

            fig, ax = plt.subplots(figsize=(1.61*fig_size, fig_size))
            for experiment in experiments:
                results = json.load(open(f"experiments/{experiment}/results.json", "r"))
                curr_color = ax._get_lines.get_next_color()
                eff_sharp_dict = results['eff_sharpness']
                lr = results[optim_name]['lr']
                thresh = eff_sharp_dict['thresh']
                x = np.arange(0 + eff_sharp_dict['n_warmup'], results['n_epochs'], step=eff_sharp_dict['freq'])
                ax.axhline(thresh, linestyle="--", alpha=0.8, color=curr_color)
                ax.plot(x, eff_sharp_dict['measurements'], label=fr'$\eta = {lr}$', color=curr_color)
            ax.set_xlabel(r'epochs')
            ax.set_ylabel(r'$\lambda_{max}(P_tH_t)$')
            if optim_name == "muon":
                ax.set_ylim(bottom=min(eff_sharp_dict['measurements'][:200]) - 3000, top=None)
            ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
            # ax.set_title(f"Effective Sharpness {optim_name.capitalize()} {model_size.capitalize()}")
            plt.tight_layout()
            filename = f"plots/{model_size}/eff_sharpness/eff_sharpness_{optim_name}_{model_size}.pdf"
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Figure saved under: {filename}")


    def plot_combined_metrics(self, optim_names=["adam", "rmsprop", "muon"], model_size='big', yscale='linear', fig_size=4.0):
        plt.rcParams.update(self.params_plot)
        
        for optim_name in optim_names:
            experiments = self.exp_dict.get(optim_name, [])
            if not experiments:
                continue

            print(f"\nPlotting combined metrics for {optim_name} {model_size}...")
            
            # Create 3 vertical subplots with shared X-axis (Epochs)
            fig, axes = plt.subplots(3, 1, figsize=(1.61 * fig_size, fig_size * 2.1), sharex=True)
            ax_loss, ax_sharp, ax_eff_sharp = axes

            handles, labels = [], []

            for experiment in experiments:
                try:
                    with open(f"experiments/{experiment}/results.json", "r") as f:
                        results = json.load(f)
                except FileNotFoundError:
                    print(f"Warning: experiment {experiment} not found.")
                    continue

                lr = results[optim_name]['lr']
                n_epochs = results['n_epochs']
                
                # Capture one color to use for all three metrics for this specific LR
                curr_color = ax_loss._get_lines.get_next_color()

                # 1. Train Loss
                x_loss = np.arange(0, n_epochs)
                l_plot, = ax_loss.plot(x_loss, results['train_loss_history'], color=curr_color, label=fr'$\eta = {lr}$')
                
                # Collect handles for the single figure legend
                handles.append(l_plot)
                labels.append(fr'$\eta = {lr}$')

                # 2. Sharpness
                s_dict = results['sharpness']
                x_s = np.arange(s_dict['n_warmup'], n_epochs, step=s_dict['freq'])
                ax_sharp.plot(x_s, s_dict['measurements'], color=curr_color)

                # 3. Effective Sharpness
                es_dict = results['eff_sharpness']
                x_es = np.arange(es_dict['n_warmup'], n_epochs, step=es_dict['freq'])
                ax_eff_sharp.axhline(es_dict['thresh'], linestyle="--", alpha=0.9, color=curr_color)
                ax_eff_sharp.plot(x_es, es_dict['measurements'], color=curr_color)

            # --- Formatting and Aesthetics ---
            ax_loss.set_ylabel(r'training loss')
            ax_loss.set_yscale(yscale)
            
            ax_sharp.set_ylabel(r'$\lambda_{max}(H_t)$')
            
            ax_eff_sharp.set_ylabel(r'$\lambda_{max}(P_tH_t)$')
            ax_eff_sharp.set_xlabel(r'epochs')
            if min(es_dict['measurements'][:200]) <= 1000: 
                ax_eff_sharp.set_ylim(bottom=min(es_dict['measurements'][:200]) - 5000)
            else:
                ax_eff_sharp.set_ylim(bottom=min(0, -0.10 * (min(es_dict['measurements']) - abs(min(es_dict['measurements'][:200])))))
            
            # ax_loss.set_title(f"{optim_name.capitalize()} {model_size.capitalize()} Training Dynamics", pad=15)

            ax_loss.legend(handles, labels, loc='best', frameon=True, fancybox=False, edgecolor='black')
            plt.tight_layout()

            # Save the figure
            filename = f"plots/{model_size}/combined/combined_{optim_name}_{model_size}_{yscale}.pdf"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Combined figure saved: {filename}")

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
    fig_size = 4.1
    for model_size in ['big', 'small']:
        exp_dict = big_exp_dict if model_size == 'big' else small_exp_dict
        eos_visualizer = EOSVisualizer(exp_dict)
        eos_visualizer.plot_train_loss(optim_names=list(exp_dict.keys()), model_size=model_size, yscale='linear', fig_size=fig_size)
        eos_visualizer.plot_train_loss(optim_names=list(exp_dict.keys()), model_size=model_size, yscale='log', fig_size=fig_size)
        eos_visualizer.plot_sharpness(optim_names=list(exp_dict.keys()), model_size=model_size, fig_size=fig_size)
        eos_visualizer.plot_eff_sharpness(optim_names=list(exp_dict.keys()), model_size=model_size, fig_size=fig_size)
        eos_visualizer.plot_combined_metrics(optim_names=list(exp_dict.keys()), model_size=model_size, fig_size=3.9)
    print("\nPlots complete.\n")