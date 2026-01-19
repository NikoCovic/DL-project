import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
from cycler import cycler




class EOSVisualizer():
    """ Given a dict of experiments create plots.
        keys: optimizer name
        values: list of experiment names corresponding to the folder name where the .json is stored
        {"adam":["experimentA", "eperimentB"], "rmsprop":[], "muon":[]}"""
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
            filename = f"plots/eos/{model_size}/train_loss_history/train_loss_history_{optim_name}_{model_size}_{yscale}.pdf"
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
            filename = f"plots/eos/{model_size}/sharpness/sharpness_{optim_name}_{model_size}.pdf"
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
            filename = f"plots/eos/{model_size}/eff_sharpness/eff_sharpness_{optim_name}_{model_size}.pdf"
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
            filename = f"plots/eos/{model_size}/combined/combined_{optim_name}_{model_size}_{yscale}.pdf"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Combined figure saved: {filename}")
# end of EOSVisualizer

# Global Methods: 
def plot_all_eos():
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

class Logfileparser:
    def __init__(self, logfile):
        self.logfile = logfile
        self.parsed_data = {}
        self.global_stats = []
        self.table_columns = [
            "Optimizer", "Row", "Kendall_tau", "Pearson_r", 
            "TrainLoss", "TestLoss", "TrainAcc", "TestAcc", 
            "LossGap", "AccGap", "Sharpness", "Model"
        ]
        if os.path.exists(logfile):
            self.parse_log()

    def parse_log(self):
        """
        Parses the logfile into a dictionary of DataFrames keyed by rho.
        """
        current_rho = None
        current_rho_stats = {}
        
        # Structure: { 'OptimizerName': {'rows': [], 'stats': {}} }
        temp_optimizer_data = {} 
        
        # Regex Patterns
        rho_pattern = re.compile(r"^rho=([0-9\.]+): (.+)")
        row_pattern = re.compile(r"^(\w+)\s+(run)\s+([\d\.\-\s]+)\s+(.*model\.pt)")
        opt_stat_pattern = re.compile(r"^(\w+)\s+(kendall_tau_\w+|pearson_r_\w+)\s+([-\d\.]+)")
        final_stats_start_pattern = re.compile(r"^(best_rho=|Optimizer\s+Kendall_tau\s+Pearson_r)")

        with open(self.logfile, 'r') as f:
            # simple iterator to allow linear processing
            line_iter = iter(f)
            
            for line in line_iter:
                line = line.strip()
                if not line:
                    continue

                # 1. Global Stats (End of file detection)
                if final_stats_start_pattern.match(line):
                    self.global_stats.append(line)
                    # Consume the rest of the file
                    for remaining_line in line_iter:
                        if remaining_line.strip():
                            self.global_stats.append(remaining_line.strip())
                    break 

                # 2. New Rho Section
                rho_match = rho_pattern.match(line)
                if rho_match:
                    # Finalize previous block
                    if current_rho is not None:
                        self._finalize_rho_block(current_rho, current_rho_stats, temp_optimizer_data)

                    # Start new block
                    current_rho = float(rho_match.group(1))
                    current_rho_stats = self._parse_inline_stats(rho_match.group(2))
                    temp_optimizer_data = {} # Reset buffer
                    continue

                # 3. Data Rows & Optimizer Stats
                if current_rho is not None:
                    # A. Standard Table Row
                    row_match = row_pattern.match(line)
                    if row_match:
                        opt = row_match.group(1)
                        row_type = row_match.group(2)
                        metrics_str = row_match.group(3)
                        model_path = row_match.group(4)
                        
                        metrics = [float(x) for x in metrics_str.split()]
                        
                        # Pad with None for Kendall/Pearson columns which are empty in 'run' rows
                        formatted_row = [opt, row_type, None, None] + metrics + [model_path]
                        
                        # Initialize dictionary structure if new optimizer
                        if opt not in temp_optimizer_data:
                            temp_optimizer_data[opt] = {'rows': [], 'stats': {}}
                            
                        temp_optimizer_data[opt]['rows'].append(formatted_row)
                        continue

                    # B. Optimizer Specific Stats
                    stat_match = opt_stat_pattern.match(line)
                    if stat_match:
                        opt = stat_match.group(1)
                        k = stat_match.group(2)
                        v = float(stat_match.group(3))
                        
                        if opt not in temp_optimizer_data:
                            temp_optimizer_data[opt] = {'rows': [], 'stats': {}}
                            
                        temp_optimizer_data[opt]['stats'][k] = v
                        continue

        # Finalize the last block found in the file
        if current_rho is not None:
            self._finalize_rho_block(current_rho, current_rho_stats, temp_optimizer_data)

        return self.parsed_data

    def _parse_inline_stats(self, stats_str):
        stats = {}
        for item in stats_str.split(','):
            if '=' in item:
                k, v = item.split('=', 1)
                try:
                    stats[k.strip()] = float(v)
                except ValueError:
                    stats[k.strip()] = v
        return stats

    def _finalize_rho_block(self, rho, stats, data_dict):
        """Converts raw rows into DataFrames and stores in self.parsed_data."""
        self.parsed_data[rho] = {
            'section_stats': stats,
            'optimizers': {}
        }
        
        for opt, content in data_dict.items():
            df = pd.DataFrame(content['rows'], columns=self.table_columns)
            self.parsed_data[rho]['optimizers'][opt] = {
                'table': df,
                'stats': content['stats']
            }

    def get_viable_rho_and_optimizers(self):
        """
        Returns a dictionary summarizing available data.
        Format: { rho_value: [list_of_optimizers] }
        """
        summary = {}
        for rho, content in self.parsed_data.items():
            opts = list(content['optimizers'].keys())
            summary[rho] = opts
        return summary

    def get_optimizer_table(self, rho, optimizer):
        try:
            return self.parsed_data[rho]['optimizers'][optimizer]['table']
        except KeyError:
            print(f"Error: Could not find data for Rho={rho}, Optimizer={optimizer}")
            return None


class LogfileVisualizer(Logfileparser):
    def __init__(self, logfile):
        super().__init__(logfile)

        self.params_plot = {
            "font.family": "serif",
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
            "figure.dpi": 300,
            # Custom color cycle for distinct optimizer colors
            "axes.prop_cycle": cycler(color=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])        
        }
    
    def plot_aggregated_comparison(self, rho, x_col, y_col, linear_regression=True, fig_size=4.0):
        """
        Plots a scatter plot comparing ALL optimizers for a specific rho.
        This helps visualize the cluster separation between optimizers.
        """
        plt.rcParams.update(self.params_plot)

        available_opts = self.get_viable_rho_and_optimizers().get(rho, [])
        
        if not available_opts:
            print(f"Error: No data found for Rho={rho}")
            return


        fig, ax = plt.subplots(figsize=(1.61 * fig_size, fig_size))
        
        valid_data_found = False

        for optimizer in available_opts:
            df = self.get_optimizer_table(rho, optimizer)

            if df is None or x_col not in df.columns or y_col not in df.columns:
                continue

            plot_data = df[[x_col, y_col]].dropna()
            if plot_data.empty:
                continue
            valid_data_found = True

            x = plot_data[x_col]
            y = plot_data[y_col]

            sc = ax.scatter(x, y, alpha=0.6, edgecolors='w', s=40, label=optimizer)

            if linear_regression==True:
                color = sc.get_facecolor()[0]

                # 2. Calculate Regression and Shading
                if len(x) > 1:
                    # A. Fit the line
                    m, b = np.polyfit(x, y, 1)
                    
                    # B. Calculate R value
                    r_value = np.corrcoef(x, y)[0, 1]
                    
                    # C. Calculate Standard Deviation of the Residuals
                    # (How far, on average, the points deviate from the line)
                    y_pred_data = m * x + b       # Predictions for the actual data points
                    residuals = y - y_pred_data   # The errors
                    std_dev = np.std(residuals)   # Standard deviation of errors

                    # D. Create the smooth plotting line
                    x_line = np.linspace(min(x), max(x), 100)
                    y_line = m * x_line + b
                    
                    # E. Plot the regression line
                    ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1.5, 
                            label=f"Fit ($R={r_value:.2f}$)")
                            
                    # F. Add the shaded area (1 std deviation up and down)
                    ax.fill_between(x_line, 
                                    y_line - std_dev,  # Lower bound
                                    y_line + std_dev,  # Upper bound
                                    color=color, 
                                    alpha=0.2,        # Low opacity for shading
                                    linewidth=0)       # No border on the shaded blob

        if not valid_data_found:
            print(f"No valid data to plot for {x_col} vs {y_col} at Rho={rho}")
            plt.close()
            return

        # 4. Styling
        # ax.set_title(f'Optimizer Comparison (rho={rho}): {y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), frameon=True, fancybox=False, edgecolor='black', framealpha=1)
        fig.tight_layout()

        filename = f"plots/analyze_{self.logfile.split('/')[-1][:-4]}/rho_{rho}/AGGREGATED_{x_col}_vs_{y_col}.pdf"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Aggregated Figure saved in: {filename}")


# --- Main Execution Block ---
if __name__ == "__main__":
    plot_all_eos() # CALL THIS TO PLOT ALL OF EOS GRAPHS
    # PLOTTING LOGFILE GRAPHS
    # Ensure correct file path
    filename = '1390543.log'
    logfile_path = os.path.join(os.getcwd(), filename)

    if os.path.exists(logfile_path):
        viz = LogfileVisualizer(logfile_path)
        
        # Get summary of data
        data_summary = viz.get_viable_rho_and_optimizers()
        
        fig_size = 5
        if data_summary:
            print("--- Generating Aggregated Plots ---")
            for rho_key in list(data_summary.keys()):
                print(f"Processing Rho: {rho_key}")
                
                # Plot 1: Sharpness vs Loss Gap (The classic generalization plot)
                viz.plot_aggregated_comparison(rho=rho_key, x_col="Sharpness", y_col="LossGap", linear_regression=True, fig_size=fig_size)
                
                # Plot 2: Train Accuracy vs Test Accuracy (To see overfitting)
                viz.plot_aggregated_comparison(rho=rho_key, x_col="TrainAcc", y_col="TestAcc", linear_regression=True, fig_size=fig_size)

                # Plot 3: Sharpness vs Acc Gap
                viz.plot_aggregated_comparison(rho=rho_key, x_col="Sharpness", y_col="AccGap", linear_regression=True, fig_size=fig_size)
    else:
        print(f"File not found: {logfile_path}")