import wandb
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import math
import numpy as np

def fetch_group_logs(entity, project_name, experiment_group, save_path="sharpness_experiments"):
    print(f"Fetching runs from project '{project_name}' with group '{experiment_group}'...")
    
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project_name}", filters={"group": experiment_group})
    
    if not runs:
        print("No runs found with that group name.")
        return

    all_logs = []

    print(f"Found {len(runs)} runs. Downloading history...")

    for run in tqdm(runs):
        history = run.scan_history()
        
        run_df = pd.DataFrame([row for row in history])
        
        # Important metadata
        run_df['run_id'] = run.id
        run_df['run_name'] = run.name
        run_df['optimizer'] = run.config.get('optimizer') 
        
        all_logs.append(run_df)

    if all_logs:
        full_df = pd.concat(all_logs, ignore_index=True)
        
        # Save to CSV
        os.makedirs(save_path, exist_ok=True)
        full_df.to_csv(f"{save_path}/{experiment_group}.csv", index=False)
        print(f"Successfully saved {len(full_df)} rows to {save_path}/{experiment_group}.csv")
        return full_df
    else:
        print("Runs were found, but no history data was available.")
        return None


def post_process(df):
    # Filter out all 0 epochs, and missing epoch rows
    df = df[df['epoch'] > 0]
    df = df.dropna(subset=['epoch'])

    # Filter out any runs that contain a NaN in the sam_sharpness column
    valid_run_ids = df.groupby('run_id')['sam_sharpness'].apply(lambda x: x.notna().all())
    valid_run_ids = valid_run_ids[valid_run_ids].index
    df = df[df['run_id'].isin(valid_run_ids)]

    # Drop tta_val_accuracy and tta_gap columns
    df = df.drop(columns=['tta_val_accuracy', 'tta_gap'], errors='ignore')

    # Rename optimizers for pretty plotting
    optimizer_rename_map = {
        'VanillaMuon': 'Decoupled Muon',
        'NormalizedMuon': 'Normalized Muon',
        'SGD': 'Coupled SGD',
        'Adam': 'Coupled Adam',
    }
    df['optimizer'] = df['optimizer'].map(optimizer_rename_map).fillna(df['optimizer'])

    column_rename_map = {
        'val_acc': 'Validation Accuracy',
        'train_acc': 'Training Accuracy',
        'hessian': 'Raw Sharpness',
        'adaptive_sharpness': 'Adaptive Sharpness',
        'gap': 'Generalization Gap',
    }
    df = df.rename(columns=column_rename_map)

    print("Runs per optimizer after filtering:")
    print(df.groupby('optimizer')['run_id'].nunique())

    return df


def plot_combined_correlations(df, epoch=None, prefix='', y_range=None):
    """
    Plots 'sharpness' and 'adaptive sharpness' vs 'generalization gap'
    side-by-side in subplots with a shared legend and embedded stats.
    """
    # 1. Setup Data
    plot_df = df.copy()
    if epoch is not None:
        plot_df = plot_df[plot_df['epoch'] == epoch]
    
    y_col = 'Generalization Gap'
    metrics = ['Raw Sharpness', 'Adaptive Sharpness']
    
    # 2. Setup Figure and Colors
    sns.set_theme(style="whitegrid")
    optimizers = plot_df['optimizer'].unique()
    # Create a consistent color palette map
    palette = sns.color_palette(n_colors=len(optimizers))
    color_map = dict(zip(optimizers, palette))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # 3. Plotting Loop
    for ax, x_col in zip(axes, metrics):
        stats_lines = []
        
        for opt in optimizers:
            subset = plot_df[plot_df['optimizer'] == opt]
            subset = subset.dropna(subset=[x_col, y_col])
            
            if len(subset) > 1:
                # Calculate Pearson R
                r, p = stats.pearsonr(subset[x_col], subset[y_col])
                stats_lines.append(f"{opt}: $R={r:.2f}$, $P={p:.4f}$")
                
                # Plot regression line + scatter
                # using sns.regplot for calculation but manual handling for clean styling
                sns.regplot(
                    data=subset, x=x_col, y=y_col,
                    ax=ax, color=color_map[opt],
                    scatter_kws={'alpha': 0.4, 's': 30},
                    line_kws={'linewidth': 2},
                    ci=None, # Remove confidence interval shading for clarity
                    label=opt
                )
            else:
                stats_lines.append(f"{opt}: N/A")

        # 4. Formatting Axes
        ax.set_xlabel(x_col, fontsize=12)
        if y_range is not None:
            ax.set_ylim(y_range)
        if ax == axes[0]:
            ax.set_ylabel("Generalization Gap", fontsize=12)
        else:
            ax.set_ylabel("") # Hide Y label on the second plot
            ax.tick_params(left=False) # Hide Y ticks on the second plot (optional visual cleanup)

        # 5. Add Statistical Text Box
        text_str = "\n".join(stats_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='lightgray')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # 6. Shared Horizontal Legend
    # Add an outline to the markers so they look like the scatter points
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', alpha=0.4, label=opt,
               markerfacecolor=color_map[opt], markersize=10)
        for opt in optimizers
    ]
    
    fig.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.99), 
        ncol=len(optimizers), 
        frameon=False,
        fontsize=12
    )

    # 7. Final Layout Adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Leave space at the top for the legend


    
    filename = f"sharpness_experiments/{prefix}_{epoch}_combined_correlation.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")


def print_table(fixed_df, scheduled_df, epoch):
    """
    Prints a latex table with header scaling.
    
    Metrics format: (DataFrame Column, Display Name, Scale Factor)
    Example: ('sharpness', 'Sharpness', 50) will multiply value by 50 
             and append (x 10^-2) (approx) to header if it's a power of 10.
    """
    metrics = [
        # (Column, Name, Scale)
        ('Training Accuracy', 'Train Acc', 1),
        ('Validation Accuracy', 'Val Acc', 1),
        ('Generalization Gap', 'Acc Gap', 1),
        ('Raw Sharpness', 'Raw', 0.01), 
        ('Adaptive Sharpness', 'ASAM', 100),
    ]

    optimizers = fixed_df['optimizer'].unique()
    
    # 2 Left aligned columns + N Center aligned metrics
    col_def = "ll" + "c" * len(metrics) 

    print("-" * 20 + " LATEX OUTPUT " + "-" * 20)
    print(r"\begin{table}[t]")
    print(rf"\caption{{Validation metrics after {epoch} epochs.}}")
    print(r"\label{sample-table}")
    print(r"\begin{center}")
    print(r"\begin{small}")
    print(r"\begin{sc}")
    print(rf"\begin{{tabular}}{{{col_def}}}") 
    print(r"\toprule")
    
    # --- HEADER GENERATION ---
    header_labels = ["LR", "Optimizer"]
    
    for _, name, scale in metrics:
        if scale == 1:
            header_labels.append(name)
        else:
            # Calculate exponent for the label. 
            # If we multiply data by 100 (10^2), the unit is 10^-2.
            # We use math.log10 to find the inverse exponent.
            try:
                exponent = int(-math.log10(scale))
                header_labels.append(rf"{name} ($\times 10^{{{exponent}}}$)")
            except ValueError:
                # Fallback for non-log10 scales (optional)
                header_labels.append(rf"{name} ($\times {scale}$)")

    print(" & ".join(header_labels) + r" \\")
    print(r"\midrule")

    for i, (sched, df) in enumerate([('Fixed', fixed_df), ('LDS', scheduled_df)]):
        if i > 0:
            print(r"\midrule")

        for opt in optimizers:
            row = [sched, opt]
            subset = df[(df['optimizer'] == opt) & (df['epoch'] == epoch)]
            
            for metric, _, scale in metrics:
                values = subset[metric].dropna()
                if not values.empty:
                    # Apply Scaling here
                    mean = values.mean() * scale
                    std = values.std() * scale
                    
                    row.append(rf"{mean:.2f} $\pm$ {std:.3f}")
                else:
                    row.append("N/A")
            
            print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\end{center}")
    print(r"\end{table}")
    print("-" * 54)


def main():
    # experiment_group="370aad54"  
    # experiment_group="76a26f61" 
    
    # Example fetching (commented out as per original structure)
    # df = fetch_group_logs(
    #      entity="padlex", 
    #      project_name="cifar10-airbench", 
    #      experiment_group=experiment_group
    # )
    # df = post_process(df)
    
    # Load data
    try:
        scheduled_df = pd.read_csv(f"sharpness_experiments/370aad54.csv")
        fixed_df = pd.read_csv(f"sharpness_experiments/76a26f61.csv")

        scheduled_df = post_process(scheduled_df)
        fixed_df = post_process(fixed_df)

        # Generate combined plots
        plot_combined_correlations(scheduled_df, epoch=16, prefix='scheduled', y_range=(0, 0.12))
        plot_combined_correlations(fixed_df, epoch=16, prefix='fixed', y_range=(0, 0.22))

        # Print latex table
        print_table(fixed_df, scheduled_df, epoch=16)
        
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        print("Please ensure the CSV files exist in the 'sharpness_experiments' folder.")


if __name__ == "__main__":
    main()