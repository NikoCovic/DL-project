import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
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
        run_df['optimizer'] = run.config.get('optimizer') # Example: grab specific config
        
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
        'SGD': 'SGD',
        'Adam': 'Adam',
    }
    df['optimizer'] = df['optimizer'].map(optimizer_rename_map).fillna(df['optimizer'])

    column_rename_map = {
        'val_acc': 'validation accuracy',
        'train_acc': 'training accuracy',
        'hessian': 'sharpness',
        'sam_sharpness': 'sam',
        'adaptive_sharpness': 'adaptive sharpness',
        'gap': 'generalization gap',
    }
    df = df.rename(columns=column_rename_map)

    print("Runs per optimizer after filtering:")
    print(df.groupby('optimizer')['run_id'].nunique())

    return df


def plot_optimizer_correlation(df, x_col, y_col='generalization gap', epoch=None, prefix=''):
    """
    Plots a scatter plot of x_col vs y_col with best fit lines, 
    colored by optimizer, including the Pearson R coefficient in the legend.
    """
    # 1. Setup the figure
    sns.set_theme(style="whitegrid")
    
    # 2. Calculate R coefficients to update legend labels
    # We create a temporary column for the legend so we don't mutate the original data
    plot_df = df.copy()
    if epoch is not None:
        plot_df = plot_df[plot_df['epoch'] == epoch]
    
    # Ensure columns exist and drop NaNs for the calculation
    plot_df = plot_df.dropna(subset=[x_col, y_col, 'optimizer'])
    
    legend_map = {}
    optimizers = plot_df['optimizer'].unique()
    
    print("\n--- Correlation Stats ---")
    for opt in optimizers:
        subset = plot_df[plot_df['optimizer'] == opt]
        
        if len(subset) > 1:
            # Calculate Pearson correlation
            r, p = stats.pearsonr(subset[x_col], subset[y_col])
            label = f"{opt} ($R={r:.2f}$)"
            print(f"{opt}: R = {r:.4f} (p={p:.4f})")
        else:
            label = opt
            print(f"{opt}: Not enough data for correlation.")
            
        legend_map[opt] = label

    # Map the new labels to a column used for 'hue'
    plot_df['Optimizer (Stats)'] = plot_df['optimizer'].map(legend_map)

    # 3. Create the plot
    # lmplot creates a scatter + regression line automatically
    g = sns.lmplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue='Optimizer (Stats)',
        height=6,
        aspect=1.5,
        scatter_kws={'alpha': 0.4, 's': 20}, # Transparent points to see density
        line_kws={'linewidth': 2}
    )

    # 4. Final Formatting
    plt.title(f"Correlation: {x_col} vs {y_col}")
    plt.subplots_adjust(top=0.9) # Adjust title space
    plt.savefig(f"sharpness_experiments/{prefix}_{epoch}_{x_col}_vs_{y_col}_correlation.pdf")


def main():
    # experiment_group="370aad54"  # Original 64 runs per optimizer, very clean, variable lr
    # experiment_group="cf7f9af4"  # 64 runs Normalized Muon and SGD with fixed lrs, only 8 epochs
    # experiment_group="05e85a1d"  # 20 runs per optimizer, 32 epochs, fixed lrs
    # experiment_group="9a609f2a"  # Tuned 4 optimizers, 64 runs per optimizer, 16 epochs, fixed lrs
    # experiment_group="76a26f61"  # Same as above, but more conservative SGD and Adam lrs, quite clean
    # df = fetch_group_logs(
    #     entity="padlex", 
    #     project_name="cifar10-airbench", 
    #     experiment_group=experiment_group
    # )
    # df = post_process(df)
    # print(df.head())


    scheduled_df = pd.read_csv(f"sharpness_experiments/370aad54.csv")
    fixed_df = pd.read_csv(f"sharpness_experiments/76a26f61.csv")

    scheduled_df = post_process(scheduled_df)
    fixed_df = post_process(fixed_df)

    plot_optimizer_correlation(scheduled_df, x_col='sharpness', y_col='generalization gap', epoch=16, prefix='scheduled')
    plot_optimizer_correlation(scheduled_df, x_col='adaptive sharpness', y_col='generalization gap', epoch=16, prefix='scheduled')

    plot_optimizer_correlation(fixed_df, x_col='sharpness', y_col='generalization gap', epoch=16, prefix='fixed')
    plot_optimizer_correlation(fixed_df, x_col='adaptive sharpness', y_col='generalization gap', epoch=16, prefix='fixed')


if __name__ == "__main__":
    main()