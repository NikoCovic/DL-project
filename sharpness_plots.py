import wandb
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
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

    print("Runs per optimizer after filtering:")
    print(df.groupby('optimizer')['run_id'].nunique())

    return df


def plot_optimizer_correlation(df, x_col, y_col='gap', epoch=None):
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
    plt.savefig(f"sharpness_experiments/{epoch}_{x_col}_vs_{y_col}_correlation.png")


def main():
    experiment_group="370aad54"
    # df = fetch_group_logs(
    #     entity="padlex", 
    #     project_name="cifar10-airbench", 
    #     experiment_group=experiment_group
    # )

    df = pd.read_csv(f"sharpness_experiments/{experiment_group}.csv")

    df = post_process(df)

    print(df.head())

    # plot_optimizer_correlation(df, x_col='hessian', y_col='gap', epoch=None)
    # plot_optimizer_correlation(df, x_col='sam_sharpness', y_col='gap', epoch=None)
    # plot_optimizer_correlation(df, x_col='adaptive_sharpness', y_col='gap', epoch=None)

    # plot_optimizer_correlation(df, x_col='hessian', y_col='gap', epoch=16)
    plot_optimizer_correlation(df, x_col='sam_sharpness', y_col='gap', epoch=16)
    plot_optimizer_correlation(df, x_col='adaptive_sharpness', y_col='gap', epoch=16)


if __name__ == "__main__":
    main()