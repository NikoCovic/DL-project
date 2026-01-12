import json
import matplotlib.pyplot as plt
import tyro
from typing import Set, Literal


def visualize(experiment:str,
              metrics:Set[Literal["sharpness", "spectral_norm", "eff_sharpness", "eff_spectral_norm", "update_sharpness", "update_spectral_norm",
                                    "train_loss_history", "val_loss_history", "train_acc_history", "val_acc_history", "all"]]={"all"},
              cols:int=3):
    
    results = json.load(open(f"experiments/{experiment}/results.json", "r"))
    
    metrics = list(metrics)
    if "all" in metrics:
        metrics = results["trackers"] + ["train_loss_history", "val_loss_history", "train_acc_history", "val_acc_history"]

    print(metrics)
    
    loss_acc_metrics = ["train_loss_history", "val_loss_history", "train_acc_history", "val_acc_history"]
    tracker_metrics = results["trackers"]

    n_cols = min(cols, len(metrics))
    n_rows = len(metrics) // cols + (len(metrics) < cols) + 1

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)

    for idx, metric in enumerate(metrics):
        if metric in loss_acc_metrics:
            measurements = results[metric]
            n_warmup = 0
            freq = 1
        elif metric in tracker_metrics:
            measurements = results[metric]["measurements"]
            n_warmup = results[metric]["n_warmup"]
            freq = results[metric]["freq"]
        else:
            raise Exception(f"Metric {metric} was not computed in this experiment.")

        x = [n_warmup + i*freq for i in range(len(measurements))]
        i = idx//cols
        j = idx % cols
        ax[i, j].plot(x, measurements, label=metric)
        ax[i, j].legend()
    plt.show()


if __name__ == "__main__":
    tyro.cli(visualize)