import json
import matplotlib.pyplot as plt
import tyro


def visualize(experiment:str):
    results = json.load(open(experiment, "r"))
    trackers = results["trackers"]

    n_cols = min(3, len(trackers))
    n_rows = len(trackers) // 3 + 1 + (len(trackers) < 3)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)

    for idx, tracker in enumerate(trackers):
        n_warmup = results[tracker]["n_warmup"]
        freq = results[tracker]["freq"]
        measurements = results[tracker]["measurements"]
        x = [n_warmup + i*freq for i in range(len(measurements))]
        i = idx//3 + 1
        j = idx % 3
        ax[i, j].plot(x, measurements, label=tracker)
        ax[i, j].legend()
    plt.show()


if __name__ == "__main__":
    tyro.cli(visualize)