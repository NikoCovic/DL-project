# 1. Flat Minima Hypothesis
To run the Flat Minima Hypothesis experiment (based on the airbench benchmark), run sharpness_experiment.py with an appropriate config. For example:
```
python sharpness_experiment.py --config sharpness_config_multiple_gpus.json
```

To reproduce our hyperparameter sweep run:
```
python sharpness_sweep.py
```

Our plots and table are defined in:
```
python sharpness_plots.py
```
# 2. Edge of Stability

To run the edge of stability experiment, run the eos_experiment.py script with the wanted parameters

To run Muon with the default set up and track the raw and effective sharpness:

```
uv run eos_experiment.py --optim muon --dataset cifar10 --model mlp --trackers sharpness eff_sharpness
```

To run Adam or RMSprop, simply replace `muon` with `adam` or `rmsprop` respectively.

To see the full list of available options run:

```
uv run eos_experiment.py -h
```

Some notable options are

- `<optim>.lr` - where `<optim>` should be replaced by `adam`, `rmsprop` or `muon`, defined the learning rate of the used algorithm
- `n_epochs` - the number of epochs to train for


The results will be stored in a new directory `experiments/experiments-<number>`. Inside this directory, there will be a `results.json` file with all the measurements and hyper-parameters of the experiment. Additionally, there will be plots for the train loss, train accuracy, validation loss and validation accuracy, as well as all the metrics that have been tracked using --tracker.
