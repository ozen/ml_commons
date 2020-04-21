from ax import load
import numpy as np


def load_ax_experiment(path):
    experiment = load(path)
    best_fn = np.argmin if experiment.optimization_config.objective.minimize else np.argmax
    best_trial = best_fn([trial.objective_mean for trial in experiment.trials.values()]).item()
    best_parameters = experiment.trials[best_trial].arm.parameters
    return best_parameters, experiment
