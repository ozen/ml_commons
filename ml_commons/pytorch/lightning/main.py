import os
import inspect

from ml_commons.pytorch.lightning.experiment import setup_experiment
from ml_commons.pytorch.lightning.experimenter import Experimenter
from ml_commons.util.stdout_capturing import CaptureStdout


def automain(function):
    """
    Sets up the experiment and automatically runs the wrapped function
    """
    if function.__module__ == "__main__":
        # ensure that automain is not used in interactive mode.
        main_filename = inspect.getfile(function)
        if main_filename == "<stdin>" or (
                main_filename.startswith("<ipython-input-")
                and main_filename.endswith(">")
        ):
            raise RuntimeError("Cannot use @automain decorator in interactive mode.")

        # setup the experiment
        model_class, data_class, cfg = setup_experiment()

        # run main function
        with CaptureStdout(os.path.join(cfg.experiment_path, 'stdout.txt')):
            function(model_class, data_class, cfg)

    return function


def optimize_and_fit():
    experimenter = Experimenter()
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.optimize_and_fit()


def fit():
    experimenter = Experimenter()
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.fit(fast_dev_run=True)


def resume():
    from argparse import ArgumentParser
    from argparse import FileType

    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('checkpoint_file', type=FileType('r'))
    args = parser.parse_args()
    config_file_path = args.config_file.name
    checkpoint_file_path = args.checkpoint_file.name

    experimenter = Experimenter(config_file_path)
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.resume(checkpoint_file_path)
