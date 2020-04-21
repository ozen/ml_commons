import os
import inspect
import shutil
from glob import glob
from importlib import import_module
from argparse import ArgumentParser, FileType
import yaml

from ml_commons.pytorch.lightning.util import override_lightning_logger
from ml_commons.util.path import uniquify_path
from ml_commons.util.stdout_capturing import capture_stdout


def automain(function):
    if function.__module__ == "__main__":
        # Ensure that automain is not used in interactive mode.
        main_filename = inspect.getfile(function)
        if main_filename == "<stdin>" or (
                main_filename.startswith("<ipython-input-")
                and main_filename.endswith(">")
        ):
            raise RuntimeError("Cannot use @automain decorator in interactive mode.")

        override_lightning_logger()

        parser = ArgumentParser()
        parser.add_argument('config_file', type=FileType('r'))
        args = parser.parse_args()
        config_file = args.config_file.name

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # set experiment name and path
        config['experiments_dir'] = config.get('experiments_dir', 'experiments')
        config['experiment_name'] = config.get('experiment_name', 'default')
        experiment_path = os.path.join(config['experiments_dir'], config['experiment_name'])
        experiment_path = os.path.abspath(uniquify_path(experiment_path))
        config['experiment_path'] = experiment_path
        os.makedirs(config['experiment_path'])

        # import experiment module
        module_parts = config['module'].split('.')
        module = import_module('.'.join(module_parts[:-1]))
        cls = getattr(module, module_parts[-1])

        # copy sources
        module_path = inspect.getfile(module)
        module_dir = os.path.dirname(module_path)
        dest_dir = os.path.join(config['experiment_path'], 'source')
        os.makedirs(dest_dir, exist_ok=True)
        for source_file in glob(os.path.join(module_dir, '*.py')):
            shutil.copy(source_file, os.path.join(dest_dir, os.path.basename(source_file)))

        # copy config file
        shutil.copy(config_file, os.path.join(config['experiment_path'], os.path.basename(config_file)))

        # run main function
        with capture_stdout(os.path.join(config['experiment_path'], 'cout.txt')):
            function(cls, config)

    return function
