from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='1.0',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tqdm', 'torch', 'torchvision', 'pytorch-lightning>=0.7',
                      'ax-platform', 'pyyaml', 'yacs'],
    entry_points={
        "console_scripts": [
            "train = ml_commons.pytorch.lightning.main:train",
            "optimize_and_train = ml_commons.pytorch.lightning.main:optimize_and_train",
        ],
    },
)
