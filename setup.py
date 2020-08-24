from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='0.2.0',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tqdm', 'torch', 'torchvision', 'pytorch-lightning>=0.9',
                      'ax-platform', 'pyyaml', 'yacs', 'wrapt'],
    entry_points={
        "console_scripts": [
            "fit = ml_commons.pytorch.lightning.main:fit",
            "optimize_and_fit = ml_commons.pytorch.lightning.main:optimize_and_fit",
            "resume = ml_commons.pytorch.lightning.main:resume",
        ],
    },
)
