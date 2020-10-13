from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='2.0.2',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tqdm', 'torch', 'torchvision', 'pytorch-lightning>=0.9',
                      'pyyaml', 'yacs', 'wrapt'],
)
