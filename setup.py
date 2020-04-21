from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='1.0',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tqdm', 'torch', 'torchvision', 'pytorch-lightning', 'ax-platform', 'pyyaml']
)
