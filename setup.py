from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='2.1.0',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'tqdm', 'pytorch-lightning>=1.0.2', 'pyyaml', 'yacs'],
)
