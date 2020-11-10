from setuptools import setup, find_packages

setup(
    name='ml_commons',
    version='2.3.0',
    author='Yigit Ozen',
    packages=find_packages(),
    install_requires=['numpy', 'tqdm'],
)
