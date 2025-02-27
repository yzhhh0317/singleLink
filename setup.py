# setup.py
from setuptools import setup, find_packages

setup(
    name="satellite_congestion_control",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'comtypes'
    ]
)