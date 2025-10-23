# setup.py
from setuptools import setup, find_packages

setup(
    name="hotpot_kv_adapter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tqdm"
    ]
)