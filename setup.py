import sys
import time
from setuptools import setup, find_packages

# Environment-specific dependencies.
extras = {
    'torch': ['torch', 'torchvision', 'dgl', 'dgllife'],
}

setup(
    name='brogui',
    verison='0.1.0',
    url='https://github.com/trumanw/bro-gui',
    maintainer='Chunan Wu',
    license='Apache License 2.0',
    description='A bayesian reaction optimization(BRO) GUI interface.',
    keywords=[
        'chemistry',
        'botorch',
        'bayesian optimization',
        'reaction',
        'streamlit',
    ],
    packages=find_packages(exclude=["*.tests"]),
    install_requires=[
        'streamlit>=1.11.1',
        'matplotlib>=3.5.2'
        'rdkit-pypi>=2022.03.4',
        'numpy>=1.23.1',
        'scipy>=1.9.0',
        'tqdm>=4.64.0',
        'torch>=1.12.1',
        'gpytorch>=1.8.1',
        'botorch>=0.6.5',
    ],
    python_requires='>=3.8,<3.10')