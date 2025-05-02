# setup.py
from setuptools import setup, find_packages

setup(
    name="pandaprot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython>=1.79",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "py3Dmol>=0.9.0",
        "networkx>=2.6.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "pandaprot=pandaprot.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for mapping protein-protein, protein-nucleic acid, and antigen-antibody interactions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pandaprot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
)
