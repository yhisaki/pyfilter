#!/usr/bin/env python
from setuptools import setup

setup(
    name="pyfilter",
    version="1.0.0",
    install_requires=["torch", "numpy", "matplotlib"],
    extras_require={"dev": ["black", "flake8", "isort"]},
    author="hisaki",
    packages=["pyfilter"],
)
