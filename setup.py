#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict

from setuptools import find_packages, setup

# Package meta-data.
NAME = "pose_estimation"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = ""

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy<=1.23.0",
    "einops==0.6.1",
    "timm==0.9.2",
    "matplotlib==3.7.1",
    "tqdm==4.65.0",
    "yacs==0.1.8",
    "numba==0.57.0",
    "filterpy==1.4.5",
    "scikit-image",
    "opencv-python==4.7.0.72",
    "ipython==8.12.2",
    "torch>=1.9.0",
    "torchvision>=0.8.0",
]

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about: Dict[str, str] = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# instruction of data including is in MANIFEST.in file
setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    python_requires=REQUIRES_PYTHON,
    url=about["__url__"],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[],
)
