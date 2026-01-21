#!/usr/bin/env python3
"""
CAR: Cognitive Architecture with Retrieval-Based Learning
Setup script for pip installation.
"""

from setuptools import setup, find_packages

setup(
    name="car_demo",
    version="1.0.0",
    author="Yingxu Wang",
    author_email="yingxuw814@gmail.com",
    description="CAR: Cognitive Architecture with Retrieval-Based Learning for Extreme Noise Recognition",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CAR_demo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "machine-learning",
        "cognitive-architecture",
        "noise-robustness",
        "retrieval-based-learning",
        "gradient-free",
        "extreme-noise",
    ],
)
