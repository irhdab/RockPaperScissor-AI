"""
Setup script for Rock Paper Scissors AI
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rock-paper-scissors-ai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered Rock Paper Scissors game using computer vision and neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RockPaperScissor-AI-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "rps-collect-data=getData:main",
            "rps-train=train:main", 
            "rps-play=play:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning, computer-vision, game, rock-paper-scissors, tensorflow, opencv",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/RockPaperScissor-AI-/issues",
        "Source": "https://github.com/yourusername/RockPaperScissor-AI-",
        "Documentation": "https://github.com/yourusername/RockPaperScissor-AI-/blob/main/README.md",
    },
) 