"""Setup script for Human Variability in Language Models package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="human-variability-llm",
    version="1.0.0",
    author="Mohammadmahdi Rahimi, Ori Brand",
    author_email="mohammadmahdi.edu@gmail.com",
    description="Fine-tuning language models (GPT-2, Mistral) with multiple labels to reproduce human variability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohammadmahdirahimi/human-variability-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gpt2-train=scripts.train:main",
            "gpt2-evaluate=scripts.evaluate:main",
            "gpt2-analyze=scripts.analyze:main",
        ],
    },
)
