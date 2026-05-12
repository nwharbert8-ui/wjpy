"""Setup script for wjpy."""
from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="wjpy",
    version="0.2.0",
    description=(
        "Weighted Jaccard methodology for correlation network analysis. "
        "Bidirectional regime detection and pairing-family decomposition."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Drake H. Harbert",
    author_email="Drake@innerarchitecturellc.com",
    url="https://github.com/nwharbert8-ui/wjpy",
    project_urls={
        "Documentation": "https://github.com/nwharbert8-ui/wjpy",
        "Bug Tracker": "https://github.com/nwharbert8-ui/wjpy/issues",
        "Source": "https://github.com/nwharbert8-ui/wjpy",
        "Methodology Paper": "https://doi.org/10.5281/zenodo.19025536",
    },
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords=(
        "weighted-jaccard correlation-network regime-detection "
        "pairing-family genomics finance industrial-monitoring "
        "ecology brain-connectivity"
    ),
)
