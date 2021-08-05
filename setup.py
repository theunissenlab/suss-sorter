from setuptools import setup

setup(
    name="suss",
    version="0.2",
    packages=["suss"],
    description=("Automated spike sorting for single channel electrophysiology"),
    author="Kevin Yu",
    author_email="kvnyu@berkeley.edu",
    url="https://github.com/theunissenlab/suss-sorter",
    keywords="spike sorting electrophysiology neuroscience",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python 3",
        "Topic :: Scientific/Engineering"
    ],
    install_requires=[
        "numpy>=1.14.0",
        "Cython>=0.28.2",
        "hdbscan>=0.8.13",
        "matplotlib>=2.2.2",
        "pandas>=1.3.0",
        "scipy>=0.19.0",
        "scikit-learn>=0.21.1",
        "networkx>=2.1",
        "umap-learn>=0.3.2",
    ],
    dependency_links=[
        "git+https://github.com/magland/isosplit5_python.git@master#egg=isosplit5",
    ],
    extras_require={
        "notebooks": ["jupyter==1.0.0"],
        "gui": ["PyQt5==5.10.1"],
    }
)
