from setuptools import setup

setup(
    name="suss",
    version="0.0",
    packages=["suss"],
    description=("Automated spike sorting for single channel electrophysiology"),
    author="Kevin Yu",
    author_email="kvnyu@berkeley.edu",
    url="https://github.com/kevinyu/solid-garbonzo", 
    keywords="spike sorting electrophysiology neuroscience",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python 3",
        "Topic :: Scientific/Engineering"
    ],
    install_requires=[
        "hdbscan==0.8.13",
        "matplotlib==2.2.2",
        "numpy==1.14.3",
        "scipy==1.1.0",
        "sklearn==0.0",
    ],
    extras_require={
        "notebooks": ["jupyter==1.0.0"],
        "spike_models": ["torch==0.4.0"],
    }
)
