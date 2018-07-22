from setuptools import setup

setup(
    name="suss",
    version="0.1-alpha",
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
        "numpy>=1.14.0",
        "Cython>=0.28.2",
        # "hdbscan==0.8.13",
        "matplotlib>=2.0.2",
        "scipy>=0.19.0",
        "scikit-learn==0.20dev0",
        "MulticoreTSNE==0.1",
        "networkx==2.1",
    ],
    dependency_links=[
        "git+https://github.com/scikit-learn/scikit-learn.git@813d7d#egg=scikit-learn-0.20dev",
        "git+https://github.com/DmitryUlyanov/Multicore-TSNE.git@d4ff4a#egg=multicoretsne-0.1"
    ],
    extras_require={
        "notebooks": ["jupyter==1.0.0"],
        "gui": ["PyQt5==5.10.1"],
    }
)
