from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='dpeva',
    version='0.1.0',
    author='James Misaka',
    author_email='ff6757442@gmail.com',
    description='A package for Deep Potential EVolution Accelerator (DP-EVA)',
    long_description=long_description,
    packages=find_packages('src/dpeva'),
    package_dir={'': 'src/dpeva'},
    classifiers = [
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Environment :: Console",
    ],
    url="https://github.com/QuantumMisaka/dpeva",  
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "dpdata",
        "ase",
        "matplotlib",
        "seaborn",
    ],
    python_requires='>=3.6',
)
