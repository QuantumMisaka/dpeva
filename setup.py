from setuptools import setup, find_packages

setup(
    name='dpeva',
    version='0.1.0',
    author='James Misaka',
    author_email='ff6757442@gmail.com',
    description='A package for Deep Potential EVolution Accelerator (DP-EVA)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "dpdata",
        "ase"
    ],
    classifiers=[  # 分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
