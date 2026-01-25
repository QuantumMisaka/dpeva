# DP-EVA
Deep Potential EVolution Accelerator

## Target:
Data-efficient concurrent learning method to accelerate the evolution of DPA LAM and other Deep Potential.

Methods used in this project:
- Data sampling based on encoder space:
- - DIRECT (from [maml](https://github.com/materialsvirtuallab/maml) package)
- - 2-DIRECT and atomic-DIECT (usage in notebook)
- Uncertainty estimation on atomic force evaluation in double variables:
- - Query-by-committee uncertainty estimation
- - Random-Network-Distillation-like uncertainty estimation

## Installation:

Note: the src libraries only maturely implemented DIRECT method now.
And the uncertainty estimation methods need more improvements.

Install the package via pip:
```bash
pip install git+https://github.com/quantummisaka/dpeva.git
```

Or clone the repository and install the package:
```bash
git clone https://github.com/quantummisaka/dpeva.git
cd dpeva
pip install -e .
```

## Usage

### Basic Workflow

All basic functions for a basic workflow are in utils.

- Parallel encoder-fixed fine-tuning: utils/dptrain
- Prediction test results in all dataset from fine-tuned models: utils/dptest
- Descriptor (encoder) generation in all dataset from fine-tuned models: utils/dpdesc
- UQ post-analysis and view based on dptest and dpdesc: utils/uq/uq-post-view.py
- Use first-principle calculation to label the new dataset: utils/fp (not used if the dataset is already labeled)

## License

This project is licensed under the LGPL-v3 License - see the LICENSE file for details.