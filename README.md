# README

## Overview

This repository contains the codebase for training an encoder (deep neural network, e.g., ResNet) on the ImageNet dataset using InfoNCE contrastive loss and evaluating its K-nearest neighbor classification performance on various out-of-distribution (OOD) datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/arashkhoeini/InfoNCE_OOD.git
cd InfoNCE_OOD
pip install -r requirements.txt
```

## Usage

### Training

To train the encoder on the ImageNet dataset, run the following command:

```bash
python train.py main_pretrain.py --dataset.root=/path/to/datasets/root
```

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.