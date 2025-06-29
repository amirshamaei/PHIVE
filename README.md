# PHIVE: Physics-Informed Variational Encoder for Rapid MRSI Fitting

This repository contains the PyTorch implementation for the paper:

**PHIVE: A Physics-Informed Variational Encoder Enables Rapid Spectral Fitting of Brain Metabolite Mapping at 7T**  
Amirmohammad Shamaei, Amir Bucha, Eva Niess, Lukas Hingerl, Bernhard Strasser, Aaron Osburg, Korbinian Eckstein, Wolfgang Bogner, Stanislav Motyka

PHIVE is a deep learning framework for the ultrafast quantification of Magnetic Resonance Spectroscopic Imaging (MRSI) data. It integrates a physics-based spectral model into a variational autoencoder (VAE) architecture to achieve highly accelerated, accurate, and robust metabolite mapping.

## Overview

PHIVE is a comprehensive deep learning solution for MRSI spectral fitting that addresses the challenges of automated metabolite quantification in magnetic resonance spectroscopy. The framework uses a variational autoencoder (VAE) architecture to learn the complex relationships between spectral data and metabolite concentrations.

### Key Features

- **Variational Autoencoder Architecture**: Robust deep learning model for spectral fitting
- **Multiple Encoder Types**: Support for both convolutional and transformer encoders
- **Macromolecular Modeling**: Built-in support for Lorentzian, Gaussian, and Voigt macromolecular models
- **B-spline Baseline Correction**: Flexible baseline modeling using B-splines
- **Ensemble Training**: Multi-model training for improved reliability
- **Monte Carlo Dropout**: Uncertainty quantification through MC dropout
- **CRLB Calculation**: Cramér-Rao Lower Bound estimation for quality assessment
- **Data Augmentation**: On-the-fly spectral augmentation during training
- **Quality Filtering**: Automatic quality assessment and filtering
- **Multiple Data Formats**: Support for various input formats (MAT, NPY, etc.)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PHIVE.git
cd PHIVE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install additional dependencies for advanced features:
```bash
pip install wandb  # for experiment tracking
pip install nibabel  # for NIfTI file support
```

## Quick Start

### Training a Model

1. Prepare your configuration file (see `runs/exp101.json` for an example):
```json
{
  "description": "Example MRSI fitting experiment",
  "runs": [
    {
      "version": "1/",
      "data_dir": "data/training_data.npy",
      "basis_dir": "data/basis_set.mat",
      "max_epoch": 100,
      "batchsize": 32,
      "enc_type": "conv",
      "MM": true,
      "MM_type": "param"
    }
  ]
}
```

2. Run training:
```bash
python main.py --mode train --json_file runs/exp101.json
```

### Testing a Model

```bash
python main.py --mode test \
  --test_path data/test_data.npy \
  --mask_path data/mask.npy \
  --affine_path data/affine.nii
```

## Configuration

### Main Parameters

- `data_dir`: Path to training data
- `basis_dir`: Path to metabolite basis set
- `max_epoch`: Maximum training epochs
- `batchsize`: Training batch size
- `enc_type`: Encoder type (`conv` or `trans`)
- `MM`: Enable macromolecular modeling
- `MM_type`: Macromolecular model type (`param`, `single`, `combined`)

### Advanced Configuration

See the configuration files in the `runs/` directory for detailed examples of different experimental setups.

## Data Format

### Input Data

The framework supports various input formats:

- **NumPy arrays** (`.npy`): Complex-valued spectral data
- **MATLAB files** (`.mat`): Using scipy.io or mat73
- **NIfTI files** (`.nii`): For spatial data

### Basis Sets

Metabolite basis sets should be provided as:
- Complex-valued time-domain signals
- Each column represents one metabolite
- Compatible with the spectral parameters (dwell time, frequency, etc.)

## Model Architecture

### Encoder Types

1. **Convolutional Encoder** (`enc_type: "conv"`):
   - U-Net-based architecture
   - Suitable for spectral data with spatial structure

2. **Transformer Encoder** (`enc_type: "trans"`):
   - Self-attention mechanism
   - Better for capturing long-range dependencies

### Output Parameters

The model outputs:
- Metabolite concentrations (amplitudes)
- Frequency shifts
- Damping factors
- Phase corrections
- Macromolecular contributions (if enabled)
- B-spline baseline coefficients (if enabled)

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **R² Score**: Coefficient of determination
- **Pearson Correlation**: Linear correlation coefficient
- **CRLB**: Cramér-Rao Lower Bound for uncertainty estimation
- **SNR**: Signal-to-noise ratio calculation

## Examples

### Basic Training Example

```python
from engine import Engine
import json

# Load configuration
with open('runs/exp101.json', 'r') as f:
    config = json.load(f)

# Initialize engine
engine = Engine(config['runs'][0])

# Train model
engine.dotrain(ens_id=0)
```

### Testing Example

```python
# Test on new data
engine.dotest(
    test_path='data/test.npy',
    mask_path='data/mask.npy',
    affine_path='data/affine.nii'
)
```

## File Structure

```
PHIVE/
├── main.py                 # Main training/testing script
├── engine.py              # Training engine
├── Model.py               # Neural network model
├── Models/                # Model architectures
│   ├── UNET.py           # U-Net implementation
│   └── transformer.py    # Transformer implementation
├── utils/                 # Utility functions
│   ├── DataLoader_MRSI.py # Data loading utilities
│   ├── Jmrui.py          # JMrui file format support
│   └── utils.py          # General utilities
├── runs/                  # Configuration files
├── experiments/           # Experiment results
├── data/                  # Data directory
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{shamaei2024phive,
  title={PHIVE: A Physics-Informed Variational Encoder Enables Rapid Spectral Fitting of Brain Metabolite Mapping at 7T},
  author={Shamaei, Amirmohammad and Bucha, Amir and Niess, Eva and Hingerl, Lukas and Strasser, Bernhard and Osburg, Aaron and Eckstein, Korbinian and Bogner, Wolfgang and Motyka, Stanislav},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- PyTorch Lightning team for the excellent training framework
- The MRS community for valuable feedback and testing

## Contact

For questions and support, please contact:
- Email: [amirmohammad.shamaei@ucalgary.ca]

## Changelog

### Version 1.0.0
- Initial release
- VAE-based MRSI fitting
- Support for multiple encoder types
- Macromolecular modeling
- B-spline baseline correction 