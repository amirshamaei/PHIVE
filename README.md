# PHIVE: Physics-Informed Variational Encoder for Rapid MRSI Fitting
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the PyTorch implementation for the paper:

> **PHIVE: A Physics-Informed Variational Encoder Enables Rapid Spectral Fitting of Brain Metabolite Mapping at 7T**
> Amirmohammad Shamaei, Amir Bucha, Eva Niess, Lukas Hingerl, Bernhard Strasser, Aaron Osburg, Korbinian Eckstein, Wolfgang Bogner, Stanislav Motyka  

PHIVE is a deep learning framework for the ultrafast quantification of Magnetic Resonance Spectroscopic Imaging (MRSI) data. It integrates a physics-based spectral model into a variational autoencoder (VAE) architecture to achieve highly accelerated, accurate, and robust metabolite mapping.

### Key Features
* **Ultrafast Quantification:** Processes a whole-brain MRSI dataset in ~6 milliseconds, a six-order-of-magnitude speedup over conventional methods.
* **Comprehensive Uncertainty Estimation:** Simultaneously estimates metabolite concentrations and their associated aleatoric (data-driven) and epistemic (model-driven) uncertainties.
* **Conditional Baseline Modeling:** Introduces a novel approach to dynamically control the flexibility of the spectral baseline during inference time without retraining the model.

---

### Architecture Overview

The PHIVE model consists of a convolutional encoder that maps an input spectrum into a low-dimensional latent space. A physics-informed decoder then uses parameters from this latent space to reconstruct the spectrum based on a known spectral model, including a basis set of metabolites and a flexible spline baseline.

![PHIVE Architecture](https://imgur.com/a/lbDJ8gZ)
*Fig 1: Overview of the PHIVE architecture and its application during inference.*

---

### Usage

The repository is structured to allow for easy training of new models and inference on new data.

#### Data Preparation

The model expects pre-processed MRSI data as input. Each spectrum should be a 1D vector. For volumetric data, the spectra should be flattened and processed voxel-by-voxel. You will need to prepare your data in `.npy` or a similar format that can be easily loaded with a custom PyTorch `Dataset` class.

You will also need a basis set of metabolite spectra used for the model-based decoder.

#### License
This project is licensed under the MIT License. See the LICENSE file for details.

#### Contact
For questions about the code or paper, please contact Amirmohammad Shamaei at [amirmohammad.shmaei@github.com].
