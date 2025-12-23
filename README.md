# Multimodal Machine Learning for Soft High-k, Low-Modulus Polymers under Data Scarcity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of our multimodal fusion framework for polymer property prediction under data scarcity conditions. Our approach leverages multiple representation learning methods and contrastive alignment to improve property prediction with limited training samples.

## Overview

Predicting polymer properties from molecular structures is challenging when training data is limited. This work addresses data scarcity through:

1. **Multimodal Representations**: Combining complementary molecular representations (sequence-based, graph-based, and fingerprint-based)
2. **Contrastive Alignment**: Property-guided contrastive learning to align heterogeneous embedding spaces
3. **Fusion Strategies**: Systematic comparison of early, late, and latent-space fusion approaches

### Key Features

- 🧬 **Multiple Encoders**: TransPolymer, PolyBERT, GIN (Graph Isomorphism Network), Morgan Fingerprints
- 🔗 **Contrastive Alignment**: Property-guided alignment of different embedding spaces
- 🔬 **GPR Prediction**: Gaussian Process Regression with optimized hyperparameters
- 📊 **Comprehensive Evaluation**: Leave-One-Out Cross-Validation (LOOCV) for robust small-data assessment
- 🎯 **Multi-Property**: Simultaneous prediction of Dielectric Constant and Young's Modulus

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── DE Data Collection.csv                 # Dataset with polymer properties
├── artifacts/                             # Pre-computed embeddings
│   ├── transPolymer_embeddings.pkl       # TransPolymer embeddings
│   ├── gin_embeddings.pkl                # GIN embeddings
│   └── Polybert_Embeddings.pkl           # PolyBERT embeddings
├── GIN_checkpoint/                        # GIN model checkpoint
├── TransPolymer_checkpoint/               # TransPolymer model checkpoint
├── GIN_Encoder.py                        # GIN-based property prediction
├── Sequence_TransPolymer.py              # TransPolymer-based prediction
├── Sequence_Polybert.py                  # PolyBERT-based prediction
├── Sequence_Morgan_Fingerprint_GRP.py    # Morgan fingerprint baseline
└── Multi_fusion.py                       # Multimodal fusion pipeline
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone this repository:

```bash
git clone https://github.com/yourusername/multimodal-polymer-prediction.git
cd multimodal-polymer-prediction
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset (`DE Data Collection.csv`) contains:

- **35 polymer samples**
- **SMILES representations** of polymer structures
- **Target properties**:
  - Dielectric Constant (k)
  - Young's Modulus (MPa)

Due to the small sample size, we use Leave-One-Out Cross-Validation (LOOCV) for evaluation.

## Usage

### 1. Single-Modality Baselines

#### Morgan Fingerprint + GPR

```bash
python Sequence_Morgan_Fingerprint_GRP.py
```

#### TransPolymer Embeddings + GPR

```bash
python Sequence_TransPolymer.py
```

#### PolyBERT Embeddings + GPR

```bash
python Sequence_Polybert.py
```

#### GIN Embeddings + GPR

```bash
python GIN_Encoder.py
```

Each script will:

1. Load pre-computed embeddings or generate them
2. Perform hyperparameter tuning via 5-fold CV
3. Evaluate using LOOCV
4. Report R² and RMSE with uncertainty estimates

### 2. Multimodal Fusion Pipeline

Run the complete fusion experiment:

```bash
python Multi_fusion.py
```

This will:

1. Load TransPolymer and GIN embeddings
2. Train contrastive alignment models (10 runs with different seeds)
3. Evaluate multiple fusion strategies:
   - **Early Fusion**: Concatenation and averaging of raw embeddings
   - **True Late Fusion**: Prediction-level fusion from separate models
   - **Latent-Space Aligned**: Contrastive-aligned embeddings with various fusion methods
4. Generate a comprehensive results table (Table 2 in paper)

## Reproducibility

### Random Seeds

All experiments use fixed random seeds for reproducibility:

- `MASTER_SEED = 42` for the main pipeline
- `RANDOM_SEED = 42` for individual baselines
- `BOOTSTRAP_SEED = 42` for uncertainty estimation

### Pre-computed Embeddings

Pre-computed embeddings are provided in `artifacts/` to ensure exact reproducibility:

- `transPolymer_embeddings.pkl`: 35 × 768 dimensional
- `gin_embeddings.pkl`: 35 × 256 dimensional
- `Polybert_Embeddings.pkl`: 35 × 768 dimensional

### Regenerating Embeddings

To regenerate embeddings from scratch:

1. **TransPolymer**: Use checkpoint in `TransPolymer_checkpoint/`
2. **GIN**: Use checkpoint in `GIN_checkpoint/`
3. **PolyBERT**: Use the publicly available pretrained PolyBERT model hosted on Hugging Face: https://huggingface.co/kuelumbus/polyBERT

## Methodology

### Property-Guided Contrastive Learning

The alignment loss encourages embeddings from different modalities to be similar when their property values are similar:

```
L = -log(Σ exp(sim(z_tp, z_gnn) / τ) × I[dist(y_i, y_j) < threshold] /
         Σ exp(sim(z_tp, z_gnn) / τ))
```

Where:

- `z_tp`, `z_gnn`: TransPolymer and GIN embeddings
- `τ`: Temperature parameter (0.10)
- `threshold`: Property distance percentile (30th)

### Evaluation Protocol

1. **Cross-Validation**: LOOCV for all experiments (critical for n=35)
2. **Metrics**:
   - R² (coefficient of determination)
   - RMSE (root mean squared error)
3. **Uncertainty Quantification**:
   - R²: Jackknife standard deviation
   - RMSE: Bootstrap standard deviation (5000 samples)
4. **Statistical Testing**: Paired t-tests between methods

## Configuration

Key hyperparameters in `Multi_fusion.py`:

```python
# Contrastive Learning
TEMPERATURE = 0.10              # Contrastive loss temperature
PROPERTY_PERCENTILE = 30        # Property similarity threshold
EPOCHS = 400                    # Training epochs
LEARNING_RATE = 5e-4            # AdamW learning rate
WEIGHT_DECAY = 1e-3             # L2 regularization

# Architecture
PROJECTION_DIM = 128            # Aligned embedding dimension
TP_HIDDEN_DIM = 128             # TransPolymer projection hidden size
GNN_HIDDEN_DIM = 256            # GIN projection hidden size
TP_DROPOUT = 0.3                # TransPolymer dropout
GNN_DROPOUT = 0.15              # GIN dropout

# Gaussian Process Regression
PCA_COMPONENTS = 20             # PCA dimensionality
GPR_RESTARTS = 10               # Optimizer restarts

# Experiment
NUM_RUNS = 10                   # Number of independent runs
MASTER_SEED = 42                # Random seed
```
