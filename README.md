# Neural Autoencoder Equalization for 3x16 GBaud Optical Communications

Deep learning pipeline implementing neural autoencoders and machine learning baselines for channel equalization, inter-carrier interference (ICI) mitigation, and constellation demodulation in **3x16 GBaud** optical transmission systems.

## 🔬 Research Overview

In high-speed Nyquist-WDM optical links, linear and non-linear optical impairments degrade the constellation diagram. This project explores data-driven equalization using neural autoencoder architectures to recover transmitted symbols under strong inter-carrier interference and chromatic dispersion, comparing performance against classic ML classifiers.

## 📂 Repository Structure

```
├── codigo/
│   ├── autoencoder.py        # PyTorch Autoencoder architecture with Optuna hyperparameter tuning
│   ├── dataloader16gb.py     # Optical dataset loader and feature tensor preprocessing
│   ├── decisiontree_dem.py   # Decision Tree baseline demodulation
│   ├── svm_dem.py            # Support Vector Machine (SVM) demodulation baseline
│   └── grid_dem.py           # Traditional grid-based decision boundary demodulator
├── resultados/               # Evaluation metrics, loss curves, and BER comparison plots
├── requirements.txt          # Python dependencies (PyTorch, Optuna, Scikit-learn, Pandas)
└── README.md
```

## 🛠️ Models & Methodologies

1. **Autoencoder Equalizer:** Multi-layer PyTorch encoder-decoder trained to reconstruct clean constellation geometries and minimize Symbol Error Rate (SER).
2. **Hyperparameter Optimization:** Automated study using [Optuna](https://optuna.org/) for tuning latent layer dimensionality, learning rates, and activation functions.
3. **ML Benchmarks:** Evaluated against Decision Trees, Support Vector Machines (SVM), and standard Cartesian grid slicing.

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Semillero-SOFA/equalization-autoencoder-3x16gbaud.git
cd equalization-autoencoder-3x16gbaud

# Install dependencies
pip install -r requirements.txt

# Run autoencoder training with Optuna
python codigo/autoencoder.py
```

## 👥 Authors & Affiliation

- **Principal Contributors:** César Alejandro Montoya Gómez, Carlos Andrés (Caliche0)
- **Affiliation:** Semillero de Óptica y Fotónica Aplicada (SOFA) / Grupo GITA, Universidad de Antioquia (UdeA).
