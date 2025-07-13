# Binary Classification – Pima Indians Diabetes with a Neural Network (From Scratch)

This project implements a **binary classification model** for predicting diabetes based on the Pima Indians dataset (Kaggle), using a **Multilayer Perceptron (MLP)** coded entirely from scratch with `NumPy`.

## Dataset

- Source: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: 8 numeric variables
- Target: `Outcome` (0 = Non-diabetic, 1 = Diabetic)
- Imbalance: ≈35% diabetic

## 🛠Methodology

- Preprocessing: replacement of invalid 0s, z-score standardization
- Architecture: `[8 → 16 → 8 → 1]`, ReLU + sigmoid
- Loss: Binary cross-entropy + **L2 regularization**
- Optimizer: `SGD` and `Adam`
- Split: 60% train / 20% validation / 20% test (stratified)

## Results

| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 73%     |
| F1-score (class 1) | 0.62 |
| Overfitting     | appears after ~20 epochs |


## Improvements considered

- Add dropout or batch norm
- Try deeper networks: `[8, 32, 16, 8, 1]`
- Class balancing (SMOTE or weighting)
- Cross-validation

## Report

A full LaTeX-written report is available in [Rapport_MLP_PimaDiabetes.pdf](Rapport_MLP_PimaDiabetes.pdf), structured in IMRAD format.

## Requirements

Install required packages:

```bash
pip install -r numpy
pandas
matplotlib
seaborn
scikit-learn
