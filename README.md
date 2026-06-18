# Network Intrusion Detection using Deep Learning (MLP)

A multi-class network intrusion detection system built with **TensorFlow/Keras**, trained on the **CIC-IDS 2017** dataset. The model classifies network traffic flows into 15 categories (1 benign + 14 attack types) using a Multi-Layer Perceptron (MLP) architecture with regularization techniques and adaptive learning rate scheduling.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Techniques](#training-techniques)
- [Experimental Results](#experimental-results)
- [Per-Class Classification Report](#per-class-classification-report)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)

---

## Project Overview

Network intrusion detection is critical for cybersecurity. This project implements a deep learning approach using a fully-connected neural network (MLP) to classify network traffic as either **benign** or one of **14 attack types**. Two optimizer experiments (**Adam** vs **SGD**) are compared to identify the best training strategy.

---

## Dataset

- **Source:** [CIC-IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity
- **File:** `data/MachineLearningCSV/MachineLearningCVE/merged_all_dataset.csv`
- **Size:** ~882 MB (~2.5 million records)
- **Features:** 78 numeric network flow features (e.g., Destination Port, Flow Duration, Packet Lengths, Flow Bytes/s, etc.)
- **Classes (15):**

| # | Label | Test Samples |
|---|-------|-------------|
| 0 | BENIGN | 419,012 |
| 1 | Bot | 390 |
| 2 | DDoS | 25,603 |
| 3 | DoS GoldenEye | 2,057 |
| 4 | DoS Hulk | 34,569 |
| 5 | DoS Slowhttptest | 1,046 |
| 6 | DoS slowloris | 1,077 |
| 7 | FTP-Patator | 1,186 |
| 8 | Heartbleed | 2 |
| 9 | Infiltration | 7 |
| 10 | PortScan | 18,139 |
| 11 | SSH-Patator | 644 |
| 12 | Web Attack – Brute Force | 294 |
| 13 | Web Attack – SQL Injection | 4 |
| 14 | Web Attack – XSS | 130 |

> **Note:** The dataset is highly imbalanced — BENIGN traffic accounts for ~83% of all samples, while rare attacks like Heartbleed and Infiltration have fewer than 10 samples in the test set.

### Data Preprocessing

1. **Cleaning:** Replace `inf` values with `NaN`, then drop all rows with missing values
2. **Deduplication:** Remove duplicate rows
3. **Label Encoding:** Convert string labels to integer-encoded one-hot vectors
4. **Splitting:** 80/20 train-test split → then 80/20 train-validation split from training set (stratified)
5. **Scaling:** `StandardScaler` fit on training data only (prevents data leakage)

---

## Model Architecture

A 4-layer MLP with Batch Normalization, progressive Dropout, and L2 regularization:

```
Input (78 features)
    │
Dense(256, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.2)
    │
Dense(128, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.25)
    │
Dense(64, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.3)
    │
Dense(32, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.4)
    │
Dense(15, Softmax)
```

- **Loss Function:** Categorical Cross-Entropy
- **Class Weights:** Computed via `sklearn.utils.class_weight.compute_class_weight('balanced')` to handle class imbalance

---

## Training Techniques

| Technique | Details |
|-----------|---------|
| **ReduceLROnPlateau** | Monitors `val_loss`, reduces LR by factor of 0.5 after 2 epochs of no improvement (min LR: 1e-6) |
| **Early Stopping** | Monitors `val_loss`, patience of 5 epochs, restores best weights |
| **Gradient Clipping** | `clipnorm=1.0` on both optimizers to prevent exploding gradients |
| **L2 Regularization** | `kernel_regularizer=l2(1e-4)` on all hidden layers |
| **Batch Normalization** | Applied after every Dense layer for training stability |
| **Progressive Dropout** | Increasing dropout rate (0.2 → 0.25 → 0.3 → 0.4) through deeper layers |
| **Balanced Class Weights** | Upweights rare attack classes during training |

---

## Experimental Results

Two experiments were conducted comparing **Adam** and **SGD (with momentum)** optimizers:

### Experiment 1 — Adam Optimizer

- **Learning Rate:** 0.001 (with ReduceLROnPlateau)
- **Epochs:** 20 | **Batch Size:** 256

### Experiment 2 — SGD Optimizer (with Momentum)

- **Learning Rate:** 0.001 | **Momentum:** 0.9
- **Epochs:** 20 (early stopped at epoch 15) | **Batch Size:** 256

### Results Comparison

| Metric | Adam | SGD |
|--------|------|-----|
| **Accuracy** | **0.9507** | 0.9315 |
| **F1 Score** | **0.9612** | 0.9462 |
| **Precision** | **0.9767** | 0.9682 |
| **Recall** | **0.9507** | 0.9315 |
| **Loss** | **0.2282** | 0.3347 |

> **Winner: Adam** — Selected as the best model based on highest test accuracy (95.07%).

---

## Per-Class Classification Report

Best model (Adam) classification report on the test set:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BENIGN | 1.00 | 0.94 | 0.97 | 419,012 |
| Bot | 0.04 | 0.99 | 0.08 | 390 |
| DDoS | 1.00 | 1.00 | 1.00 | 25,603 |
| DoS GoldenEye | 0.79 | 1.00 | 0.88 | 2,057 |
| DoS Hulk | 0.87 | 1.00 | 0.93 | 34,569 |
| DoS Slowhttptest | 0.78 | 0.99 | 0.87 | 1,046 |
| DoS slowloris | 0.81 | 0.99 | 0.89 | 1,077 |
| FTP-Patator | 0.73 | 0.99 | 0.84 | 1,186 |
| Heartbleed | 0.00 | 0.00 | 0.00 | 2 |
| Infiltration | 0.00 | 0.00 | 0.00 | 7 |
| PortScan | 0.73 | 1.00 | 0.84 | 18,139 |
| SSH-Patator | 0.56 | 0.96 | 0.71 | 644 |
| Web Attack – Brute Force | 0.12 | 0.90 | 0.21 | 294 |
| Web Attack – SQL Injection | 0.00 | 0.00 | 0.00 | 4 |
| Web Attack – XSS | 0.17 | 0.02 | 0.04 | 130 |
| | | | | |
| **Accuracy** | | | **0.95** | **504,160** |
| **Macro Avg** | 0.51 | 0.72 | 0.55 | 504,160 |
| **Weighted Avg** | 0.98 | 0.95 | 0.96 | 504,160 |

---

## Visualizations

The following plots are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `adam_accuracy.png` | Train vs Validation Accuracy over epochs (Adam) |
| `adam_loss.png` | Train vs Validation Loss over epochs (Adam) |
| `results_comparison.png` | Bar chart comparing Adam vs SGD metrics |
| `confusion_matrix.png` | 15×15 confusion matrix heatmap for the best model |

---

## Project Structure

```
DL/
├── data/
│   └── MachineLearningCSV/
│       └── MachineLearningCVE/
│           └── merged_all_dataset.csv    # CIC-IDS 2017 dataset
├── results/
│   ├── adam_accuracy.png                 # Accuracy curves
│   ├── adam_loss.png                     # Loss curves
│   ├── confusion_matrix.png             # Confusion matrix heatmap
│   └── results_comparison.png           # Adam vs SGD bar chart
├── main.py                              # Training & evaluation script
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, for faster training)

### Install Dependencies

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Download Dataset

1. Download the CIC-IDS 2017 dataset (CSV format) from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Place the merged CSV file at: `data/MachineLearningCSV/MachineLearningCVE/merged_all_dataset.csv`

---

## Usage

### Train & Evaluate

```bash
python main.py
```

The script will:
1. Load and preprocess the CIC-IDS 2017 dataset
2. Train an MLP model with Adam optimizer (Experiment 1)
3. Train an MLP model with SGD optimizer (Experiment 2)
4. Compare results and select the best model
5. Generate accuracy/loss plots, confusion matrix, and classification report
6. Save all visualizations to the `results/` directory

### Hardware Used

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
- **RAM:** 16 GB
- **Framework:** TensorFlow 2.10.0 with CUDA 11.x

Contact Information

Ahmed Elgiar
Student ID: 2023017870
Email: ahmedwaelgiar@gmail.com
Phone: +20 109 691 0575
---
