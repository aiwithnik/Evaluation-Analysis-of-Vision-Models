# 🔍 Evaluation & Analysis of Vision Models

**Understanding why vision models fail --- not just how often.**


![](https://img.shields.io/badge/Python-3.8+-blue.svg)
![](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![](https://img.shields.io/badge/Focus-Model%20Evaluation-orange.svg)
![](https://img.shields.io/badge/Status-Research%20Playground-green.svg)

## 🧠 Project Overview

This repository is a **learning-focused experimental framework** for
evaluating computer vision models under **real-world uncertainty**.

-   Noise & blur robustness
-   Confidence calibration
-   Failure mode analysis
-   Out-of-distribution (OOD) behavior
-   Model uncertainty & misconfidence

## 🚀 Getting Started


### Prerequisites

-   Python 3.8+
-   pip
-   Git
-   Optional: CUDA-enabled GPU
:::

## 📦 Installation


### 1️⃣ Clone the repository

    git clone https://github.com/aiwithnik/Evaluation-Analysis-of-Vision-Models.git
    cd Evaluation-Analysis-of-Vision-Models

### 2️⃣ Create a virtual environment

    python -m venv venv

### Activate it

    # Linux / macOS
    source venv/bin/activate

    # Windows
    venv\Scripts\activate

### 3️⃣ Install dependencies

    pip install -r requirements.txt
    
## 📁 Project Structure

    Evaluation-Analysis-of-Vision-Models/
    ├── assets/
    ├── configs/
    │   ├── paths.yaml
    │   ├── model_params.yaml
    │   └── experiment.yaml
    ├── data/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── src/
    │   ├── config/
    │   ├── models/
    │   ├── evaluation/
    │   ├── perturbations/
    │   ├── utils/
    │   └── evaluator.py
    ├── reports/
    ├── logs/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## ⚡ Usage


### 🔧 Configure an Experiment

    model:
      name: "efficientnet_b0"
      pretrained: true

    evaluation:
      batch_size: 16
      uncertainty_threshold: 0.7

    perturbations:
      - type: "gaussian_noise"
        severity: 2
      - type: "motion_blur"
        severity: 1

### ▶️ Run the Evaluator

    python src/evaluator.py --config configs/experiment.yaml

## 🧪 Experiments Checklist

-   Baseline evaluation on clean data
-   Noise & blur robustness testing
-   Confidence calibration (ECE, reliability diagrams)
-   OOD stress testing
-   Confusion matrix & per-class breakdown
-   Failure case inspection

## 📊 Outputs Generated

    reports/
    ├── metrics.json
    ├── confusion_matrix.png
    ├── reliability_diagram.png
    └── experiment_metadata.json

## 📦 Dependencies

    torch
    torchvision
    numpy
    opencv-python
    scikit-learn
    matplotlib
    seaborn
    pyyaml
    tqdm

## 🔒 Data & Git Hygiene

    data/
    checkpoints/
    logs/
    reports/
    *.pt
    *.pth
    .env
    venv/
    __pycache__/

## 📜 Philosophy

> Accuracy answers *"how often"*\
> Evaluation answers *"when, where, and why"*

Built as a research playground for understanding model behavior under
uncertainty.

