# 🔍 Evaluation & Analysis of Vision Models

A structured experimental framework to **evaluate computer vision models
under real-world uncertainty**, including noise, blur, confidence
calibration, and out-of-distribution behavior.

This project is designed as a **learning-focused research playground**
for understanding *why models fail*, not just how accurate they are.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   pip
-   Git

## 📦 Installation

### 1. Clone the repository

    git clone https://github.com/aiwithnik/Evaluation-Analysis-of-Vision-Models.git
    cd Evaluation-Analysis-of-Vision-Models


### 2. Create a virtual environment

    python -m venv venv
    source venv/bin/activate      # Linux / macOS
    venv\Scripts\activate         # Windows

### 3. Install dependencies

    pip install -r requirements.txt

## ⚡ Usage

This project uses **YAML-based experiment configuration** to ensure
reproducibility and clean experiment tracking.

### Configure an Experiment

    model:
      name: "EffecientNET B0"
      pretrained: true

    evaluation:
      batch_size: 16
      uncertainty_threshold: 0.7

    perturbations:
      - type: "gaussian_noise"
        severity: 2

### Run the Evaluator

    python src/evaluator.py --config configs/experiment.yaml


## 🧪 Experiments Checklist

-   Baseline Evaluation on clean data
-   Perturbation Testing (noise, blur, contrast)
-   Confidence Calibration (Reliability Diagrams)
-   Out-of-Distribution (OOD) Evaluation

## 🤝 Contributing

This is a personal research project, but suggestions and discussions
around ML evaluation are welcome.

    git checkout -b feature/AmazingFeature
    git commit -m "Add some AmazingFeature"
    git push origin feature/AmazingFeature

