# Evaluation & Analysis of Vision Models under Uncertainty

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational%20Mini--Project-orange)
![Focus](https://img.shields.io/badge/Focus-Model%20Behavior%20%26%20Failure%20Analysis-green)

## 📖 About The Project

**Evaluation-Analysis-of-Vision-Models** is a Machine Learning Engineering mini-project designed to explore the nuances of computer vision beyond standard accuracy metrics.

While traditional leaderboards focus on top-1 accuracy or mAP (mean Average Precision), this project focuses on **behavioral analysis**. It investigates how vision models perform when faced with uncertainty, edge cases, and noise. The goal is to understand the "why" and "how" of model failures and their potential downstream consequences in real-world systems.

### 🎯 Key Objectives
* **Beyond Accuracy:** Shift focus from performance optimization to robust evaluation.
* **Uncertainty Quantification:** Analyze how confident the model is when it is wrong (calibration).
* **Failure Mode Analysis:** Identify specific patterns where the model breaks (e.g., occlusion, lighting changes, adversarial noise).
* **Downstream Impact:** Simulate how these failures might affect a larger application logic.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Configuration:** YAML (for managing experiment parameters and model configs)
* **Data Serialization:** JSON (for logging evaluation metrics and failure reports)

---

## 📂 Project Structure

```text
Evaluation-Analysis-of-Vision-Models/
├── configs/              # YAML configuration files for experiments
│   ├── config.yaml
|   ├── paths.yaml         
│   └── model_params.yaml
├── data/                 # Sample datasets or input images 
├── notebooks/            # Jupyter notebooks for EDA and visualization
├── logs/
├── src/                  # Source code
│   ├── __init__.py
│   ├── ingestion         
│   ├── pre processing
|   ├── model             
|   ├── feature 
|   ├── evaluation
|   ├── pipelines 
│   └── utils       
├── tests/                # Output JSON logs and analysis reports
├── requirements.txt      # Python dependencies
├── run_pipeline.py       # Automate the whole pipeline
└── README.md
```
## 🚀 Getting Started

### Prerequisites

- Python **3.8 or higher**
- `pip`
- Git

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/aiwithnik/Evaluation-Analysis-of-Vision-Models.git
cd Evaluation-Analysis-of-Vision-Models
```
### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

