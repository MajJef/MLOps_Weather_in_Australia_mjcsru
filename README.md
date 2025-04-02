# MLOps Weather in Australia 🌦️  

This repository contains an MLOps pipeline for analyzing weather data in Australia. The goal is to build, train, and deploy a machine learning model for weather prediction.  

---

## 📂 Project Structure  


```bash
MLOps_Weather_in_Australia/
│── data/ # Raw & processed data
│ ├── raw/ # Original dataset
│ ├── processed/ # Transformed dataset
│
│── logs/ # Logs from training & processing
│── metrics/ # Evaluation results
│── notebooks/ # Jupyter notebooks for exploration
│ ├── workflow_steps.ipynb # Step-by-step notebook
│
│── src/ # Main source code
│ ├── common_utils.py # Helper functions
│ ├── config.yaml # Config file (paths & settings)
│ ├── params.yaml # Model hyperparameters
│ ├── config_manager.py # Configuration manager
│ │
│ ├── app/ # API for model inference
│ │ ├── app.py
│ │
│ ├── data_module_def/ # Data processing
│ │ ├── data_ingestion.py
│ │ ├── data_validation.py
│ │ ├── data_transformation.py
│ │ ├── schema.yaml # Dataset structure
│ │
│ ├── models_module_def/ # Model training & evaluation
│ │ ├── model_trainer.py
│ │ ├── model_evaluation.py
│ │
│ ├── pipeline_steps/ # MLOps pipeline steps
│ ├── stage01_data_ingestion.py
│ ├── stage02_data_validation.py
│ ├── stage03_data_transformation.py
│ ├── stage04_model_trainer.py
│ ├── stage05_model_evaluation.py
│
│── requirements.txt # Required Python packages
│── .gitignore # Files to ignore in Git
│── .dvcignore # Files to ignore in DVC
│── dvc.yaml # DVC pipeline configuration
│── README.md # Project documentation
│── init.py

```

---

## 🚀 Getting Started

### **1️⃣ Install Dependencies**  
