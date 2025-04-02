MLOps_Weather_in_Australia/
│
├── .github/workflows/        # (CI/CD with GitHub Actions if needed)
├── data/                     # (Raw & processed datasets)
│   ├── raw/
│   ├── processed/
│
├── logs/                     # (Logging outputs)
├── metrics/                  # (Evaluation results)
├── notebooks/
│   ├── workflow_steps.ipynb  # (Your guide notebook)
│
├── src/                      # (Main source code)
│   ├── common_utils.py       # (Helper functions like reading YAML, creating directories)
│   ├── config.py             # (Global paths for YAML files)
│   ├── config_manager.py     # (Manages configurations)
│   ├── config.yaml           # (File paths and dataset details)
│   ├── params.yaml           # (Model hyperparameters)
│   ├── entity.py             # (Configuration dataclasses)
│
│   ├── app/                  # (Model API)
│   │   ├── app.py
│
│   ├── data_module_def/      # (Data processing)
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── schema.yaml       # (Dataset structure)
│
│   ├── models_module_def/    # (Model training & evaluation)
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── params.yaml
│
│   ├── pipeline_steps/       # (Stage-wise pipeline scripts)
│   │   ├── stage01_data_ingestion.py
│   │   ├── stage02_data_validation.py
│   │   ├── stage03_data_transformation.py
│   │   ├── stage04_model_trainer.py
│   │   ├── stage05_model_evaluation.py
│
├── templates/                # (Optional: for web UI if needed)
│
├── .gitignore
├── .dvcignore
├── dvc.yaml
├── dvc.lock
├── README.md
├── requirements.txt
└── __init__.py
