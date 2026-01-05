# 🇧🇷 Macro-Enhanced Credit Default Risk Model

## 🎯 Project Goal
To build a highly predictive credit risk scoring model for consumer loans by augmenting a client-level (micro) dataset with key Brazilian macroeconomic indicators (macro).

We aim to demonstrate that the combination of micro and macro features significantly improves the model's ability to predict default (measured by AUC/ROC).

***

## 🚀 Repository Structure (MLOps Focus)
This project adheres to **MLOps best practices**, structured to separate training, serving, and artifacts:

* `data/`: Stores raw and processed datasets (e.g., Kaggle files, BCB time series).
* `notebooks/`: Jupyter Notebooks for analysis, modeling, and results (notebooks 01 to 06).
* `src/`: **Production-ready Python modules** (`predict.py`, `schemas.py`) for the API.
* `models/`: **MLOps Artifacts** (`.pkl` files: model, encoder map, imputation map).
* `Dockerfile`: Defines the reproducible serving environment for the API.
* `PLANNING.md`: Detailed execution plan for project phases.
* `README.md`: This project overview.

***

## 📊 Modeling & Feature Engineering Highlights

To handle the high dimensionality and feature types of the dataset, the following techniques were crucial:

* **Feature Engineering:** Calculated key financial ratios ($\frac{\text{Credit}}{\text{Income}}$, $\frac{\text{Annuity}}{\text{Income}}$) and fixed critical data anomalies (e.g., `DAYS_EMPLOYED`).
* **Dimensionality Reduction:** Instead of high-cardinality One-Hot Encoding, **Target Encoding** was used on categorical features. This greatly improved model performance and reduced the feature count, resulting in $\approx 250+$ features.
* **Final Model:** **LightGBM** (`lgbm.LGBMClassifier`) was chosen for its speed and superior performance on tree-based problems, trained with optimized hyperparameters.

***

## 🌐 MLOps Deployment Pipeline (FastAPI & Docker)

The final optimized model is deployed as a resilient, containerized microservice. 

| Component | Technology | Role in Pipeline |
| :--- | :--- | :--- |
| **Prediction Service** | **FastAPI** | Exposes a low-latency `/predict` endpoint that accepts raw client data (Pydantic validation). |
| **Serving Environment** | **Docker** | Packages the entire application (`src/`, dependencies, and the `models/` artifacts) into a single, reproducible image. |
| **Prediction Logic** | **PredictionHandler** | Python class that loads the saved LightGBM model, the **Imputation Map**, and the **Target Encoder**. It applies all preprocessing steps consistently to live raw input data. |

***

## 🛠️ Environment Setup (VS Code + Colab VM)
This project is developed using a remote connection:
1.  **VS Code Remote Development** via the Microsoft Tunnel extension.
2.  **Google Colab VM** as the primary compute environment.
3.  **GitHub** for version control, secured via a Personal Access Token (PAT).

***

## 💡 Next Steps (Current Branch: Deployment)
The current focus is on **Finalizing MLOps Deployment**:
1.  Verify MLOps artifacts: Model, Imputation Map, and Target Encoder are saved. ✅
2.  Finalize the **PredictionHandler** class and implement the **FastAPI** service locally. ⏳
3.  Create and test the **Dockerfile** for containerization.
