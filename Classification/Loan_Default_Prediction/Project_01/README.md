# 🇧🇷 Macro-Enhanced Credit Default Risk Model

## 🎯 Project Goal
To build a highly predictive credit risk scoring model for consumer loans by augmenting a client-level (micro) dataset with key Brazilian macroeconomic indicators (macro).

We aim to demonstrate that the combination of micro and macro features significantly improves the model's ability to predict default (measured by AUC/ROC).

## 🚀 Repository Structure
This project is structured logically to separate data, code, and documentation:

* `data/`: Stores raw and processed datasets (e.g., Kaggle files, BCB time series).
* `notebooks/`: Jupyter Notebooks for analysis, modeling, and results.
* `src/`: Production-ready Python modules and utility scripts.
* `PLANNING.md`: Detailed execution plan for project phases.
* `README.md`: This project overview.

## �� Data Sources
| Data Type | Source | Purpose |
| :--- | :--- | :--- |
| **Micro (Client Level)** | Kaggle: Home Credit Default Risk | Client demographics and loan history (the core predictive features). |
| **Macro (Brazilian Context)** | BCB/SGS & Ipeadata | Economic indicators (SELIC, IPCA, Unemployment Rate) for risk adjustment. |

## 🛠️ Environment Setup (VS Code + Colab VM)
This project is developed using a remote connection:
1.  **VS Code Remote Development** via the Microsoft Tunnel extension.
2.  **Google Colab VM** as the primary compute environment.
3.  **GitHub** for version control, secured via a Personal Access Token (PAT).

## 💡 Next Steps (Current Branch: Feature_Load_EDA)
The current focus is on **Data Acquisition and Enrichment**:
1.  Load micro data (Kaggle) and macro data (BCB/Ipeadata).
2.  Implement the **simulated date strategy** to link macro data to micro records.
3.  Perform initial Data Cleaning and Exploratory Data Analysis (EDA).
