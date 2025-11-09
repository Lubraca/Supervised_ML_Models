# 🗺️ Execution Plan: Macro-Enhanced Credit Default Risk Model

## Phase 1: Data Acquisition & Enrichment (Current Focus)

The goal of this phase is to consolidate the micro (Kaggle) and macro (Brazilian indicators) datasets and prepare a unified table for modeling.

### Step 1: Data Loading and Setup (Notebook: 01_Data_Load_EDA.ipynb)

1.  **Load Micro Data:** Download `application\_train.csv` (and auxiliary tables like `bureau.csv` for advanced features) using the Kaggle API.
2.  **Load Macro Data:** Mount Google Drive and load the pre-collected `dados\_macro\_brasil.csv` (Selic, IPCA, Unemployment) file.
3.  **Initial Cleaning:** Conduct immediate checks on data types and handle initial missing values in the primary Kaggle table.

### Step 2: Date Simulation and Merge Strategy

1.  **Date Simulation:** Create a new column, `SIMULATED\_DATE`, by randomly assigning a loan issuance month/year to each client record, covering the macro data's historical period (e.g., 2017-2022).
2.  **Date Alignment:** Ensure both micro and macro data are aligned by their respective month/year to serve as the merge key.
3.  **Enrichment Merge:** Perform a **Left Join** (merge) operation, linking each client record to the economic indicators that were active during their loan's simulated issuance date.

### Step 3: Exploratory Data Analysis (EDA) and Feature Engineering

1.  **EDA Focus:** Analyze the distribution of the target variable (`TARGET`), relationship between default and the new macro features, and overall feature correlation.
2.  **Initial Feature Engineering:** Create basic features (e.g., Ratios: Credit/Income) and prepare categorical features for encoding.
3.  **Commit Milestone:** Save the merged and processed file to `data/processed/` and commit this state.
