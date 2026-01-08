# 🇧🇷 Macro-Enhanced Credit Default Risk Model  
### An MLOps Learning Project

This repository documents an **end-to-end Machine Learning & MLOps learning project**, focused on building, evaluating, and deploying a **credit default risk model** while rigorously testing the hypothesis that **macroeconomic features improve predictive performance**.

The project prioritizes **methodological rigor, reproducibility, and production-oriented thinking**, even when experimental results do **not** confirm the initial hypothesis.

---

## 🎯 Project Goal

To evaluate whether augmenting a **client-level (micro) credit dataset** with **Brazilian macroeconomic indicators** improves the predictive performance of a loan default model.

**Initial hypothesis**:  
> The combination of micro-level borrower data and macroeconomic indicators (e.g., inflation, interest rates) improves AUC/ROC performance.

**Final conclusion**:  
> After systematic experimentation and feature selection, **macroeconomic variables did not improve the model’s predictive power**.  
> The final optimized model relies exclusively on **micro-level features**.

This outcome is explicitly documented as part of the learning process.

---

## 🧪 Experimental Outcome (Key Insight)

- Macroeconomic variables were engineered, lagged, and tested
- Multiple model configurations were evaluated
- Feature importance analysis and validation metrics showed **no statistically or practically meaningful performance gain**
- The final model contains **12 features**, **none of which are macroeconomic**

This reinforces an important real-world lesson:  
**Not all theoretically relevant features add predictive signal in practice**.

---

## 🚀 Repository Structure (MLOps-Oriented)

This project follows **MLOps best practices**, clearly separating experimentation, training artifacts, and serving logic:

project-root/n
│/n
├── data/ # Raw and processed datasets (Kaggle + BCB time series)/n
├── notebooks/ # Exploratory analysis and modeling notebooks (01 → 06)/n
├── src/ # Production-ready Python modules/n
│ ├── predict.py # Inference pipeline (PredictionHandler)/n
│ └── schemas.py # Pydantic input/output schemas/n
│/n
├── models/ # MLOps artifacts/n
│ ├── model.pkl/n
│ ├── target_encoder.pkl/n
│ └── imputation_map.json/n
│/n
├── Dockerfile # Reproducible serving environment/n
├── PLANNING.md # Execution plan and project phases/n
├── README.md # Project overview/n


---

## 📊 Modeling & Feature Engineering Highlights

To handle a high-dimensional credit dataset, the following techniques were applied:

### Feature Engineering
- Financial ratios such as:
  - Credit / Income
  - Annuity / Income
- Correction of known data anomalies (e.g., `DAYS_EMPLOYED` sentinel values)

### Categorical Encoding
- **Target Encoding** was used instead of One-Hot Encoding
- Reduced dimensionality dramatically
- Improved stability and performance of tree-based models

### Model Choice
- **LightGBM (`LGBMClassifier`)**
- Selected for:
  - Strong performance on tabular data
  - Fast training
  - Compatibility with production inference pipelines

---

## 🧠 Final Model Summary

- **Problem type**: Binary classification (default vs. non-default)
- **Metric focus**: ROC-AUC
- **Final feature count**: 12
- **Macroeconomic features used**: ❌ None
- **Reason**: No demonstrated predictive gain

This decision reflects **evidence-based feature selection**, not theoretical preference.

---

## 🌐 MLOps Deployment Pipeline (FastAPI & Docker)

The final model is deployed as a **production-style microservice**, emphasizing training–serving parity.

| Component | Technology | Role |
|--------|------------|------|
| API Layer | FastAPI | Exposes `/predict` endpoint with schema validation |
| Inference Logic | PredictionHandler | Applies preprocessing, encoding, and prediction consistently |
| Artifacts | joblib / JSON | Model, encoder, and imputation maps |
| Runtime | Docker | Reproducible, containerized serving environment |

---

## 🛠️ Development Environment

This project was developed using a **remote-first workflow**:

1. **VS Code Remote Development**
2. **Google Colab VM** as the main compute environment
3. **GitHub** for version control (PAT-based authentication)

This setup mirrors real-world constraints where training and serving often occur on different machines.

---

## 💡 Learning Outcomes

This project demonstrates:

- How to test and **reject a hypothesis responsibly**
- The importance of **training–serving parity**
- Proper management of **ML artifacts**
- Clean separation between experimentation and production code
- How MLOps adds value even when model performance plateaus

---

## 🔜 Next Steps (Deployment Track)

Current focus: **MLOps completion**, not model tuning.

- [x] Save final model and preprocessing artifacts
- [x] Implement PredictionHandler
- [ ] Finalize and test Dockerfile
- [ ] Run containerized API locally
- [ ] (Optional) Add CI/CD and monitoring

---

## 👤 Author

**Lucas Casarin**  
Economist | Machine Learning | MLOps-Oriented Analytics Engineering  

This repository is part of my professional portfolio and reflects **realistic ML system development**, including failed hypotheses, engineering trade-offs, and production concerns.


