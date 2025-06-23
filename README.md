# 🧬 Multi-Label Mutation Tag Prediction Using Machine Learning

This repository contains a complete pipeline for predicting multiple mutation tags from high-dimensional biological data. The project includes data preprocessing, exploratory data analysis (EDA), dimensionality reduction, and multi-label classification using models like Logistic Regression, Random Forest, LightGBM, and XGBoost.

---

## 📂 Project Structure

```
.
🔘 data/
│   🔘 raw/                 # Contains original K9.data and K9.instance.tags
│   └🔘 processed/           # Cleaned CSVs, encoded labels, and tag classes
🔘 artifacts/               # Dimensionality reduced features and labels
🔘 saved_model/             # Trained XGBoost model (.pkl)
🔘 notebooks/               # Jupyter/Colab notebooks for analysis and training
🔘 README.md                # This file
🔘 requirements.txt         # All dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MasterBiswal/USD-Term1-AppliedAI-GroupSynergy
cd mutation-tag-prediction
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

These are the key Python libraries used in this project:

```text
pandas
numpy
joblib
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

All are listed in `requirements.txt`. You can generate or update it using:

```bash
pip freeze > requirements.txt
```

---

## 📊 Project Workflow

### 🔹 1. Data Cleaning

- Parse raw `.tags` and `.data` files
- Align tags and features
- Handle missing values and dimensional mismatches
- Encode multi-label tags with `MultiLabelBinarizer`

### 🔹 2. Exploratory Data Analysis (EDA)

- Analyze feature sparsity and tag distributions
- Visualize tag frequency
- Correlation analysis and dimensionality reduction via Truncated SVD

### 🔹 3. Modeling

- Baseline: Logistic Regression (OvR)
- Random Forest (OvR)
- Final: XGBoost (OvR) – Best performance with F1-micro ≈ 0.995

### 🔹 4. Model Evaluation

- Classification reports
- Micro and macro F1 scores
- Hamming loss and subset accuracy

### 🔹 5. Model Export

- Trained model saved as `xgb_multilabel_model.pkl` using `joblib`

---

## 📈 Performance Summary (XGBoost)

| Metric            | Score |
| ----------------- | ----- |
| Micro F1 Score    | 0.995 |
| Macro F1 Score    | 0.985 |
| Weighted F1 Score | 1.000 |
| Samples Avg. F1   | 0.990 |

---

## 🧪 Reproducibility

Ensure your directory structure matches the expected format:

```text
project_root/
🔘 data/
│   🔘 raw/ (place your original K9.data and K9.instance.tags here)
│   └🔘 processed/
🔘 artifacts/
🔘 saved_model/
```

Run all scripts or notebooks in order, or load up the Jupyter notebook version to experiment interactively.

---

## 🧰 Future Improvements

- Add SHAP or LIME interpretability
- Handle rare tags using meta-labeling or oversampling (e.g., SMOTE)
- Build REST API or Streamlit interface
- Evaluate on external datasets

---

## 📄 License

This project is under the MIT License. See the LICENSE file for details.

---

## 🙌 Acknowledgments

- scikit-learn, LightGBM, and XGBoost teams for powerful ML libraries
- INRIA and OpenML for benchmarking techniques
- Your team, collaborators, and professors for feedback and guidance

---

## 🤝 Contributions

Feel free to fork, raise issues, or submit pull requests to improve this project!

