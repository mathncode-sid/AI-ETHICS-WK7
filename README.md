# 🔍 Dynamic COMPAS Fairness Audit (Streamlit App)

This project is an interactive **AI fairness audit tool** built using [Streamlit](https://streamlit.io/). It analyzes bias in the COMPAS dataset across protected attributes like **race**, **sex**, and **age category**, using fairness metrics like **False Positive Rate**, **Equal Opportunity Difference**, and **Disparate Impact**.

## 📁 Files

- `app.py` — The main Streamlit app script.
- `compas_clean.csv` — Preprocessed COMPAS dataset file (must be placed in the same directory).
- `README.md` — This documentation file.

## 🚀 How to Run

1. **Install requirements**

```bash
pip install streamlit pandas scikit-learn seaborn matplotlib
```

2. **Place the dataset**

Make sure `compas_clean.csv` is in the same folder as `app.py`.

3. **Start the app**

```bash
streamlit run app.py
```

---

## ⚙️ Features

- **Dynamic protected attribute** selector:
  - `race` (Caucasian vs African-American)
  - `sex` (Male vs Female)
  - `age_cat` (≥30 vs <30)

- **Model selection**:
  - Logistic Regression
  - Random Forest
  - Decision Tree

- **Fairness Metrics**:
  - False Positive Rate
  - Disparate Impact
  - Equal Opportunity Difference

- **Visuals**:
  - Group-wise FPR bar chart
  - Metric displays

- **Customizable test split size** via sidebar

---

## 📊 Fairness Metrics Explained

- **False Positive Rate (FPR)**: How often the model incorrectly predicts recidivism for someone who doesn’t reoffend.
- **Equal Opportunity Difference (EOD)**: Difference in true positive rates between groups.
- **Disparate Impact (DI)**: Ratio of positive outcomes between unprivileged and privileged groups. Values < 0.8 typically suggest bias.

---

## 📝 Ethical Use

This tool is designed for **educational and audit purposes**. Real-world deployment of predictive systems in sensitive areas (e.g., criminal justice) should always involve rigorous legal, ethical, and social review.

---

## 📌 Credits

Dataset Source: [ProPublica COMPAS Dataset](https://github.com/propublica/compas-analysis)

Developed for educational use as part of an AI Ethics assignment.
