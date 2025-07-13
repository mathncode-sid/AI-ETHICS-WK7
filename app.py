import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Dynamic COMPAS Fairness Audit", layout="wide")
st.title("🔍 Dynamic COMPAS Fairness Audit (No AIF360)")

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("compas_clean.csv")
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ]
    df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]
    df['age_cat'] = df['age'].apply(lambda x: "<30" if x < 30 else "30+")
    df['sex'] = df['sex'].fillna("Unknown")
    features = ['age', 'sex', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                'priors_count', 'c_charge_degree']
    df = df[features + ['two_year_recid', 'race', 'age_cat']]
    df = pd.get_dummies(df, columns=['sex', 'c_charge_degree'], drop_first=True)
    return df

df = load_and_clean_data()

# Sidebar: dynamic config
st.sidebar.header("⚙️ Audit Configuration")

protected_attr = st.sidebar.selectbox("Protected Attribute", ["race", "sex", "age_cat"], index=0)
model_choice = st.sidebar.selectbox("Classifier", ["Logistic Regression", "Random Forest", "Decision Tree"])
test_size_pct = st.sidebar.slider("Test set size (%)", 20, 50, 30)
show_fpr = st.sidebar.checkbox("Show False Positive Rate", True)
show_eod = st.sidebar.checkbox("Show Equal Opportunity Difference", True)
show_di = st.sidebar.checkbox("Show Disparate Impact", True)

# Encode protected attribute
if protected_attr == "race":
    df['protected'] = df['race'].apply(lambda r: 1 if r == 'Caucasian' else 0)
    priv_label, unpriv_label = "Caucasian", "African-American"
elif protected_attr == "sex":
    df['protected'] = df['sex_Male']
    priv_label, unpriv_label = "Male", "Female"
elif protected_attr == "age_cat":
    df['protected'] = df['age'].apply(lambda x: 1 if x >= 30 else 0)
    priv_label, unpriv_label = "30+", "<30"

X = df.drop(columns=['two_year_recid', 'race', 'age_cat', 'protected'])
y = df['two_year_recid']
groups = df['protected']

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, groups, test_size=test_size_pct / 100, random_state=42, stratify=groups
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mask_unpriv = g_test == 0
mask_priv = g_test == 1

def false_positive_rate(y_true, y_pred, mask):
    fp = ((y_true == 0) & (y_pred == 1) & mask).sum()
    actual_neg = ((y_true == 0) & mask).sum()
    return fp / actual_neg if actual_neg != 0 else 0

def equal_opportunity_diff(y_true, y_pred, mask_a, mask_b):
    tpr_a = ((y_true == 1) & (y_pred == 1) & mask_a).sum() / ((y_true == 1) & mask_a).sum()
    tpr_b = ((y_true == 1) & (y_pred == 1) & mask_b).sum() / ((y_true == 1) & mask_b).sum()
    return tpr_a - tpr_b

def disparate_impact(y_pred, mask_a, mask_b):
    rate_a = y_pred[mask_a].mean()
    rate_b = y_pred[mask_b].mean()
    return rate_a / rate_b if rate_b != 0 else 0

# Compute metrics
fpr_unpriv = false_positive_rate(y_test, y_pred, mask_unpriv)
fpr_priv = false_positive_rate(y_test, y_pred, mask_priv)
eod = equal_opportunity_diff(y_test, y_pred, mask_unpriv, mask_priv)
di = disparate_impact(y_pred, mask_unpriv, mask_priv)

# Display Metrics
st.subheader(f"📊 Fairness Metrics by {protected_attr.capitalize()}")
cols = st.columns(3)
if show_fpr:
    cols[0].metric(f"FPR ({unpriv_label})", f"{fpr_unpriv:.2f}")
    cols[0].metric(f"FPR ({priv_label})", f"{fpr_priv:.2f}")
if show_di:
    cols[1].metric("Disparate Impact", f"{di:.2f}")
if show_eod:
    cols[2].metric("Equal Opportunity Difference", f"{eod:.2f}")

# Plot
if show_fpr:
    fpr_data = pd.DataFrame({
        protected_attr.capitalize(): [unpriv_label, priv_label],
        "False Positive Rate": [fpr_unpriv, fpr_priv]
    })
    fig, ax = plt.subplots()
    sns.barplot(data=fpr_data, x=protected_attr.capitalize(), y="False Positive Rate", palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("False Positive Rate by Group")
    st.pyplot(fig)

st.subheader("📝 Summary")
st.markdown(f"""
You selected **{protected_attr}** as the protected attribute.  
- Privileged group: **{priv_label}**  
- Unprivileged group: **{unpriv_label}**  
- Classifier used: **{model_choice}**  
- Test size: **{test_size_pct}%**

This dynamic audit lets you explore how different groups are treated by predictive models, revealing disparities in outcomes.
""")
