import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_iris

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ML Dashboard", layout="wide")

# -------------------------------------------------
# CUSTOM CSS: GRADIENT BACKGROUND & TEXT COLORS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #E8E2DB, #1A3263, #547792);
    font-family: 'Arial', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #E8E2DB, #1A3263, #547792);
}
h1, h2, h3, h4, h5, h6, p, label, .stText {
    color: #ffffff !important;
    text-shadow: 1px 1px 2px #000000;
}
.stButton>button {
    background: linear-gradient(135deg, #1A3263, #547792);
    color: #ffffff;
    border-radius: 10px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #0f2245, #415f7a);
}
.stSelectbox, .stFileUploader, .stSlider {
    background-color: rgba(255,255,255,0.1);
    color: #ffffff;
}
.stMetricLabel, .stMetricValue {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1 style='text-align:center; color:#ffffff;'>🚀 Interactive ML App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#f0f0f0;'>Classification & Regression with Decision Tree, KNN, Ensemble, Naive Bayes</p>", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

task_type = st.sidebar.selectbox("Select Task Type", ["Classification", "Regression"])

# Add Naive Bayes only for Classification
if task_type == "Classification":
    model_type = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "KNN", "Ensemble", "Naive Bayes"])
else:
    model_type = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "KNN", "Ensemble"])

st.sidebar.subheader("Dataset Selection")
dataset_choice = st.sidebar.radio("Choose Dataset", ["Use Iris Dataset (built-in)", "Upload your CSV"])

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
if dataset_choice == "Use Iris Dataset (built-in)":
    data = load_iris(as_frame=True)
    df = data.frame
    target = data.target.name
else:
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is None:
        st.info("Please upload a CSV to continue")
        st.stop()
    df = pd.read_csv(file)
    target = st.selectbox("Select Target Column", df.columns)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

X = df.drop(columns=[target])
y = df[target]

# One-hot encode features
X = pd.get_dummies(X)

# Encode target if classification
if task_type == "Classification":
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Scale features for KNN
if model_type == "KNN":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------
if model_type == "Decision Tree":
    model = DecisionTreeClassifier() if task_type == "Classification" else DecisionTreeRegressor()
elif model_type == "KNN":
    k = st.sidebar.slider("K Neighbors", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k) if task_type == "Classification" else KNeighborsRegressor(n_neighbors=k)
elif model_type == "Ensemble":
    trees = st.sidebar.slider("Number of Trees", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=trees) if task_type == "Classification" else RandomForestRegressor(n_estimators=trees)
elif model_type == "Naive Bayes":
    model = GaussianNB()

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
with st.spinner("Training model..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

st.success("✅ Model trained successfully!")

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
st.subheader("📈 Model Results")

if task_type == "Classification":
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{round(acc*100, 2)} %")


    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

else:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error", round(mse, 4))
    st.metric("R² Score", round(r2, 4))
