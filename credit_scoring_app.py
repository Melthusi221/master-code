import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Dissertation\Master\Credit-Scoring-master\dataset\estadistical.csv")
    x = dataset.drop("Receive/ Not receive credit ", axis=1)
    y = dataset["Receive/ Not receive credit "]
    cat_cols = x.columns[x.dtypes == object].tolist()
    le = LabelEncoder()
    x[cat_cols] = x[cat_cols].apply(lambda col: le.fit_transform(col))
    return x, y

@st.cache_resource
def train_models():
    x, y = load_data()
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y)
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=10).fit(xtrain_scaled, ytrain),
        "Random Forest": RandomForestClassifier(max_depth=2).fit(xtrain_scaled, ytrain),
        "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000).fit(xtrain_scaled, ytrain),
        "SVM": SVC(probability=True).fit(xtrain_scaled, ytrain)
    }
    return models, scaler

def main():
    st.title("🎯 MFI Credit Scoring Dashboard")
    models, scaler = train_models()
    
    # Dynamic input fields based on your dataset
    x, _ = load_data()
    user_inputs = {col: st.sidebar.number_input(col) for col in x.columns}
    
    if st.sidebar.button("🔮 Predict"):
        input_df = pd.DataFrame([user_inputs])
        input_scaled = scaler.transform(input_df)
        results = {
            name: ("✅ APPROVE" if model.predict(input_scaled)[0] == 1 else "❌ REJECT", 
            f"{max(model.predict_proba(input_scaled)[0])*100:.1f}%"
            ) for name, model in models.items()
        }
        
        st.subheader("📊 Results")
        for model_name, (pred, conf) in results.items():
            st.write(f"**{model_name}**: {pred} (Confidence: {conf})")

if __name__ == "__main__":
    main()