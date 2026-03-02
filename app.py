import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import base64

st.set_page_config(page_title="FraudHunter Pro", page_icon="🛡️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FraudHunter Pro: Advanced Anomaly Detection System")
st.markdown("Predict fraudulent transactions and analyze customer patterns with explainable AI (XAI).")

menu = ["Home", "Data Upload & Training", "Real-time Detection", "Customer Personalization", "Model Explainability (SHAP)"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.markdown("""
    <div class='card'>
    <h2>Welcome to FraudHunter Pro</h2>
    <p>This application is an enterprise-grade solution that utilizes robust machine learning algorithms, primarily Anomaly Detection and Association Rule Mining, to identify potential fraud and discover customer behavior patterns.</p>
    <ul>
        <li><b>Data Upload & Training</b>: Upload credit card or bank transaction datasets, clean the data dynamically, and train an Isolation Forest model seamlessly.</li>
        <li><b>Real-time Detection</b>: Input distinct transaction details to get an instant prediction on whether it exhibits characteristics of fraud.</li>
        <li><b>Customer Personalization</b>: Use FP-Growth to identify frequent itemsets and recommend specific features or products to certain customer segments.</li>
        <li><b>Model Explainability (SHAP)</b>: Move beyond the 'black-box' nature of ML models to see <i>exactly</i> why a specific transaction was flagged as fraudulent using SHAP values.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif choice == "Data Upload & Training":
    st.header("Upload & Train")
    data_file = st.file_uploader("Upload CSV Data (Credit Card / Bank)", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.write("### Data Preview", df.head())
        
        # Determine numeric columns for quick isolation forest training
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = st.multiselect("Select features for training", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
        
        if st.button("Train Anomaly Detection Model"):
            with st.spinner("Training Isolation Forest..."):
                X = df[feature_cols].fillna(0)
                model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                model.fit(X)
                
                # Save model and features list
                joblib.dump({"model": model, "features": feature_cols}, "models/isolation_forest.pkl")
                
                st.success("Model trained successfully and saved to 'models/isolation_forest.pkl'!")
                
                # Show predictions on the uploaded data
                preds = model.predict(X)
                # Isolation Forest predicts -1 for anomalies, 1 for normal
                df['Anomaly'] = preds
                df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Fraud'})
                
                st.write("### Detection Statistics")
                st.write(df['Anomaly'].value_counts())
                
elif choice == "Real-time Detection":
    st.header("Real-time Fraud Detection")
    try:
        saved_data = joblib.load("models/isolation_forest.pkl")
        model = saved_data["model"]
        features = saved_data["features"]
        
        st.write("Please enter the values for the transaction:")
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(features):
            input_data[feature] = cols[i % 3].number_input(f"{feature}", value=0.0)
            
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            if pred == -1:
                st.error("🚨 ALERT: This transaction is predicted as FRAUDULENT 🚨")
            else:
                st.success("✅ This transaction appears to be NORMAL.")
    except FileNotFoundError:
        st.warning("No trained model found. Please go to 'Data Upload & Training' to train a model first.")

elif choice == "Model Explainability (SHAP)":
    st.header("Explainable AI: Why was it flagged?")
    try:
        saved_data = joblib.load("models/isolation_forest.pkl")
        model = saved_data["model"]
        features = saved_data["features"]
        
        st.write("Generating SHAP explanations requires a sample of your training data. Please upload your original CSV again:")
        data_file = st.file_uploader("Upload CSV Data", type=["csv"], key="shap")
        if data_file is not None:
            df = pd.read_csv(data_file)
            X = df[features].fillna(0)
            
            st.write("Select a transaction index to explain:")
            idx = st.number_input("Transaction Row Index", min_value=0, max_value=len(X)-1, value=0)
            
            if st.button("Generate SHAP Explanation"):
                with st.spinner("Calculating SHAP values..."):
                    explainer = shap.Explainer(model.predict, X.iloc[:100]) # use a background dataset
                    shap_values = explainer(X.iloc[idx:idx+1])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(shap_values[0], show=False)
                    st.pyplot(fig)
                    
                    st.info("The waterfall plot above shows how each feature contributed to the final prediction. Red bars push the prediction score higher, while blue bars push it lower. For Anomaly Detection, extreme feature values typically push the score toward an anomaly.")
    except FileNotFoundError:
        st.warning("No trained model found. Please go to 'Data Upload & Training' to train a model first.")

elif choice == "Customer Personalization":
    st.header("Customer Personalization & Associated Patterns")
    st.markdown("If you upload the Bank Transaction dataset, we can identify frequent itemsets (e.g., specific transaction devices + transaction types + age bins) to understand customer behavior.")
    
    data_file = st.file_uploader("Upload Bank Transaction Data (CSV)", type=["csv"], key="assoc")
    if data_file is not None:
        df = pd.read_csv(data_file)
        # Assuming we just grab categorical variables for association
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write("Categorical Columns found:", cat_cols)
        
        if st.button("Run FP-Growth"):
            with st.spinner("Extracting frequent patterns..."):
                from mlxtend.frequent_patterns import fpgrowth, association_rules
                basket = pd.get_dummies(df[cat_cols].astype(str))
                frequent_itemsets = fpgrowth(basket, min_support=0.1, use_colnames=True)
                
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    st.write("### Top Association Rules")
                    st.dataframe(rules.sort_values(by="lift", ascending=False).head(10))
                else:
                    st.warning("No frequent itemsets found with current support threshold.")
