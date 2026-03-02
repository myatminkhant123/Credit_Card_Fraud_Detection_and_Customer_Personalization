# 🛡️ FraudHunter Pro: Credit Card Fraud Detection & Customer Personalization

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
FraudHunter Pro is an end-to-end data mining and machine learning application designed to identify fraudulent credit card transactions and provide tailored customer recommendations for safer and smarter digital payments. This project offers a comprehensive suite of powerful tools designed to catch anomalies in real-time, explain ML predictions using **Explainable AI (SHAP)**, and discover frequent customer patterns through **Association Rule Mining (FP-Growth)**.

### ✨ Key Features
- **Dynamic Model Training**: Upload your own transaction data to instantly train an Isolation Forest anomaly detection model.
- **Real-Time Fraud Prevention**: Enter transaction parameters manually to detect unseen fraudulent activity instantly.
- **Explainable AI (XAI)**: Demystify the "black box" using **SHAP (SHapley Additive exPlanations)** waterfall plots to understand exactly *why* a transaction was flagged.
- **Customer Personalization**: Extract meaningful shopping habits and demographic associations using the APRIORI/FP-Growth algorithms to suggest tailored banking options.

## 🚀 Live Application
Access the deployed Streamlit Web App here: **[Link to Streamlit App (Update this once deployed)](#)**

## 📂 Repository Structure
```
📁 Credit_Card_Fraud_Detection_and_Customer_Personalization/
├── 📁 notebooks/
│   ├── Anomaly Detection.ipynb (Exploratory Data Analysis & Modeling)
│   ├── Fraud_Detection Project.ipynb (Association Rules for Bank Transactions)
│   ├── DBSCAN_project..ipynb (Clustering analysis)
│   └── Data_Mining_Group_Project_(Recomender_System).ipynb
├── 📁 models/
│   └── isolation_forest.pkl (Trained weights & features)
├── 📁 data/
│   └── (Ignored via .gitignore) Put your raw CSV files here.
├── app.py (Main Streamlit application file)
├── requirements.txt
└── README.md
```

## 🧠 Machine Learning Algorithms
1. **Anomaly Detection**: 
   - *Isolation Forest*, *Local Outlier Factor (LOF)*, and *One-Class SVM*. 
   - Proved highly effective at separating the sparse outliers (fraudulent transactions) from dense, normal behavioral clusters without requiring balanced training data.
2. **Association Rule Mining**: 
   - *FP-Growth* is applied on grouped banking transactions to draw strong associations between Customer demographics, Transaction devices (ATMs, Kiosks), Transaction Types, and Account balances.
3. **Explainability**:
   - *SHAP* provides feature importance, giving critical context to banking analysts tracking down the root cause of fraud alerts.

## 💻 Local Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/myatminkhant123/Credit_Card_Fraud_Detection_and_Customer_Personalization.git
   cd Credit_Card_Fraud_Detection_and_Customer_Personalization
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Performance Metrics
For specific model evaluation reports, referencing Notebooks (`Anomaly Detection.ipynb`), Precision, Recall, and highly specialized Area Under PR Curve metrics are thoroughly documented to address severe class imbalance challenges.

## 🤝 Let's Connect
Feel free to drop a star ⭐ on this repository, fork the project to perform your own magic, or connect with me if you have any questions!
