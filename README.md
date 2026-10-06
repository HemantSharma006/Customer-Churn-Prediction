# Customer Churn Prediction System

A machine learning-based web application that predicts whether a customer is likely to churn and provides an estimated churn probability, risk level, key prediction factors, and recommended retention action.

This project was developed as part of an Artificial Intelligence internship and demonstrates an end-to-end workflow from data preprocessing and model training to web-based customer churn prediction.

---

## 📌 Project Overview

Customer churn refers to the situation where a customer stops using a company's products or services.

Predicting customer churn in advance can help businesses identify high-risk customers and take appropriate retention actions.

This project uses machine learning to analyze customer information and estimate the probability of churn.

The trained model is integrated into a Flask web application where users can enter customer details and receive an instant prediction.

---

## 🎯 Objectives

- Predict whether a customer is likely to churn.
- Calculate the estimated probability of customer churn.
- Classify customers into Low, Medium, or High risk.
- Identify important customer attributes associated with the prediction.
- Provide an AI-based recommendation for customer retention.
- Compare different machine learning classification models.
- Integrate the trained model into a web application.

---

## ✨ Features

### Machine Learning

- Data cleaning and preprocessing
- Numerical feature scaling
- Categorical feature encoding
- Stratified train-test split
- Logistic Regression
- Random Forest Classifier
- Model performance comparison
- Automatic selection of the model with the better F1-score
- Model serialization using Joblib

### Web Application

- Modern dark-themed user interface
- Customer information input form
- Churn probability prediction
- Low, Medium, and High risk classification
- Key prediction factors
- AI-based retention recommendation
- Responsive design
- Local network access from other devices

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Flask | Backend web framework |
| HTML | Web page structure |
| CSS | User interface styling |
| JavaScript | Frontend interactions |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |

---

## 📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information including:

- Gender
- Senior Citizen status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Technical Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

The `customerID` column is removed during preprocessing because it is an identifier and does not provide useful predictive information.

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Convert Total Charges

The `TotalCharges` column is converted from text to numeric values.

Invalid values are converted to missing values and the corresponding rows are removed.

### 2. Remove Customer ID

The `customerID` column is removed because it is only an identifier.

### 3. Convert Target Variable

The `Churn` column is converted into binary values:

```text
No  → 0
Yes → 1