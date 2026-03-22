
# 📊 Customer Churn Prediction

## 🚀 Project Overview

This project aims to predict whether a customer is likely to churn (leave the service) using machine learning techniques.
Customer churn prediction helps businesses take proactive actions to retain customers and reduce revenue loss.

---

## 🎯 Problem Statement

Customer churn is a major problem in industries like telecom, banking, and subscription services.
The goal of this project is to build a machine learning model that predicts:

👉 **Will a customer churn or not?**

---

## 📊 Dataset

* Dataset: Telco Customer Churn Dataset
* Source: Kaggle
* Records: 7000+ customers
* Features: 20+ attributes (demographics, billing, services)

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:**

  * Pandas, NumPy
  * Scikit-learn
  * Imbalanced-learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing

* Handled missing values
* Converted categorical features to numerical
* Feature scaling and encoding

### 2️⃣ Exploratory Data Analysis (EDA)

* Analyzed churn distribution
* Identified patterns in customer behavior
* Used visualizations (countplot, heatmap)

### 3️⃣ Feature Engineering

* One-hot encoding
* Converted target variable (Yes/No → 1/0)

### 4️⃣ Handling Imbalanced Data

* Used **SMOTE** to balance churn vs non-churn classes

### 5️⃣ Model Training

Models used:

* Logistic Regression
* Random Forest ✅ (Final Model)
* Gradient Boosting

---

## 📈 Model Performance

| Metric         | Score |
| -------------- | ----- |
| Accuracy       | 76%   |
| ROC-AUC        | 0.80  |
| Recall (Churn) | ~43%  |

---

## 🔍 Key Business Insights

* Customers with **month-to-month contracts** churn more
* Lack of **tech support** increases churn probability
* **High monthly charges** lead to higher churn
* **New customers** are more likely to leave

---

## 🖥️ Streamlit App

An interactive web application is built using Streamlit where users can:

* Enter customer details
* Predict churn probability
* Identify high-risk customers

---

## ▶️ Run the Project Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/hitmanp007/customer-churn-prediction.git
cd customer_churn_project
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
customer_churn_project/
│
├── app/
│   └── streamlit_app.py
├── data/
├── model/
│   ├── churn_model.pkl
│   └── model_columns.pkl
├── notebook/
├── requirements.txt
└── README.md
```

---

## 🏆 Key Learnings

* End-to-end machine learning pipeline
* Handling real-world data issues
* Model evaluation beyond accuracy
* Avoiding data leakage
* Building and deploying ML apps

---

## 🚀 Future Improvements

* Deploy app on Streamlit Cloud
* Add probability gauge visualization
* Use advanced models (XGBoost)
* Build API using FastAPI

---

## 👨‍💻 Author

Pranav Sahu

Muskan Sahu

---

# customer-churn-prediction
