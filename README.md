# 📊 Predicting Term Deposit Subscription – Streamlit ML App

A machine learning web application built using **Streamlit** to predict whether a client will subscribe to a **term deposit** based on their personal and marketing data.

---

## 🎯 Problem Statement

A Portuguese bank conducted several direct marketing campaigns via phone calls. The goal is to improve the efficiency of these campaigns by predicting which clients are likely to subscribe to a term deposit, using historical data.

---

## 💼 Business Use Case

- Improve **targeted marketing** by identifying high-potential customers.
- **Reduce telemarketing costs** by avoiding unlikely customers.
- Enhance **customer experience** with relevant communication.
- Increase **campaign performance** using data-driven insights.

---

## 🤖 Model Used

- **Random Forest Classifier**
  - Handles both numerical and categorical data
  - Resistant to overfitting
  - Provides feature importance
  - Works well on tabular datasets like this

---

## 📦 Features Used for Prediction

- `age` — client age
- `balance` — average yearly account balance
- `duration` — last contact duration (in seconds)
- `campaign` — number of contacts during current campaign
- `pdays` — days since client was last contacted
- `previous` — number of contacts in previous campaigns
- `housing` — has a housing loan (yes/no)
- `loan` — has a personal loan (yes/no)
- `default` — has credit in default (yes/no)

---

## 📁 Dataset

- File: `bank-full.csv`
- Source: UCI Bank Marketing Dataset
- Size: 45,000+ rows
- Target variable: `y` – whether client subscribed to a term deposit (`yes` or `no`)

---

## 🧠 App Features

Built using **Streamlit** with multiple pages:
- 🏠 **Home**: Project intro and use case
- 📊 **EDA**: Exploratory Data Analysis (bar charts, histograms, job-wise analysis)
- 🔮 **Prediction**: Input form to predict client subscription
- 📘 **Model Info**: Explanation of the model used

---

## 🚀 How to Run the App Locally

1. **Clone this repository**
```bash
git clone https://github.com/yourusername/term-deposit-predictor.git
cd term-deposit-predictor