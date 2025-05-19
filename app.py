import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Load model and data
model = joblib.load("rf_model.pkl")
df = pd.read_csv("C:/Users/gomat/Downloads/bank-full.csv", sep=';')

# Encode target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Sidebar Navigation
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction","Model"])

# --- HOME PAGE ---
if page == "Home":
    st.title("🏦 Term Deposit Subscription Prediction")
    st.write("""
        This app helps banks predict whether a client will subscribe to a term deposit 
        based on their information from past marketing campaigns.
    """)



# --- EDA PAGE ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='y', ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution by Subscription")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='age', hue='y', kde=True, multiple='stack', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Job Type vs Subscription")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x='job', hue='y', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.title(" Predict Subscription")

    age = st.slider("Age", 18, 95, 30)
    balance = st.number_input("Balance (€)", step=100)
    duration = st.number_input("duration", step=10)
    campaign = st.number_input("Number of Contacts", step=1)
    pdays = st.number_input("Days Since Last Contact", step=1)
    previous = st.number_input("Previous Contacts", step=1)
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])
    default = st.selectbox("Credit in Default?", ["yes", "no"])

    # Encode inputs
    housing = 1 if housing == "yes" else 0
    loan = 1 if loan == "yes" else 0
    default = 1 if default == "yes" else 0

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'age': age,
            'balance': balance,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'housing': housing,
            'loan': loan,
            'default': default
        }])

        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("✅ The client is **likely to subscribe**.")
        else:
            st.warning("❌ The client is **unlikely to subscribe**.")

st.title("📘 Model Information")
st.markdown("""
    ### 🤖 Model Used: Random Forest Classifier

    The model used in this project is a **Random Forest Classifier**, which is an ensemble learning method based on decision trees. It works by constructing multiple decision trees during training and outputting the class that is the mode of the classes (for classification).

    **Why Random Forest?**
    - Handles both numerical and categorical features.
    - Robust to overfitting (compared to single decision trees).
    - Automatically estimates feature importance.
    - Works well on tabular banking data.

    **Features Used for Prediction:**
    - `age`
    - `balance`
    - `duration` (call duration in seconds)
    - `campaign` (number of contacts in current campaign)
    - `pdays` (days since last contact)
    - `previous` (number of previous contacts)
    - `housing` (has housing loan)
    - `loan` (has personal loan)
    - `default` (has credit in default)

    **Target Variable:**
    - `y`: whether the client subscribed to a term deposit (`yes`/`no`)

    **Model Training Tools:**
    - **Scikit-learn**
    - **Pandas** for data preprocessing
- **Joblib** for saving the trained model
- **Streamlit** for app deployment

---
This model helps bank staff predict which customers are most likely to subscribe to a term deposit, so they can focus their marketing efforts more effectively.""")

