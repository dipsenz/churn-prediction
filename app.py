import streamlit as st
import requests

API_URL ="http://127.0.0.1:8000/predict"
st.title("Churn Prediction Model")
st.markdown("Enter your details below: ")

# Categorical columns (Literals) mapped to selectboxes
gender = st.selectbox("Gender", options=['male', 'female'])
seniorcitizen = st.selectbox("Is your age above 60?", options=['0', '1'])
partner = st.selectbox("Marriage Status", options=['yes', 'no'])
dependents = st.selectbox("Dependents", options=['yes', 'no'])
phoneservice = st.selectbox("Phone Service", options=['yes', 'no'])
multiplelines = st.selectbox("Multiple Lines", options=['no', 'yes', 'no_phone_service'])
internetservice = st.selectbox("Internet Service", options=['no', 'dsl', 'fiber_optic'])
onlinesecurity = st.selectbox("Online Security", options=['no_internet_service', 'yes', 'no'])
onlinebackup = st.selectbox("Online Backup", options=['no_internet_service', 'yes', 'no'])
deviceprotection = st.selectbox("Device Protection", options=['no_internet_service', 'yes', 'no'])
techsupport = st.selectbox("Tech Support", options=['no_internet_service', 'yes', 'no'])
streamingtv = st.selectbox("Streaming TV", options=['no_internet_service', 'no', 'yes'])
streamingmovies = st.selectbox("Streaming Movies", options=['no_internet_service', 'yes', 'no'])
contract = st.selectbox("Contract", options=['two_year', 'one_year', 'month-to-month'])
paperlessbilling = st.selectbox("Paperless Billing", options=['no', 'yes'])
paymentmethod = st.selectbox("Payment Method", options=['mailed_check', 'credit_card_(automatic)', 'bank_transfer_(automatic)', 'electronic_check'])

# Numerical columns mapped to number_inputs (ge=0 handled by min_value=0)
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
totalcharges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

if st.button("Predict Churn"):
    input_data = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }
    try:
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extracting the values from the JSON response
            prob = result.get("Churn_Probability", 0.0)
            pred = result.get("Churn_Prediction", "Unknown")
            
            # Displaying the result as a formatted string
            if pred == 1:
                st.error(f"🔴 Customer is likely to churn | Churn probability: {float(prob):.2%}")
            else:
                st.success(f"🟢 Customer is unlikely to churn | Churn probability: {float(prob):.2%}" )

            st.write(f"Churn probability: {float(prob):.2%}")
            
        else:
            st.error(f"Error: Received status code {response.status_code} from API.")
            st.write(response.text)
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API. Details: {e}")
    