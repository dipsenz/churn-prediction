# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.
The project includes a custom Logistic Regression model, a FastAPI backend, and an interactive Streamlit frontend.

## Project Overview
Customer churn prediction helps businesses identify customers who may stop using their services.
This project takes customer information such as tenure, contract type, internet service, monthly charges, and other service details and predicts the probability of customer churn.

## Architecture
Customer
   │
   ▼
Streamlit Frontend
   │
   │ HTTP POST Request
   ▼
FastAPI Backend
   │
   ▼
Saved ML Model
   │
   ▼
Churn Probability
   │
   ▼
Prediction

## Features
* Customer churn prediction
* Logistic Regression implemented using NumPy
* Feature standardization
* One-hot encoding of categorical features
* Model persistence using Pickle
* REST API using FastAPI
* Interactive frontend using Streamlit
* FastAPI Swagger documentation
* Probability-based churn prediction

## Model
The prediction model is a Logistic Regression model implemented manually using NumPy.
The prediction process is:

Input Data → One-Hot Encoding → Feature Standardization → Weighted Sum → Sigmoid Function → Churn Probability → Classification

The model calculates:
z = w · x + b
probability = sigmoid(z)

A probability of 0.5 or higher is classified as churn:
probability >= 0.5 → Churn
probability < 0.5  → No Churn

## Model Performance
The model was evaluated on a test dataset.

| Metric | Score |
| :--- | :--- |
| ROC-AUC | 0.8467 |
| Accuracy | 80.34% |
| Precision | 67.64% |
| Recall | 54.15% |
| F1 Score | 60.14% |

The ROC-AUC score of 0.8467 indicates that the model has good ability to distinguish between customers who churn and those who do not.

## Tech Stack
* Python
* NumPy
* Pandas
* Scikit-learn — evaluation and data-processing utilities
* Matplotlib — visualization
* FastAPI — REST API
* Uvicorn — API server
* Streamlit — frontend
* Pydantic — API input validation
* Pickle — model persistence
* Jupyter Notebook — model development

## Project Structure
ChurnPred/
│
├── app.py                    # Streamlit frontend
├── main.py                   # FastAPI backend
├── churn_model.pkl           # Saved trained model
├── churn.csv                 # Dataset
├── churn_prediction.ipynb    # Model development and evaluation
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Files ignored by Git
└── README.md                 # Project documentation

## Installation

1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ChurnPred

2. Create a virtual environment (Windows)
python -m venv myenv

3. Activate the virtual environment (PowerShell)
.\myenv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

## Running the Project
The project consists of two applications:
1. FastAPI backend
2. Streamlit frontend

Both need to be running.

### Start FastAPI
Open a terminal in the project directory:
uvicorn main:app --reload

The API will run at:
http://127.0.0.1:8000

FastAPI provides interactive API documentation at:
http://127.0.0.1:8000/docs

### Start Streamlit
Open another terminal in the project directory and activate the environment.
Then run:
streamlit run app.py

The Streamlit application will open in your browser.

## API
Endpoint: POST /predict

The API accepts customer information and returns the predicted churn probability and classification.

Example Response:
{
    "Churn_Probability": 0.4244,
    "Churn_Prediction": 0
}

Where:
Churn_Prediction = 0 → Customer is unlikely to churn
Churn_Prediction = 1 → Customer is likely to churn

## Streamlit Interface
The Streamlit application allows users to enter customer information through an interactive interface.
The application sends the information to the FastAPI backend, which processes the input using the saved model and returns the prediction.

Example:
🟢 Customer is unlikely to churn
Churn probability: 42.44%

or:

🔴 Customer is likely to churn
Churn probability: 73.21%

## Future Improvements
* Tune the classification threshold to improve churn recall
* Improve frontend design and visualization
* Add probability visualizations
* Deploy the application online
* Add automated testing
* Add Docker support
* Add model versioning
* Monitor model performance after deployment

---
Author
Soumyadip Sen

This project was developed as an end-to-end machine learning project combining model development, model persistence, REST API development, and an interactive frontend.