from fastapi import FastAPI, Path, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field,computed_field
from typing import Annotated, Literal, Optional
import json
import pickle
import pandas as pd
import numpy as np

app = FastAPI() #obj

# 1. Load the model parameters on startup (fixed indentation here)
with open("churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    
w = model_data['w']
b = model_data['b']
mean = model_data['mean']
std = model_data['std']
expected_columns = model_data['columns']

# 2. Sigmoid mathematical function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class UserInput(BaseModel):
    gender: Annotated[Literal['male', 'female'],Field(...)]
    seniorcitizen: Annotated[Literal['0', '1'],Field(...)]
    partner: Annotated[Literal['yes', 'no'],Field(...)]
    dependents: Annotated[Literal['yes', 'no'],Field(...)]
    phoneservice: Annotated[Literal['yes', 'no'],Field(...)]
    multiplelines: Annotated[Literal['no', 'yes', 'no_phone_service'],Field(...)]
    internetservice: Annotated[Literal['no', 'dsl', 'fiber_optic'],Field(...)]
    onlinesecurity: Annotated[Literal['no_internet_service', 'yes', 'no'],Field(...)]
    onlinebackup: Annotated[Literal['no_internet_service', 'yes', 'no'],Field(...)]
    deviceprotection: Annotated[Literal['no_internet_service', 'yes', 'no'],Field(...)]
    techsupport: Annotated[Literal['no_internet_service', 'yes', 'no'],Field(...)]
    streamingtv: Annotated[Literal['no_internet_service', 'no', 'yes'],Field(...)]
    streamingmovies: Annotated[Literal['no_internet_service', 'yes', 'no'],Field(...)]
    contract: Annotated[Literal['two_year', 'one_year', 'month-to-month'],Field(...)]
    paperlessbilling: Annotated[Literal['no', 'yes'],Field(...)]
    paymentmethod: Annotated[Literal['mailed_check', 'credit_card_(automatic)','bank_transfer_(automatic)', 'electronic_check'],Field(...)]
    tenure: Annotated[int, Field(...,ge=0)]
    monthlycharges: Annotated[float, Field(...,ge=0)]
    totalcharges: Annotated[float, Field(...,ge=0)]

@app.post('/predict')
def predict_churn(data: UserInput):
    df=pd.DataFrame([data.model_dump()])
    df_dummies=pd.get_dummies(df,drop_first=True)
    df_encoded = df_dummies.reindex(columns=expected_columns, fill_value=0)
    x=df_encoded.to_numpy(dtype=np.float64)[0]
    x_scaled=(x-mean)/std
    z=np.dot(w,x_scaled)+b
    prob=sigmoid(z)
    pred=(int)(prob>=0.5)
    return JSONResponse(status_code=200, content={
        "Churn_Probability": float(prob),
        "Churn_Prediction": pred
    })
