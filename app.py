import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load saved model
model = joblib.load("credit_default_model.pkl")

# Create FastAPI app
app = FastAPI(title="Credit Default Prediction API")


# Define input format
class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float


@app.get("/")
def home():
    return {"message": "Credit Default Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": data.RevolvingUtilizationOfUnsecuredLines,
        "age": data.age,
        "NumberOfTime30-59DaysPastDueNotWorse": data.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": data.DebtRatio,
        "MonthlyIncome": data.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": data.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": data.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": data.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": data.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": data.NumberOfDependents
    }])

    # Predict class and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Map decision
    decision = "High Risk" if probability >= 0.5 else "Low Risk"

    return {
        "prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "decision": decision
    }