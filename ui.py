import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_default_model.pkl")

def predict_default(
    revolving_utilization,
    age,
    past_due_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    past_due_90,
    real_estate_loans,
    past_due_60_89,
    dependents
):
    # Create input DataFrame with same feature names used in training
    input_data = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberOfTimes90DaysLate": past_due_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
    "NumberOfDependents": dependents
}])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_level = "High Risk" if probability >= 0.5 else "Low Risk"

    return int(prediction), round(probability, 4), risk_level


demo = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Revolving Utilization of Unsecured Lines"),
        gr.Number(label="Age"),
        gr.Number(label="Number of Times 30-59 Days Past Due"),
        gr.Number(label="Debt Ratio"),
        gr.Number(label="Monthly Income"),
        gr.Number(label="Number of Open Credit Lines and Loans"),
        gr.Number(label="Number of Times 90 Days Late"),
        gr.Number(label="Number of Real Estate Loans or Lines"),
        gr.Number(label="Number of Times 60-89 Days Past Due"),
        gr.Number(label="Number of Dependents"),
    ],
    outputs=[
        gr.Number(label="Prediction: 0 = No Default, 1 = Default"),
        gr.Number(label="Default Probability"),
        gr.Textbox(label="Risk Level")
    ],
    title="Credit Default Prediction",
    description="Enter customer financial information to predict default risk."
)

demo.launch()