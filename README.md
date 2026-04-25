# 💳 Credit Default Prediction (Give Me Some Credit)

## 📌 Overview
This project uses machine learning to predict whether a customer will experience serious financial delinquency within the next two years.

The goal is to help financial institutions identify high-risk customers and support better credit decision-making.

---

## 🎯 Problem Statement
Banks need to assess the risk of lending money to customers.

Traditional approaches rely on:
- Manual credit review  
- Simple rules (e.g., income thresholds, debt limits)  

These methods are:
- Time-consuming  
- Not scalable  
- Unable to capture complex patterns  

This project builds a machine learning model to improve risk prediction.

---

## 🤖 Machine Learning Task
- Type: Binary Classification
- Target: SeriousDlqin2yrs  
  - 0 → No Default  
  - 1 → Default  

---

## 📊 Dataset
Source: Kaggle — Give Me Some Credit

- ~150,000 customer records  
- Financial and behavioral features  
- Missing values handled using imputation  
- Predefined train/test split (test set has no labels)

---

## 📈 Data Assessment (ART)

Availability:  
All features are available at prediction time.

Representativeness:  
The dataset reflects real-world financial behavior.

Trust:  
Missing values were handled carefully using preprocessing.

---

## ⚙️ Feature Engineering & Preprocessing
- Missing values handled using median imputation
- Feature scaling using StandardScaler
- Implemented using Pipeline to prevent data leakage

---

## 🧠 Models Used
- Logistic Regression (baseline)
- Random Forest (final model)

---

## 📊 Evaluation

We used 5-Fold Cross-Validation since test labels are unavailable.

Metric:
- AUC (Area Under Curve)

Results:

| Model | Mean CV AUC |
|------|------------|
| Logistic Regression | 0.791 |
| Random Forest | 0.835 |

Random Forest selected as final model.

---

## 🔍 Key Features
- NumberOfTimes90DaysLate  
- RevolvingUtilizationOfUnsecuredLines  
- DebtRatio  

---

## 🎯 Prediction & Decision Mapping

Model Output:  
Probability (0–1) of default

Decision Rule:
- > 0.5 → High Risk  
- ≤ 0.5 → Low Risk  

---

## ⚙️ Setup and Usage

### 1. Clone the Repository
git clone https://github.com/your-username/Give-Me-Some-Credit-ML-Project  
cd Give-Me-Some-Credit-ML-Project

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train and Save the Model
Run the notebook and execute all cells to generate:
credit_default_model.pkl

### 4. Use the Model in Python
import joblib  
import pandas as pd  

model = joblib.load("credit_default_model.pkl")  

sample = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.8,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.5,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 1,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
}])

prediction = model.predict(sample)  
probability = model.predict_proba(sample)[0][1]

### 5. Run the UI
python ui.py  
http://127.0.0.1:7860

### 6. Run the API
uvicorn app:app --reload  
http://127.0.0.1:8000/docs

---

## 💾 Outputs
- credit_default_model.pkl  
- credit_default_predictions.csv  

---

## 📊 Success & Failure Criteria

Success:
- AUC ≥ 0.80  
- Correct identification of high-risk customers  

Failure:
- Poor performance  
- High false positives  
- Poor generalization  

---

## 📌 Conclusion
Random Forest achieved the best performance and supports better credit risk decision-making.
