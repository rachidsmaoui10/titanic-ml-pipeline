# Titanic ML Pipeline

End-to-end machine learning pipeline for predicting Titanic passenger survival.

This project demonstrates a complete supervised learning workflow:
data cleaning, exploratory data analysis, feature engineering, model training,
evaluation, and threshold optimization.

---

## Project Structure

titanic-ml-pipeline/
├── data/                     # Raw dataset
├── notebooks/                # EDA and experiments
│   └── eda_titanic.ipynb
├── src/                      # ML pipeline modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── outputs/                  # Predictions and reports
├── requirements.txt
└── README.md


---

## Workflow

1. **Data Cleaning**
   - Handle missing values (Age, Embarked)
   - Drop irrelevant features
   - Convert data types

2. **Exploratory Data Analysis**
   - Distribution of Age
   - Survival rates by Sex and Pclass
   - Survival probability analysis

3. **Feature Engineering**
   - IsChild
   - IsMother
   - IsAlone
   - One-hot encoding of categorical variables

4. **Modeling**
   - Logistic Regression
   - Decision Tree

5. **Evaluation**
   - Accuracy
   - Confusion matrix
   - Classification report
   - Threshold tuning for recall optimization

---

## Example Results

| Model | Train Accuracy | Test Accuracy |
|------|---------------|--------------|
| Logistic Regression | ~0.80 | ~0.79 |
| Decision Tree | ~0.84 | ~0.77 |

---

## Technologies Used

- Python
- Pandas / NumPy
- Seaborn / Matplotlib
- Scikit-learn

---

## Author

Rachid Smaoui

