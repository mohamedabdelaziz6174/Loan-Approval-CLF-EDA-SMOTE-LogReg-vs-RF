# 🏦 Loan Approval Classification | EDA, SMOTE & Model Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
In the banking sector, loan approval is a critical process that balances profitability with risk management. Relying solely on manual reviews is time-consuming and prone to human bias. 

This project builds an automated, end-to-end predictive pipeline to classify loan applications as **Approved** or **Rejected**. We dive deep into demographic and financial data, engineering new features, balancing the dataset, and comparing multiple machine learning models to find the most accurate predictor of credit risk.

## 🎯 Key Objectives
- **Exploratory Data Analysis (EDA):** Uncover patterns distinguishing approved vs. rejected applications.
- **Feature Engineering:** Create meaningful financial metrics such as *Total Assets* and *Loan-to-Income Ratios*.
- **Handling Imbalance:** Use **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model isn't biased toward the majority class.
- **Model Comparison:** Train and evaluate Logistic Regression, Decision Tree, and Random Forest models.

## 🛠️ Project Workflow

1. **Data Loading & Cleaning:** Stripped formatting issues, removed unnecessary IDs, and verified data integrity.
2. **EDA & Visualizations:** Analyzed target distribution, demographics, and crucial financial metrics.
3. **Feature Engineering:** Aggregated residential, commercial, luxury, and bank assets into a unified `total_assets` feature.
4. **Data Preprocessing:** Encoded categorical variables, scaled financial features using `StandardScaler`.
5. **Class Balancing:** Applied `SMOTE` to synthetically balance the training set.
6. **Modeling:** Trained three algorithms and compared their performance using Confusion Matrices and ROC-AUC curves.
7. **Insights Extraction:** Extracted Feature Importances from the Random Forest model to understand what truly drives approval.

## 📊 Key Findings
- **The Decider:** The **CIBIL Score** proved to be the single most important factor in determining loan approval, heavily outweighing total assets or annual income.
- **Algorithm Performance:** The **Random Forest** classifier outperformed both Logistic Regression and the Decision Tree, providing robust predictions and a higher AUC score.

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
