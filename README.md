# Credit Risk Modeling 🚀
## This project focuses on predicting credit risk using machine learning techniques. The goal is to classify borrowers into different risk levels based on their financial history.

## Project Overview
### Financial institutions need to assess the risk before approving loans. This project builds a credit risk prediction model that helps classify borrowers into different priority levels based on their likelihood of repayment.


Target Variable (Output: Priority Levels)
Borrowers are classified into different priority levels based on credit risk:

p1 (Highest Priority – Low Risk) 🟢 → Most financially stable borrowers

p2 (Medium-High Priority) 🟡 → Slightly higher risk than p1 but still reliable

p3 (Medium-Low Priority) 🟠 → Moderate risk of default

p4 (Lowest Priority – High Risk) 🔴 → Highest risk of default

Machine Learning Approach
We used various machine learning algorithms to predict borrower risk:
✔ CatBoost Classifier
✔ AdaBoost Classifier
✔ Random Forest
✔ Gradient Boosting (CatBoost, XGBoost)

The model is trained to classify borrowers into priority levels, helping financial institutions make better loan decisions.

## 📁 Project Structure

credit-risk-modeling/
│── notebooks/              # Jupyter Notebooks for EDA & Model Training
│── src/                    # Source code for model pipeline
│── artifacts/              # Stored model artifacts (ignored in Git)
│── Logs/                   # Logs from training & evaluation (ignored in Git)
│── requirements.txt        # Python dependencies
│── setup.py                # Project setup
│── template.py             # Project template
│── predict_unseen_data.py  # Script to predict on new data
│── .gitignore              # Files to ignore in Git
│── README.md               # Project Documentation (You are here! 📄)

