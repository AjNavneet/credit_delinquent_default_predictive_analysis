# Credit Delinquent/Default Preditive Analysis

## Business Context
Banks generate revenue primarily by lending money to borrowers. To maximize profits and maintain a good reputation, they need to identify borrowers who may have trouble repaying their loans. When borrowers fail to make payments, we often categorize them as "delinquent" or "defaulted."

- Delinquent borrowers are slightly behind on their payments.
- Defaulted borrowers have been behind for a long time and are unlikely to repay.
- This project aims to identify borrowers likely to default in the next two years after being seriously delinquent for more than three months.
- We use various borrower characteristics and historical data to predict this risk.
- These predictions help banks take proactive measures.

---

## Objective
Build a model using borrower information and historical records to predict serious delinquency within the next two years.

---

## Tech Stack
- Language: `Python`
- Libraries: 
    - `Pandas`, `Matplotlib`, `NumPy`, `Scikit Learn`, `Imblearn`
    - `Shap` and `LIME` for model interpretation.
    - `Keras`

---

## Concepts Explored

- Exploratory Data Analysis (EDA) to understand data distribution and feature behavior in relation to the target variable (SeriousDelinquencyin2Years).
- Data preprocessing, including error handling, outlier treatment, and missing value imputation.
- Splitting the dataset using Stratified Sampling.
- Feature engineering for better model decision-making.
- Feature scaling using BoxCox transformation and standardization.
- Training a model using a Neural Network as a deep learning architecture.
- Training a model using statistical techniques like Logistic Regression.
- Training a model using tree-based algorithms (Bagging and Boosting).
- Hyperparameter tuning and its impact on model performance.
- Recursive Feature Elimination using Cross Validation to identify highly correlated features.
- Evaluating model performance using metrics such as F1 score, Precision, Recall, and AUCROC.
- Model interpretation using SHAP (global) and LIME (local) explanations.

---
