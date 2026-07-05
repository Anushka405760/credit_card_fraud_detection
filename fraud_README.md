# Credit Card Fraud Detection Model

A machine learning project that detects fraudulent credit card transactions by comparing multiple classification algorithms on highly imbalanced real-world data.

## Overview

Credit card fraud detection is a classic imbalanced classification problem — fraud transactions are extremely rare compared to legitimate ones. This project handles class imbalance using undersampling and compares four ML models to identify the best performer.

## Features

- **Data Analysis** — Explored distribution of fraud vs legitimate transactions (492 fraud out of 284,315 total)
- **Handling Class Imbalance** — Applied undersampling to create a balanced dataset for fair model training
- **Model Comparison** — Trained and compared Logistic Regression, Decision Tree, Random Forest, and KNN
- **Evaluation** — Models evaluated using accuracy, precision, recall, F1 score, and confusion matrix

## Dataset

- 284,807 real credit card transactions
- Highly imbalanced — only 0.17% fraudulent transactions
- Features: V1-V28 (PCA transformed), Amount, Time
- Target: Class (0 = Legitimate, 1 = Fraud)
- Source: Kaggle — Credit Card Fraud Detection Dataset

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN)

## Key Concepts

- **Class Imbalance** — Dataset had 492 fraud vs 284,315 legitimate transactions. Random undersampling applied to balance classes before training
- **Undersampling** — Randomly sampled 492 legitimate transactions to match fraud count, creating a 50-50 balanced dataset
- **Multiple Model Comparison** — Four algorithms trained and evaluated side by side to find best performer

## Results

- Logistic Regression and Random Forest achieved 93% accuracy
- Models evaluated using precision, recall, F1 score, and confusion matrix
- Random Forest best suited for production due to robustness on imbalanced data

## How to Run

1. Clone the repository
2. Install dependencies:
```
pip install pandas numpy scikit-learn
```
3. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. Open and run `Credit_Card_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab
