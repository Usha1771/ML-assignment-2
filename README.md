# ML Classification Models Comparison

## Problem Statement
This project implements and compares six machine learning classification models to predict wine quality using chemical properties. The objective is to design a complete ML pipeline that covers model training, evaluation, and deployment through an interactive Streamlit web application.

The assignment requires:
- Implementation of six classification models  
- Evaluation using multiple metrics  
- Interactive web application for model comparison and prediction  
- Deployment on Streamlit Community Cloud  

## Dataset Description
**Dataset:** Wine Quality (Red Wine) — UCI Machine Learning Repository  
**Source:** UCI ML Repository - Wine Quality  

The dataset contains physicochemical properties of red wine samples and their quality ratings. Quality is scored on a scale from 3 to 8, making this a multi-class classification problem.

**Features (11):** Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol  

**Target Variable:** Quality (score between 3 and 8 → six classes)  
**Dataset Statistics:** 1,599 instances, 11 features, 6 classes, no missing values  

## Models Used
- Logistic Regression  
- Decision Tree  
- k‑Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

## Final Comparison Table

| ML Model Name             | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|---------------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression       | 0.5820   | 0.8012  | 0.5651    | 0.5820  | 0.5670   | 0.3255  |
| Decision Tree             | 0.6094   | 0.6584  | 0.6121    | 0.6094  | 0.6095   | 0.3982  |
| kNN                       | 0.6094   | 0.6983  | 0.5841    | 0.6094  | 0.5959   | 0.3733  |
| Naive Bayes (GaussianNB)  | 0.5625   | 0.6838  | 0.5745    | 0.5625  | 0.5681   | 0.3299  |
| Random Forest             | 0.6750   | 0.7907  | 0.6539    | 0.6750  | 0.6599   | 0.4746  |
| XGBoost                   | 0.6594   | 0.8374  | 0.6520    | 0.6594  | 0.6488   | 0.4535  |

## Observations on Model Performance

| ML Model Name       | Observation |
|---------------------|-------------|
| Logistic Regression | Fast to train but limited by non-linear relationships. Struggles with minority classes. |
| Decision Tree       | Captures non-linear patterns but prone to overfitting. Better than Logistic Regression but affected by class imbalance. |
| kNN                 | Slightly better than Decision Tree. Effective at capturing local patterns but computationally expensive and sensitive to k. |
| Naive Bayes         | Lowest accuracy but reasonable AUC. Assumption of independence doesn’t hold well here. Provides a quick baseline. |
| Random Forest       | Best accuracy (67.5%) and strong MCC. Ensemble reduces overfitting and generalizes well. Robust predictions compared to single models. |
| XGBoost             | Highest AUC (0.8374) and strong overall performance. Slightly lower accuracy than Random Forest but excellent discrimination ability. |

## Project Structure
ML-2/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── results_summary.csv     # Evaluation metrics table
├── 2025aa05614_ml_assignment.ipynb  # Jupyter notebook
├── models/                 # Classifier implementations
│   ├── init.py
│   ├── main.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost.py
└── .gitignore


## Installation and Setup
Prerequisites: Python 3.8+ and pip  

```bash
git clone <repository-url>
cd ML-2
pip install -r requirements.txt
python models/main.py
streamlit run app.py
