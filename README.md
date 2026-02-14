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

```
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
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-assignment-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   python download_dataset.py
   ```
   Alternatively, download manually from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) and save as `winequality-red.csv` in the project root.

4. **Train the models**
   ```bash
   python train_models.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train all 6 models
   - Calculate evaluation metrics
   - Save models and metrics to the `model/` directory

5. **Run the Streamlit app locally**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Streamlit Application Features

1. **Model Comparison Page**
   - View comparison table of all models
   - Select a model to see detailed metrics
   - View confusion matrix and classification report

2. **Predict on New Data Page**
   - Upload CSV file with test data
   - Select a model for prediction
   - View predictions and download results

3. **Dataset Info Page**
   - Learn about the dataset
   - View feature statistics

## Deployment

### Streamlit Community Cloud

**Quick Deployment Steps:**

1. **Push your code to GitHub** (already done ✅)
   ```bash
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App**
   - Click "New App" button
   - Select repository: `Usha1771/ML-assignment-2`
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Deployment takes 2-5 minutes
   - Monitor the deployment logs
   - Your app will be live at: `https://ml-assignment-2-p76petyy6p94fvkbjyvwqt.streamlit.app`

**📖 Detailed Guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for complete step-by-step instructions and troubleshooting.

**✅ Deployment Checklist:** See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) to ensure everything is ready.

### Automatic Updates

- Streamlit Cloud automatically redeploys when you push to the `main` branch
- Just push your changes: `git push origin main`
- Wait 2-5 minutes for automatic redeployment

## Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **AUC Score**: Area Under the ROC Curve (one-vs-rest for multi-class)
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **MCC Score**: Matthews Correlation Coefficient, a balanced measure for multi-class classification

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

## Author

Usha Kiran


## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.

## Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- BITS Pilani for the assignment framework
