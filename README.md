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
**Source:** [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

The dataset contains physicochemical properties of red wine samples and their quality ratings. Quality is scored on a scale from 3 to 8, making this a multi-class classification problem.

**Features (11 input features):**
1. **Fixed Acidity** - Most acids involved with wine or fixed or nonvolatile
2. **Volatile Acidity** - The amount of acetic acid in wine
3. **Citric Acid** - Found in small quantities, adds freshness and flavor
4. **Residual Sugar** - The amount of sugar remaining after fermentation stops
5. **Chlorides** - The amount of salt in the wine
6. **Free Sulfur Dioxide** - The free form of SO2 exists in equilibrium between molecular SO2 and bisulfite ion
7. **Total Sulfur Dioxide** - Amount of free and bound forms of S02
8. **Density** - The density of water is close to that of water depending on the percent alcohol and sugar content
9. **pH** - Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic)
10. **Sulphates** - A wine additive which can contribute to sulfur dioxide gas (S02) levels
11. **Alcohol** - The percent alcohol content of the wine 

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
| Logistic Regression | Logistic Regression achieved 58.20% accuracy with 80.12% AUC. While it provides a linear decision boundary and is fast to train, its performance is limited by the non-linear relationships in the wine quality data. The model struggles with the multi-class classification task, particularly for minority classes (quality 3, 4, 8).  |
| Decision Tree       | Decision Tree achieved 60.94% accuracy with 65.84% AUC. The model can capture non-linear relationships but shows signs of overfitting. It performs better than Logistic Regression but still struggles with class imbalance, especially for rare quality levels.|
| kNN                 | K-Nearest Neighbors achieved 60.94% accuracy with 69.83% AUC, performing slightly better than Decision Tree. The model benefits from feature scaling and captures local patterns effectively. However, it's computationally expensive and sensitive to the choice of k parameter. |
| Naive Bayes         | Naive Bayes achieved the lowest accuracy (56.25%) but a reasonable AUC (68.38%). The model's assumption of feature independence doesn't hold well for wine quality data, where chemical properties are correlated. Despite this, it provides a fast baseline model. |
| Random Forest       | Random Forest achieved the best performance (67.50% accuracy, 79.07% AUC). By combining multiple decision trees, it effectively reduces overfitting and handles non-linear relationships. The ensemble approach provides robust predictions and better generalization compared to individual models. |
| XGBoost             | XGBoost achieved the second-best performance (65.94% accuracy, 83.74% AUC). It effectively captures complex patterns and feature interactions. While slightly lower than Random Forest in this case, it shows strong performance with good generalization. The model benefits from gradient boosting's ability to correct errors iteratively. |

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
