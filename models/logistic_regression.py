import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

def run_logistic_regression(X_train, y_train, X_test, y_test):
    # Split training into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_split, y_train_split)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr')
    precision = precision_score(y_val, y_val_pred, average='weighted')
    recall = recall_score(y_val, y_val_pred, average='weighted')
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    mcc = matthews_corrcoef(y_val, y_val_pred)

    return pd.DataFrame({
        'Model': ['Logistic Regression'],
        'Accuracy': [acc],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'MCC': [mcc]
    })
