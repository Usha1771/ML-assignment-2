import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def run_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_test_pred)

    return pd.DataFrame({
        'Model': ['Decision Tree'],
        'Accuracy': [acc],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'MCC': [mcc]
    })
