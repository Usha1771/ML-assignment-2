import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def run_xgboost(X_train, y_train, X_test, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train_enc)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test_enc, y_test_pred)
    auc = roc_auc_score(y_test_enc, y_test_proba, multi_class='ovr')
    precision = precision_score(y_test_enc, y_test_pred, average='weighted')
    recall = recall_score(y_test_enc, y_test_pred, average='weighted')
    f1 = f1_score(y_test_enc, y_test_pred, average='weighted')
    mcc = matthews_corrcoef(y_test_enc, y_test_pred)

    return pd.DataFrame({
        'Model': ['XGBoost'],
        'Accuracy': [acc],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'MCC': [mcc]
    })
