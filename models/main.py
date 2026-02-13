import pandas as pd

# Import all model functions from the package
from models import (
    run_logistic_regression,
    run_decision_tree,
    run_knn,
    run_naive_bayes,
    run_random_forest,
    run_xgboost
)

def main(X_train, y_train, X_test, y_test):
    # Run each model and collect results
    results = pd.concat([
        run_logistic_regression(X_train, y_train, X_test, y_test),
        run_decision_tree(X_train, y_train, X_test, y_test),
        run_knn(X_train, y_train, X_test, y_test),
        run_naive_bayes(X_train, y_train, X_test, y_test),
        run_random_forest(X_train, y_train, X_test, y_test),
        run_xgboost(X_train, y_train, X_test, y_test)
    ], ignore_index=True)

    # Save to CSV
    results.to_csv("results_summary.csv", index=False)
    print("✅ Results saved to results_summary.csv")

    return results

# Allow running directly
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Load your wine dataset
    data = pd.read_csv("winequality-red.csv")

    # Normalize column names to avoid KeyError
    data.columns = data.columns.str.strip().str.lower()

    # Separate features and target
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Run all models
    main(X_train, y_train, X_test, y_test)
