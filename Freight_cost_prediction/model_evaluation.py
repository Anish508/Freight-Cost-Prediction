import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)


# ── Regression model trainers ──────────────────────────────────────────────────

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, max_depth=6):
    model = RandomForestRegressor(
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ── Regression evaluation ──────────────────────────────────────────────────────

def evaluate_regression_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model using MAE, RMSE, and R².
    Used by train.py for freight cost prediction.
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{model_name} Performance:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2 * 100:.2f}%")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ── Classification evaluation ──────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a classification model using Accuracy and Classification Report.
    Used by Risk_Flagging.ipynb for invoice risk flagging.
    """
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report_str = classification_report(y_test, preds)
    report_dict = classification_report(y_test, preds, output_dict=True)

    print(f"\n{model_name} Performance:")
    print(f"  Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(report_str)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "classification_report": report_dict,
    }