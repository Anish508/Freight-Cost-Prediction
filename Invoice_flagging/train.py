"""
train.py — Invoice Flagging Training Pipeline
==============================================
"""

from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from model_preprocessing import (
    add_risk_label,
    load_data,
    prepare_features,
    select_significant_features,
)
from model_evaluation import (
    evaluate_model,
    get_feature_importance,
    scale_features,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
    tune_random_forest,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = "data/inventory.db"

# Selected features (including engineered deltas & ratios)
SELECTED_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "dollar_discrepancy",
    "qty_discrepancy",
    "freight_ratio",
    "dollar_diff_ratio",
    "days_po_to_invoice",
]

TARGET = "flag_invoice"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = Path("models")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(tune: bool = False) -> None:
    """
    Run the end-to-end invoice-flagging training pipeline.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load data
    print("Loading data ...")
    df = load_data(DB_PATH)
    print(f"  Rows loaded : {len(df):,}")

    # 2. Engineer features & create risk label
    df = add_risk_label(df)
    label_counts = df[TARGET].value_counts()
    print(f"\nLabel distribution:\n{label_counts.to_string()}")

    # 3. Statistical feature selection check
    sig, non_sig = select_significant_features(df, SELECTED_FEATURES, target=TARGET)
    print(f"\nStatistically significant features (p < 0.05): {sig}")

    # 4. Prepare X / y
    X, y = prepare_features(df, feature_cols=SELECTED_FEATURES, target_col=TARGET)

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain size : {len(X_train):,}  |  Test size : {len(X_test):,}")

    # 6. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_type="minmax"
    )

    # 7. Train baseline models
    print("\n--- Training baseline models ---")
    lr_model = train_logistic_regression(X_train_scaled, y_train, random_state=RANDOM_STATE)
    dt_model = train_decision_tree(X_train_scaled, y_train, random_state=RANDOM_STATE)
    rf_model = train_random_forest(X_train_scaled, y_train, random_state=RANDOM_STATE)

    # 8. Evaluate baseline models
    print("\n--- Evaluating baseline models ---")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    dt_results = evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree Classifier")
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest Classifier")

    all_results = [lr_results, dt_results, rf_results]

    # 9. Feature importance (Random Forest)
    fi_df = get_feature_importance(rf_model, SELECTED_FEATURES)
    print("\nFeature Importances (Random Forest):")
    print(fi_df.to_string(index=False))

    # 10. Best model selection
    best_model_info = max(all_results, key=lambda r: r["f1"])
    model_map = {
        "Logistic Regression": lr_model,
        "Decision Tree Classifier": dt_model,
        "Random Forest Classifier": rf_model,
    }
    best_model = model_map[best_model_info["model_name"]]
    print(f"\nBest baseline model: {best_model_info['model_name']} (F1 = {best_model_info['f1']:.4f})")

    # 11. Save best model and scaler
    model_path = MODEL_DIR / "invoice_flagging_model.pkl"
    scaler_path = MODEL_DIR / "invoice_flagging_scaler.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved  : {model_path}")
    print(f"Scaler saved : {scaler_path}")


if __name__ == "__main__":
    main(tune=False)
