"""
Predict_Freight.py — Freight Cost Inference
============================================

Loads the saved Random Forest model and predicts freight cost for new
vendor invoice data.

Usage (from the project root or the Inference/ directory):

    python Inference/Predict_Freight.py

Or import ``predict_freight`` to call it programmatically.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # d:\PDS Project
_MODEL_PATH = _ROOT / "Freight_cost_prediction" / "models" / "predict_freight_model.pkl"


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict_freight(dollars: float | list[float]) -> pd.DataFrame:
    """
    Predict freight cost from invoice dollar amount(s).

    Parameters
    ----------
    dollars : A single dollar value (float) or a list of dollar values.

    Returns
    -------
    pd.DataFrame with columns ``invoice_dollars`` and ``predicted_freight``.

    Example
    -------
    >>> result = predict_freight(5000.0)
    >>> print(result)
       invoice_dollars  predicted_freight
    0           5000.0              123.4
    """
    model = joblib.load(_MODEL_PATH)

    if isinstance(dollars, (int, float)):
        dollars = [dollars]

    X = pd.DataFrame({"Dollars": dollars})
    predictions = model.predict(X)

    result = pd.DataFrame({
        "invoice_dollars": dollars,
        "predicted_freight": predictions.round(2),
    })
    return result


# ---------------------------------------------------------------------------
# CLI demo — runs when executed directly
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  Freight Cost Prediction — Inference Demo")
    print("=" * 55)

    # Sample invoice dollar values
    sample_dollars = [214.26, 140.55, 106.60, 137483.78, 15527.25, 3608.11]

    result = predict_freight(sample_dollars)

    print(f"\nModel loaded from:\n  {_MODEL_PATH}\n")
    print("Sample Predictions:")
    print("-" * 45)
    print(result.to_string(index=False))
    print("-" * 45)

    # Interactive mode
    print("\nEnter invoice dollar amount to predict freight (or 'q' to quit):")
    while True:
        raw = input("  Invoice $ > ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        try:
            amount = float(raw)
            r = predict_freight(amount)
            print(f"  → Predicted Freight: ${r['predicted_freight'].iloc[0]:.2f}\n")
        except ValueError:
            print("  ⚠  Please enter a valid number.\n")


if __name__ == "__main__":
    main()
