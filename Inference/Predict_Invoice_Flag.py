"""
Predict_Invoice_Flag.py — Invoice Risk Flagging Inference
=========================================================

Loads the saved Random Forest classifier + MinMaxScaler and predicts whether
a vendor invoice should be flagged for manual review.

Usage (from the project root or the Inference/ directory):

    python Inference/Predict_Invoice_Flag.py

Or import ``predict_invoice_flag`` to call it programmatically.
"""

from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # d:\PDS Project
_MODEL_PATH  = _ROOT / "Invoice_flagging" / "models" / "invoice_flagging_model.pkl"
_SCALER_PATH = _ROOT / "Invoice_flagging" / "models" / "invoice_flagging_scaler.pkl"

# Feature columns expected by the model (must match training order)
FEATURE_COLS = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict_invoice_flag(
    invoice_quantity: float | list,
    invoice_dollars:  float | list,
    Freight:          float | list,
    total_item_quantity: float | list,
    total_item_dollars:  float | list,
) -> pd.DataFrame:
    """
    Predict whether one or more invoices should be flagged for manual review.

    Parameters
    ----------
    invoice_quantity     : Number of items on the invoice.
    invoice_dollars      : Total dollar amount on the invoice.
    freight              : Freight charge on the invoice.
    total_item_quantity  : Total item quantity from purchase records (PO level).
    total_item_dollars   : Total item dollars from purchase records (PO level).

    All parameters can be scalar floats or equal-length lists.

    Returns
    -------
    pd.DataFrame with input features plus:
      - ``flag_invoice``     : 1 = flag for review, 0 = auto-approve
      - ``flag_probability`` : Model's confidence that the invoice should be flagged
      - ``decision``         : Human-readable label

    Example
    -------
    >>> result = predict_invoice_flag(
    ...     invoice_quantity=6,
    ...     invoice_dollars=214.26,
    ...     freight=3.47,
    ...     total_item_quantity=6,
    ...     total_item_dollars=214.26,
    ... )
    """
    model  = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)

    # Normalise scalars to lists
    def _to_list(v):
        return [v] if isinstance(v, (int, float)) else list(v)

    data = {
        "invoice_quantity":    _to_list(invoice_quantity),
        "invoice_dollars":     _to_list(invoice_dollars),
        "Freight":             _to_list(Freight),
        "total_item_quantity": _to_list(total_item_quantity),
        "total_item_dollars":  _to_list(total_item_dollars),
    }

    X = pd.DataFrame(data, columns=FEATURE_COLS)
    X_scaled = scaler.transform(X)

    flags = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]          # P(flag=1)

    result = X.copy()
    result["flag_invoice"]     = flags
    result["flag_probability"] = probs.round(3)
    result["decision"] = result["flag_invoice"].map(
        {1: "[FLAG]    Manual Review Required",
         0: "[APPROVE] Auto-process"}
    )
    return result


# ---------------------------------------------------------------------------
# CLI demo — runs when executed directly
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Invoice Risk Flagging — Inference Demo")
    print("=" * 60)
    print(f"\nModel loaded from:\n  {_MODEL_PATH}")
    print(f"Scaler loaded from:\n  {_SCALER_PATH}\n")

    # Sample invoices (mix of normal and risky)
    sample_data = {
        "invoice_quantity":    [6,      15,     10100,  1935,   90],
        "invoice_dollars":     [214.26, 140.55, 137483, 15527,  1563],
        "Freight":             [3.47,   8.57,   2935.2, 429.2,  8.60],
        "total_item_quantity": [6,      15,     10100,  1935,   223],
        "total_item_dollars":  [214.26, 140.55, 1000,   15527,  6823],
        # ↑ row 3 has invoice_dollars >> total_item_dollars → should be flagged
    }


    result = predict_invoice_flag(**sample_data)

    print("Sample Predictions:")
    print("-" * 60)
    display_cols = ["invoice_dollars", "Freight", "flag_invoice",
                    "flag_probability", "decision"]
    print(result[display_cols].to_string(index=False))
    print("-" * 60)

    # Interactive single-invoice prediction
    print("\n--- Interactive Prediction ---")
    print("Enter invoice details (or 'q' to quit):\n")

    while True:
        raw = input("  invoice_quantity    > ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        try:
            iq  = float(raw)
            iD  = float(input("  invoice_dollars     > "))
            fr  = float(input("  Freight             > "))
            tiq = float(input("  total_item_quantity > "))
            tiD = float(input("  total_item_dollars  > "))

            r = predict_invoice_flag(
                invoice_quantity=iq,
                invoice_dollars=iD,
                Freight=fr,
                total_item_quantity=tiq,
                total_item_dollars=tiD,
            )
            print(f"\n  → Decision          : {r['decision'].iloc[0]}")
            print(f"  → Flag Probability  : {r['flag_probability'].iloc[0]:.1%}\n")
        except ValueError:
            print("  ⚠  Please enter valid numbers.\n")


if __name__ == "__main__":
    main()
