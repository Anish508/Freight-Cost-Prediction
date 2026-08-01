"""
Predict_Invoice_Flag.py — Invoice Risk Flagging Inference Layer
================================================================

Loads the saved Random Forest classifier + MinMaxScaler and predicts whether
a vendor invoice should be flagged for manual review using a Hybrid (Rule + ML) Architecture.
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

# Feature columns expected by the trained model (must match training order)
FEATURE_COLS = [
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


# ---------------------------------------------------------------------------
# Core inference function (Hybrid Architecture: Rule Engine + ML Model)
# ---------------------------------------------------------------------------

def predict_invoice_flag(
    invoice_quantity: float | list,
    invoice_dollars:  float | list,
    Freight:          float | list,
    total_item_quantity: float | list,
    total_item_dollars:  float | list,
    days_po_to_invoice:  float | list = 0.0,
) -> pd.DataFrame:
    """
    Predict whether one or more invoices should be flagged for manual review.

    Parameters
    ----------
    invoice_quantity     : Number of items on the invoice.
    invoice_dollars      : Total dollar amount on the invoice.
    Freight              : Freight charge on the invoice.
    total_item_quantity  : Total item quantity from purchase records (PO level).
    total_item_dollars   : Total item dollars from purchase records (PO level).
    days_po_to_invoice   : Days between PO Date and Invoice Date (default 0.0).

    Returns
    -------
    pd.DataFrame with input features plus:
      - ``flag_invoice``     : 1 = flag for review, 0 = auto-approve
      - ``flag_probability`` : Confidence score that the invoice should be flagged
      - ``decision``         : Human-readable label
    """
    model  = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)

    def _to_list(v):
        return [v] if isinstance(v, (int, float)) else list(v)

    iq  = _to_list(invoice_quantity)
    iD  = _to_list(invoice_dollars)
    fr  = _to_list(Freight)
    tiq = _to_list(total_item_quantity)
    tiD = _to_list(total_item_dollars)
    dpi = _to_list(days_po_to_invoice)

    # Make length of dpi match other inputs if scalar default was passed
    if len(dpi) == 1 and len(iq) > 1:
        dpi = dpi * len(iq)

    df_in = pd.DataFrame({
        "invoice_quantity": iq,
        "invoice_dollars": iD,
        "Freight": fr,
        "total_item_quantity": tiq,
        "total_item_dollars": tiD,
        "days_po_to_invoice": dpi,
    })

    # Engineer delta and ratio features
    df_in["dollar_discrepancy"] = (df_in["invoice_dollars"] - df_in["total_item_dollars"]).abs()
    df_in["qty_discrepancy"] = (df_in["invoice_quantity"] - df_in["total_item_quantity"]).abs()
    df_in["freight_ratio"] = df_in["Freight"] / (df_in["invoice_dollars"] + 1e-5)
    df_in["dollar_diff_ratio"] = df_in["dollar_discrepancy"] / (df_in["total_item_dollars"] + 1e-5)

    X = df_in[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # ML Model Prediction
    ml_flags = model.predict(X_scaled)
    ml_probs = model.predict_proba(X_scaled)[:, 1]

    # Hybrid Rule Engine Override (Deterministic Hard Flags)
    final_flags = []
    final_probs = []

    for idx, row in df_in.iterrows():
        # Hard Rule 1: Dollar Discrepancy > $5
        # Hard Rule 2: Quantity Discrepancy > 0
        if row["dollar_discrepancy"] > 5 or row["qty_discrepancy"] > 0:
            final_flags.append(1)
            final_probs.append(1.0)
        else:
            final_flags.append(int(ml_flags[idx]))
            final_probs.append(float(ml_probs[idx]))

    result = df_in.copy()
    result["flag_invoice"]     = final_flags
    result["flag_probability"] = [round(p, 3) for p in final_probs]
    result["decision"] = result["flag_invoice"].map(
        {1: "[FLAG]    Manual Review Required",
         0: "[APPROVE] Auto-process"}
    )
    return result


def main() -> None:
    print("=" * 60)
    print("  Invoice Risk Flagging — Inference Demo (Hybrid Architecture)")
    print("=" * 60)

    # Perfect sample invoice test
    sample_data = {
        "invoice_quantity":    [6,      15,     10100],
        "invoice_dollars":     [214.26, 140.55, 137483],
        "Freight":             [3.47,   8.57,   2935.2],
        "total_item_quantity": [6,      15,     10100],
        "total_item_dollars":  [214.26, 140.55, 1000],
    }

    result = predict_invoice_flag(**sample_data)

    print("Predictions:")
    print("-" * 60)
    display_cols = ["invoice_dollars", "Freight", "flag_invoice", "flag_probability", "decision"]
    print(result[display_cols].to_string(index=False))
    print("-" * 60)


if __name__ == "__main__":
    main()
