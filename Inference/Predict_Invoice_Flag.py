"""
Predict_Invoice_Flag.py — Invoice Risk Flagging Inference Layer
================================================================

Loads the saved Random Forest classifier + MinMaxScaler and predicts whether
a vendor invoice should be flagged for manual review using a Hybrid (Rule + ML) Architecture.
Supports configurable risk tolerances and multi-factor risk audit tags.
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
# Core inference function (Hybrid Architecture: Configurable Rules + ML Model)
# ---------------------------------------------------------------------------

def predict_invoice_flag(
    invoice_quantity: float | list,
    invoice_dollars:  float | list,
    Freight:          float | list,
    total_item_quantity: float | list,
    total_item_dollars:  float | list,
    days_po_to_invoice:  float | list = 0.0,
    dollar_tolerance:    float = 5.0,
    prob_cutoff:         float = 0.50,
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
    dollar_tolerance     : Maximum allowed dollar discrepancy tolerance (default $5.00).
    prob_cutoff          : ML model flag probability threshold (default 0.50).

    Returns
    -------
    pd.DataFrame with input features plus:
      - ``flag_invoice``     : 1 = flag for review, 0 = auto-approve
      - ``flag_probability`` : Confidence score that the invoice should be flagged
      - ``risk_reasons``     : Specific risk factors triggered
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
    df_in["dollar_discrepancy"] = (df_in["invoice_dollars"] - df_in["total_item_dollars"]).abs().round(2)
    df_in["qty_discrepancy"] = (df_in["invoice_quantity"] - df_in["total_item_quantity"]).abs()
    df_in["freight_ratio"] = (df_in["Freight"] / (df_in["invoice_dollars"] + 1e-5)).round(4)
    df_in["dollar_diff_ratio"] = (df_in["dollar_discrepancy"] / (df_in["total_item_dollars"] + 1e-5)).round(4)

    X = df_in[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # ML Model Prediction
    ml_flags = model.predict(X_scaled)
    ml_probs = model.predict_proba(X_scaled)[:, 1]

    # Multi-Factor Risk Assessment Engine
    final_flags = []
    final_probs = []
    risk_reasons_list = []

    for idx, row in df_in.iterrows():
        reasons = []
        d_disc = row["dollar_discrepancy"]
        q_disc = row["qty_discrepancy"]
        prob = float(ml_probs[idx])

        if d_disc > dollar_tolerance:
            reasons.append(f"Price Discrepancy (${d_disc:,.2f} > ${dollar_tolerance:,.2f})")
        if q_disc > 0:
            reasons.append(f"Quantity Mismatch ({int(q_disc)} units)")
        if row["days_po_to_invoice"] > 15:
            reasons.append(f"PO Delay ({int(row['days_po_to_invoice'])} days)")
        if prob >= prob_cutoff and not reasons:
            reasons.append(f"ML Anomaly Score ({prob:.1%} >= {prob_cutoff:.1%})")

        if reasons:
            final_flags.append(1)
            final_probs.append(max(prob, 1.0 if d_disc > dollar_tolerance or q_disc > 0 else prob))
            risk_reasons_list.append("; ".join(reasons))
        else:
            final_flags.append(0)
            final_probs.append(prob)
            risk_reasons_list.append("None (All Audit Rules Passed)")

    result = df_in.copy()
    result["flag_invoice"]     = final_flags
    result["flag_probability"] = [round(p, 3) for p in final_probs]
    result["risk_reasons"]     = risk_reasons_list
    result["decision"]         = result["flag_invoice"].map(
        {1: "[FLAG]    Manual Review Required",
         0: "[APPROVE] Auto-process"}
    )
    return result


def main() -> None:
    print("=" * 60)
    print("  Invoice Risk Flagging — Multi-Factor Risk Audit Demo")
    print("=" * 60)

    sample_data = {
        "invoice_quantity":    [6,      15,     10100],
        "invoice_dollars":     [214.26, 1850.00, 137483],
        "Freight":             [3.47,   45.00,   2935.2],
        "total_item_quantity": [6,      15,     10100],
        "total_item_dollars":  [214.26, 1200.00, 1000],
    }

    result = predict_invoice_flag(**sample_data)
    print(result[["invoice_dollars", "total_item_dollars", "flag_invoice", "decision", "risk_reasons"]].to_string(index=False))


if __name__ == "__main__":
    main()
