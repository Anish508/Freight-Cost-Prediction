"""
Predict_Freight.py — Freight Cost & Overcharge Inference
========================================================

Loads the saved Random Forest model and predicts expected freight charges
from invoice dollar amounts. Optionally compares actual billed freight against
the ML baseline to detect freight overcharges.
"""

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

def predict_freight(
    dollars: float | list[float],
    actual_freight: float | list[float] | None = None,
    variance_threshold_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Predict expected freight cost from invoice dollar amount(s) and optionally
    audit actual billed freight for overcharge anomalies.

    Parameters
    ----------
    dollars                : Invoice dollar value(s).
    actual_freight         : Optional actual billed freight cost(s).
    variance_threshold_pct : Overcharge alert threshold percentage (default 15.0%).

    Returns
    -------
    pd.DataFrame with:
      - ``invoice_dollars``     : Total invoice dollar amount
      - ``predicted_freight``   : ML baseline predicted freight charge
      - ``actual_freight``      : Billed freight (if provided)
      - ``freight_variance``    : Dollar difference (actual - predicted)
      - ``variance_pct``        : Percentage variance over baseline
      - ``freight_status``      : NORMAL | OVERCHARGED | BELOW EXPECTED
    """
    model = joblib.load(_MODEL_PATH)

    def _to_list(v):
        return [v] if isinstance(v, (int, float)) else list(v)

    dlr_list = _to_list(dollars)
    X = pd.DataFrame({"Dollars": dlr_list})
    predictions = model.predict(X).round(2)

    result = pd.DataFrame({
        "invoice_dollars": dlr_list,
        "predicted_freight": predictions,
    })

    if actual_freight is not None:
        act_list = _to_list(actual_freight)
        if len(act_list) == 1 and len(dlr_list) > 1:
            act_list = act_list * len(dlr_list)

        result["actual_freight"] = [round(a, 2) for a in act_list]
        result["freight_variance"] = (result["actual_freight"] - result["predicted_freight"]).round(2)
        
        pcts = []
        statuses = []
        for p, a, v in zip(result["predicted_freight"], result["actual_freight"], result["freight_variance"]):
            pct = (v / p * 100) if p > 0 else 0.0
            pcts.append(round(pct, 1))
            if v > 5.0 and pct > variance_threshold_pct:
                statuses.append("OVERCHARGED")
            elif v < -5.0:
                statuses.append("BELOW EXPECTED")
            else:
                statuses.append("NORMAL")

        result["variance_pct"] = pcts
        result["freight_status"] = statuses

    return result


def main() -> None:
    print("=" * 60)
    print("  Freight Cost Prediction & Overcharge Audit — Demo")
    print("=" * 60)

    sample_dollars = [214.26, 1850.00, 137483.78]
    actual_freight = [3.47, 120.00, 2935.20]

    result = predict_freight(sample_dollars, actual_freight=actual_freight)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
