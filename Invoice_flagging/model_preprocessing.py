import pandas as pd
import sqlite3
from scipy.stats import ttest_ind


def load_data(db_path: str) -> pd.DataFrame:
    """
    Connect to the SQLite database and load the merged vendor invoice
    + purchase aggregation dataset into a DataFrame.
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        WITH purchase_agg AS (
            SELECT
                p.PONumber,
                COUNT(DISTINCT p.Brand) AS total_brands,
                SUM(p.Quantity)         AS total_item_quantity,
                SUM(p.Dollars)          AS total_item_dollars,
                AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
            FROM purchases p
            GROUP BY p.PONumber
        )

        SELECT
            vi.PONumber,
            vi.Quantity                                              AS invoice_quantity,
            vi.Dollars                                               AS invoice_dollars,
            vi.Freight,
            (julianday(vi.InvoiceDate) - julianday(vi.PODate))      AS days_po_to_invoice,
            (julianday(vi.PayDate)     - julianday(vi.InvoiceDate)) AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay

        FROM vendor_invoice AS vi

        LEFT JOIN purchase_agg AS pa
            ON vi.PONumber = pa.PONumber;
        """,
        conn,
    )

    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate explicit delta and ratio features required for non-linear tree splits:
      - dollar_discrepancy : abs(invoice_dollars - total_item_dollars)
      - qty_discrepancy    : abs(invoice_quantity - total_item_quantity)
      - freight_ratio      : Freight / (invoice_dollars + 1e-5)
      - dollar_diff_ratio  : dollar_discrepancy / (total_item_dollars + 1e-5)
    """
    df = df.copy()

    # Fill NA values gracefully
    df["total_item_dollars"] = df["total_item_dollars"].fillna(df["invoice_dollars"])
    df["total_item_quantity"] = df["total_item_quantity"].fillna(df["invoice_quantity"])
    df["days_po_to_invoice"] = df["days_po_to_invoice"].fillna(0.0)

    # Engineered delta features
    df["dollar_discrepancy"] = (df["invoice_dollars"] - df["total_item_dollars"]).abs()
    df["qty_discrepancy"] = (df["invoice_quantity"] - df["total_item_quantity"]).abs()
    df["freight_ratio"] = df["Freight"] / (df["invoice_dollars"] + 1e-5)
    df["dollar_diff_ratio"] = df["dollar_discrepancy"] / (df["total_item_dollars"] + 1e-5)

    return df


def create_invoice_risk_label(row) -> int:
    """
    Cleaned Ground Truth Risk Label Generator.

    Targeting Risk Indicators Available at Inference Time:
      1. Dollar Discrepancy > $5
      2. Quantity Discrepancy > 0
      3. PO-to-Invoice Delay > 15 days
    """
    if row["dollar_discrepancy"] > 5:
        return 1
    if row["qty_discrepancy"] > 0:
        return 1
    if row["days_po_to_invoice"] > 15:
        return 1

    return 0


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and `create_invoice_risk_label` row-wise.
    """
    df = engineer_features(df)
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def select_significant_features(
    df: pd.DataFrame,
    candidate_features: list[str],
    target: str = "flag_invoice",
    alpha: float = 0.05,
) -> tuple[list[str], list[str]]:
    """
    Run a Welch two-sample t-test for each candidate feature.
    """
    flagged = df[df[target] == 1]
    normal = df[df[target] == 0]

    significant_features: list[str] = []
    non_significant_features: list[str] = []

    for metric in candidate_features:
        if metric not in df.columns:
            continue
        _, p_value = ttest_ind(
            flagged[metric].dropna(),
            normal[metric].dropna(),
            equal_var=False,
        )

        if p_value < alpha:
            significant_features.append(metric)
        else:
            non_significant_features.append(metric)

    return significant_features, non_significant_features


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "flag_invoice",
):
    """
    Split the DataFrame into feature matrix X and target vector y.
    """
    if feature_cols is None:
        feature_cols = [
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

    X = df[feature_cols]
    y = df[target_col]
    return X, y
