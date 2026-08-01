import pandas as pd
import sqlite3
from scipy.stats import ttest_ind


def load_data(db_path: str) -> pd.DataFrame:
    """
    Connect to the SQLite database and load the merged vendor invoice
    + purchase aggregation dataset into a DataFrame.

    The query joins vendor_invoice with an aggregated purchases CTE to
    produce one row per PO-invoice, enriched with item-level totals and
    an average receiving delay.
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


def create_invoice_risk_label(row) -> int:
    """
    Rule-based binary label for invoice risk.

    Returns 1 (flag for manual review) when:
      - The invoice-level dollar total differs from the item-level total by > $5, OR
      - The average receiving delay is abnormally high (> 10 days).
    Returns 0 otherwise.
    """
    # Invoice total mismatch with item-level total
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1

    # Abnormally high receiving delay
    if row["avg_receiving_delay"] > 10:
        return 1

    return 0


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply `create_invoice_risk_label` row-wise and store the result in a
    new column called ``flag_invoice``.
    """
    df = df.copy()
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def select_significant_features(
    df: pd.DataFrame,
    candidate_features: list[str],
    target: str = "flag_invoice",
    alpha: float = 0.05,
) -> tuple[list[str], list[str]]:
    """
    Run a Welch two-sample t-test for each candidate feature between the
    flagged and non-flagged groups.

    Parameters
    ----------
    df                  : DataFrame that already contains the target column.
    candidate_features  : Feature names to test.
    target              : Binary target column (0 / 1).
    alpha               : Significance threshold (default 0.05).

    Returns
    -------
    significant_features     : Features where p-value < alpha.
    non_significant_features : Remaining features.
    """
    flagged = df[df[target] == 1]
    normal = df[df[target] == 0]

    significant_features: list[str] = []
    non_significant_features: list[str] = []

    for metric in candidate_features:
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

    Parameters
    ----------
    df           : Labelled DataFrame (must already have ``flag_invoice``).
    feature_cols : Columns to use as predictors.  Defaults to the five
                   most important features identified during EDA.
    target_col   : Name of the binary target column.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if feature_cols is None:
        feature_cols = [
            "invoice_quantity",
            "invoice_dollars",
            "Freight",
            "total_item_quantity",
            "total_item_dollars",
        ]

    X = df[feature_cols]
    y = df[target_col]
    return X, y
