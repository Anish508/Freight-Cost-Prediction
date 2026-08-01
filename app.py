"""
app.py  —  PDS Project  |  Procurement Analytics Dashboard
============================================================
Streamlit front-end for two ML inference modules:
  • Freight Cost Prediction  (regression)
  • Invoice Risk Flagging     (classification)

Run from the project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure Inference/ is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference.Predict_Freight import predict_freight
from Inference.Predict_Invoice_Flag import predict_invoice_flag

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Procurement Analytics | PDS Project",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Typography & base ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1544g2n {
        padding-top: 1.5rem;
    }

    /* ---- Main background ---- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ---- Page title banner ---- */
    .page-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
    }
    .page-banner h1 {
        color: #f1f5f9;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .page-banner p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ---- Cards ---- */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* ---- Result badges ---- */
    .badge-approve {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 0.35rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-flag {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 0.35rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 0.7rem;
        margin: 1.5rem 0 0.9rem 0;
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ---- Input labels ---- */
    label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: #1e3a5f;
        color: #f1f5f9;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.8rem;
        font-size: 0.9rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s ease;
    }
    .stButton > button:hover {
        background: #2d5a9e;
        color: #ffffff;
    }

    /* Hide Streamlit default header elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ---- Mobile Responsiveness ---- */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem 2rem 0.5rem !important;
        }
        .page-banner {
            padding: 1.2rem 1rem !important;
            margin-bottom: 1.2rem !important;
        }
        .page-banner h1 {
            font-size: 1.35rem !important;
        }
        .page-banner p {
            font-size: 0.85rem !important;
        }
        .stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.6rem !important;
        }
        .metric-card {
            padding: 0.9rem 0.8rem !important;
        }
        .metric-card .value {
            font-size: 1.4rem !important;
        }
        .metric-card .label {
            font-size: 0.7rem !important;
        }
        .section-header {
            font-size: 0.95rem !important;
            margin: 1.2rem 0 0.7rem 0 !important;
        }
    }
    @media screen and (max-width: 480px) {
        .stats-grid {
            grid-template-columns: 1fr !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='margin-bottom:1.2rem;'>"
        "<div style='font-size:1.05rem;font-weight:700;color:#f1f5f9;'>PDS Project</div>"
        "<div style='font-size:0.78rem;color:#94a3b8;'>Procurement Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=["Overview", "Freight Cost Prediction", "Invoice Risk Flagging"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;'>"
        "Models<br>"
        "<span style='color:#94a3b8;'>Random Forest Regressor</span><br>"
        "<span style='color:#94a3b8;'>Random Forest Classifier</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "Overview":
    st.markdown(
        """
        <div class="page-banner">
            <h1>Procurement Analytics & Risk Management System</h1>
            <p>An end-to-end Machine Learning solution for Freight Cost Estimation and Invoice Risk Flagging.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Statistics Bar
    st.markdown(
        """
        <div class="stats-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.8rem;">
            <div class="metric-card">
                <div class="label">Total Invoices Analyzed</div>
                <div class="value">5,543</div>
                <div class="sub">Historical dataset (2024)</div>
            </div>
            <div class="metric-card">
                <div class="label">Invoice Risk Accuracy</div>
                <div class="value" style="color: #16a34a;">89.0%</div>
                <div class="sub">Random Forest Classifier</div>
            </div>
            <div class="metric-card">
                <div class="label">Flagged Precision</div>
                <div class="value" style="color: #2563eb;">94.0%</div>
                <div class="sub">Low false alarm rate</div>
            </div>
            <div class="metric-card">
                <div class="label">Active ML Modules</div>
                <div class="value">2</div>
                <div class="sub">Regression & Classification</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tabs = st.tabs([
        "Project Executive Summary",
        "ML Modules & Architecture",
        "Dataset & Feature Engineering",
    ])

    # ---- TAB 1: EXECUTIVE SUMMARY ----
    with overview_tabs[0]:
        st.markdown("<div class='section-header'>The Core Problems & AI Solutions</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                """
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;height:100%;">
                    <div style="font-size:0.75rem;font-weight:700;color:#2563eb;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">
                        Problem Statement 1 — Unpredictable Freight Costs
                    </div>
                    <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
                        Freight Overcharge & Variance Risk
                    </div>
                    <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
                        Procurement teams process thousands of invoices where freight charges fluctuate based on product mix, order size, and vendor billing policies. Without automated benchmarks, overcharges go undetected.
                    </div>
                    <div style="margin-top:1.2rem;padding-top:0.8rem;border-top:1px solid #cbd5e1;font-size:0.83rem;color:#1e293b;">
                        <strong>AI Solution:</strong> Random Forest Regression model that predicts standard expected freight costs ($) for any invoice dollar total, giving finance teams an instant validation baseline.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;height:100%;">
                    <div style="font-size:0.75rem;font-weight:700;color:#dc2626;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.6rem;">
                        Problem Statement 2 — Manual Review Bottlenecks
                    </div>
                    <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
                        High-Volume Invoice Audit Bottlenecks
                    </div>
                    <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
                        Manual auditing of every invoice is labor-intensive and inefficient. Conversely, skipping manual verification exposes the enterprise to billing discrepancies, delayed deliveries, and pricing mismatches.
                    </div>
                    <div style="margin-top:1.2rem;padding-top:0.8rem;border-top:1px solid #cbd5e1;font-size:0.83rem;color:#1e293b;">
                        <strong>AI Solution:</strong> Automated binary risk classifier that screens invoices and flags high-risk transactions for human review with a <strong>94% precision rate</strong>, enabling 67% of low-risk invoices to be auto-processed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-header'>Business Impact & Operational Benefits</div>", unsafe_allow_html=True)
        
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                """
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:0.3rem;">Operational Efficiency</div>
                    <div style="font-size:0.85rem;color:#64748b;">Reduces manual audit workload by up to 66%, allowing finance teams to focus only on flagged high-risk items.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b2:
            st.markdown(
                """
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:0.3rem;">Financial Risk Reduction</div>
                    <div style="font-size:0.85rem;color:#64748b;">Detects price mismatches (> $5 discrepancy) and abnormal supplier delivery delays (> 10 days) automatically.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                """
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:1.2rem;">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:0.3rem;">Batch Automation</div>
                    <div style="font-size:0.85rem;color:#64748b;">Supports single-record lookups and bulk CSV processing with instant exportable CSV reports for ERP integration.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- TAB 2: ML MODULES & ARCHITECTURE ----
    with overview_tabs[1]:
        st.markdown("<div class='section-header'>End-to-End System Pipeline</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#0f172a;color:#f8fafc;padding:1.2rem;border-radius:8px;font-family:monospace;font-size:0.83rem;line-height:1.8;overflow-x:auto;">
            [SQLite Data Source] -> [SQL Query & CTE Aggregations] -> [Welch t-Test Feature Selection]
                                                                                |
                                                                                v
            [Streamlit UI Dashboard] <- [Inference Layer (Python)] <- [MinMaxScaler & Trained Models]
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header'>Model Benchmark Comparison</div>", unsafe_allow_html=True)

        perf_df = pd.DataFrame([
            {
                "Module": "Freight Cost Prediction",
                "Task": "Regression",
                "Best Model": "Random Forest Regressor",
                "Key Metric": "MAE: ~2.4, R²: ~82%",
                "Features Used": "Invoice Dollar Amount",
                "Status": "Production Ready",
            },
            {
                "Module": "Invoice Risk Flagging",
                "Task": "Binary Classification",
                "Best Model": "Random Forest Classifier",
                "Key Metric": "Accuracy: 89.0%, Precision: 94.0%, F1: 0.81",
                "Features Used": "5 Core Features (Qty, Dollars, Freight, PO Qty, PO Dollars)",
                "Status": "Production Ready",
            },
        ])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>Classifier Performance Breakdown (Risk Flagging)</div>", unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns([1, 1], gap="large")
        with m_col1:
            st.markdown("##### Classification Metrics (Test Set: 1,109 Invoices)")
            report_df = pd.DataFrame([
                {"Class": "Class 0 (Auto-Approve)", "Precision": "87.0%", "Recall": "98.0%", "F1-Score": "0.92", "Support": "725"},
                {"Class": "Class 1 (Flagged)", "Precision": "94.0%", "Recall": "72.0%", "F1-Score": "0.81", "Support": "384"},
                {"Class": "Overall Model Accuracy", "Precision": "-", "Recall": "-", "F1-Score": "89.0%", "Support": "1,109"},
            ])
            st.dataframe(report_df, use_container_width=True, hide_index=True)

        with m_col2:
            st.markdown("##### Confusion Matrix")
            cm_df = pd.DataFrame(
                [[707, 18], [109, 275]],
                columns=["Predicted: Auto-Approve", "Predicted: Flagged"],
                index=["Actual: Auto-Approve", "Actual: Flagged"],
            )
            st.dataframe(cm_df, use_container_width=True)

    # ---- TAB 3: DATASET & FEATURE ENGINEERING ----
    with overview_tabs[2]:
        st.markdown("<div class='section-header'>SQLite Database Schema (`inventory.db`)</div>", unsafe_allow_html=True)
        
        schema_df = pd.DataFrame([
            {"Table Name": "vendor_invoice", "Record Count": "5,543", "Key Columns": "PONumber, VendorNumber, Quantity, Dollars, Freight, InvoiceDate, PODate, PayDate"},
            {"Table Name": "purchases", "Record Count": "~2.4M", "Key Columns": "PONumber, Brand, Description, PurchasePrice, Quantity, Dollars, PODate, ReceivingDate"},
            {"Table Name": "purchase_prices", "Record Count": "~12K", "Key Columns": "Brand, Description, Price, PurchasePrice, VendorNumber"},
            {"Table Name": "begin_inventory", "Record Count": "~200K", "Key Columns": "InventoryId, Store, Brand, onHand, Price, startDate"},
            {"Table Name": "end_inventory", "Record Count": "~220K", "Key Columns": "InventoryId, Store, Brand, onHand, Price, endDate"},
        ])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>Feature Importance (Random Forest Risk Classifier)</div>", unsafe_allow_html=True)
        
        fi_df = pd.DataFrame([
            {"Feature": "total_item_dollars", "Importance Weight": "27.1%", "Source": "Purchases aggregation (PO level)"},
            {"Feature": "total_item_quantity", "Importance Weight": "19.9%", "Source": "Purchases aggregation (PO level)"},
            {"Feature": "invoice_dollars", "Importance Weight": "18.9%", "Source": "Vendor invoice record"},
            {"Feature": "Freight", "Importance Weight": "17.8%", "Source": "Vendor invoice record"},
            {"Feature": "invoice_quantity", "Importance Weight": "16.1%", "Source": "Vendor invoice record"},
        ])
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.82rem;color:#94a3b8;'>Use the sidebar to navigate to a prediction module.</div>",
        unsafe_allow_html=True,
    )




# ===========================================================================
# PAGE: Freight Cost Prediction
# ===========================================================================
elif page == "Freight Cost Prediction":
    st.markdown(
        """
        <div class="page-banner">
            <h1>Freight Cost Prediction</h1>
            <p>Estimate freight charges from invoice dollar amounts using a trained Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # ---- Single prediction ------------------------------------------------
    with tab1:
        st.markdown("<div class='section-header'>Invoice Details</div>", unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            invoice_dollars = st.number_input(
                "Invoice Dollar Amount ($)",
                min_value=0.01,
                value=5000.00,
                step=100.0,
                format="%.2f",
                help="Total dollar value of the vendor invoice.",
            )
            predict_btn = st.button("Run Prediction", key="freight_single")

        with col_out:
            if predict_btn:
                with st.spinner("Running model..."):
                    result = predict_freight(invoice_dollars)
                    predicted = float(result["predicted_freight"].iloc[0])
                    ratio = (predicted / invoice_dollars * 100) if invoice_dollars > 0 else 0

                st.markdown("<div class='section-header'>Result</div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Invoice Amount</div>"
                        f"<div class='value'>${invoice_dollars:,.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Predicted Freight</div>"
                        f"<div class='value'>${predicted:,.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Freight / Invoice</div>"
                        f"<div class='value'>{ratio:.2f}%</div>"
                        f"<div class='sub'>Cost ratio</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ---- Batch prediction ------------------------------------------------
    with tab2:
        st.markdown("<div class='section-header'>Batch Input</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.85rem;color:#64748b;margin-bottom:0.8rem;'>"
            "Enter one invoice dollar amount per line."
            "</div>",
            unsafe_allow_html=True,
        )

        raw_input = st.text_area(
            "Invoice Dollar Amounts (one per line)",
            value="214.26\n140.55\n106.60\n137483.78\n15527.25\n3608.11",
            height=180,
            label_visibility="collapsed",
        )
        batch_btn = st.button("Run Batch Prediction", key="freight_batch")

        if batch_btn:
            try:
                values = [float(line.strip()) for line in raw_input.strip().splitlines() if line.strip()]
                if not values:
                    st.warning("No valid values found.")
                else:
                    with st.spinner("Running model..."):
                        result = predict_freight(values)
                        result["freight_ratio_%"] = (
                            result["predicted_freight"] / result["invoice_dollars"] * 100
                        ).round(2)

                    st.markdown("<div class='section-header'>Batch Results</div>", unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='label'>Invoices Processed</div>"
                            f"<div class='value'>{len(result)}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='label'>Total Invoice Value</div>"
                            f"<div class='value'>${result['invoice_dollars'].sum():,.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='label'>Total Predicted Freight</div>"
                            f"<div class='value'>${result['predicted_freight'].sum():,.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    display_df = result.rename(columns={
                        "invoice_dollars":    "Invoice Amount ($)",
                        "predicted_freight":  "Predicted Freight ($)",
                        "freight_ratio_%":    "Freight Ratio (%)",
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    csv = display_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results as CSV",
                        data=csv,
                        file_name="freight_predictions.csv",
                        mime="text/csv",
                    )

            except ValueError:
                st.error("Please enter valid numeric values, one per line.")


# ===========================================================================
# PAGE: Invoice Risk Flagging
# ===========================================================================
elif page == "Invoice Risk Flagging":
    st.markdown(
        """
        <div class="page-banner">
            <h1>Invoice Risk Flagging</h1>
            <p>Classify vendor invoices as auto-approve or flag for manual review using a trained Random Forest classifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Single Invoice", "Batch via CSV"])

    # ---- Single invoice ---------------------------------------------------
    with tab1:
        st.markdown("<div class='section-header'>Invoice Features</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            inv_qty = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=6,
                step=1,
                help="Total number of items listed on this invoice.",
            )
            inv_dollars = st.number_input(
                "Invoice Dollar Amount ($)",
                min_value=0.01,
                value=214.26,
                step=10.0,
                format="%.2f",
                help="Total dollar amount on the invoice.",
            )
            freight = st.number_input(
                "Freight Charge ($)",
                min_value=0.0,
                value=3.47,
                step=1.0,
                format="%.2f",
                help="Freight/shipping charge on the invoice.",
            )

        with col_b:
            total_item_qty = st.number_input(
                "Total Item Quantity (PO Level)",
                min_value=1,
                value=6,
                step=1,
                help="Aggregated item quantity from the corresponding purchase order records.",
            )
            total_item_dollars = st.number_input(
                "Total Item Dollars (PO Level) ($)",
                min_value=0.01,
                value=214.26,
                step=10.0,
                format="%.2f",
                help="Aggregated dollar total from the corresponding purchase order records.",
            )

        dollar_diff = abs(inv_dollars - total_item_dollars)
        if dollar_diff > 0:
            st.markdown(
                f"<div style='font-size:0.83rem;color:#92400e;background:#fef3c7;"
                f"border:1px solid #fde68a;border-radius:6px;padding:0.5rem 0.8rem;margin-top:0.5rem;'>"
                f"Dollar discrepancy: ${dollar_diff:,.2f} between invoice and PO-level totals."
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        flag_btn = st.button("Classify Invoice", key="flag_single")

        if flag_btn:
            with st.spinner("Running classifier..."):
                result = predict_invoice_flag(
                    invoice_quantity=inv_qty,
                    invoice_dollars=inv_dollars,
                    Freight=freight,
                    total_item_quantity=total_item_qty,
                    total_item_dollars=total_item_dollars,
                )
                flag_val = int(result["flag_invoice"].iloc[0])
                prob_val = float(result["flag_probability"].iloc[0])

            st.markdown("<div class='section-header'>Classification Result</div>", unsafe_allow_html=True)

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                badge = (
                    "<span class='badge-flag'>FLAGGED — Manual Review</span>"
                    if flag_val == 1
                    else "<span class='badge-approve'>APPROVED — Auto-process</span>"
                )
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>Decision</div>"
                    f"<div style='margin-top:0.6rem;'>{badge}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with res_col2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>Flag Probability</div>"
                    f"<div class='value'>{prob_val:.1%}</div>"
                    f"<div class='sub'>Confidence (flag=1)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with res_col3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>Dollar Discrepancy</div>"
                    f"<div class='value'>${dollar_diff:,.2f}</div>"
                    f"<div class='sub'>Invoice vs. PO total</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if flag_val == 1:
                st.markdown(
                    "<div style='background:#fef2f2;border-left:4px solid #ef4444;"
                    "border-radius:4px;padding:0.9rem 1rem;margin-top:1rem;"
                    "font-size:0.88rem;color:#991b1b;'>"
                    "<strong>Action Required:</strong> This invoice has been flagged for manual review. "
                    "Check for dollar amount mismatches, excessive freight charges, or receiving delays."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:#f0fdf4;border-left:4px solid #22c55e;"
                    "border-radius:4px;padding:0.9rem 1rem;margin-top:1rem;"
                    "font-size:0.88rem;color:#166534;'>"
                    "<strong>No Issues Detected:</strong> This invoice meets all automated criteria "
                    "and can be processed without manual intervention."
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ---- Batch via CSV ----------------------------------------------------
    with tab2:
        st.markdown("<div class='section-header'>Upload Invoice CSV</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.85rem;color:#64748b;margin-bottom:0.6rem;'>"
            "CSV must contain these columns (exact names):"
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(
            "invoice_quantity, invoice_dollars, Freight, total_item_quantity, total_item_dollars",
            language=None,
        )

        # Downloadable sample template
        sample_template = pd.DataFrame({
            "invoice_quantity":    [6,      15,     10100,  1935],
            "invoice_dollars":     [214.26, 140.55, 137483, 15527],
            "Freight":             [3.47,   8.57,   2935.2, 429.2],
            "total_item_quantity": [6,      15,     10100,  1935],
            "total_item_dollars":  [214.26, 140.55, 1000,   15527],
        })
        st.download_button(
            "Download Sample Template",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="invoice_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            try:
                df_input = pd.read_csv(uploaded)
                required_cols = [
                    "invoice_quantity", "invoice_dollars", "Freight",
                    "total_item_quantity", "total_item_dollars",
                ]
                missing = [c for c in required_cols if c not in df_input.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    with st.spinner("Running batch classification..."):
                        results = predict_invoice_flag(
                            invoice_quantity=df_input["invoice_quantity"].tolist(),
                            invoice_dollars=df_input["invoice_dollars"].tolist(),
                            Freight=df_input["Freight"].tolist(),
                            total_item_quantity=df_input["total_item_quantity"].tolist(),
                            total_item_dollars=df_input["total_item_dollars"].tolist(),
                        )

                    total = len(results)
                    flagged_count = int(results["flag_invoice"].sum())
                    approved_count = total - flagged_count

                    st.markdown(
                        "<div class='section-header'>Batch Results</div>",
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f"<div class='metric-card'><div class='label'>Total Invoices</div>"
                            f"<div class='value'>{total}</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"<div class='metric-card'><div class='label'>Flagged</div>"
                            f"<div class='value' style='color:#dc2626;'>{flagged_count}</div>"
                            f"<div class='sub'>{flagged_count/total*100:.1f}%</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"<div class='metric-card'><div class='label'>Approved</div>"
                            f"<div class='value' style='color:#16a34a;'>{approved_count}</div>"
                            f"<div class='sub'>{approved_count/total*100:.1f}%</div></div>",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        avg_prob = float(results["flag_probability"].mean())
                        st.markdown(
                            f"<div class='metric-card'><div class='label'>Avg Flag Probability</div>"
                            f"<div class='value'>{avg_prob:.1%}</div></div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Highlight flagged rows
                    display_results = results.rename(columns={
                        "invoice_quantity":    "Qty",
                        "invoice_dollars":     "Invoice ($)",
                        "Freight":             "Freight ($)",
                        "total_item_quantity": "PO Qty",
                        "total_item_dollars":  "PO Dollars ($)",
                        "flag_invoice":        "Flag",
                        "flag_probability":    "Probability",
                        "decision":            "Decision",
                    })

                    st.dataframe(
                        display_results.style.applymap(
                            lambda v: "background-color:#fee2e2;color:#991b1b;"
                            if v == 1 else "background-color:#dcfce7;color:#166534;",
                            subset=["Flag"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_out = display_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results as CSV",
                        data=csv_out,
                        file_name="invoice_flagging_results.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
