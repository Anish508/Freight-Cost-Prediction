"""
Procurement Analytics & Risk Management System — Streamlit Application
=======================================================================
An end-to-end Machine Learning web interface for Freight Cost Estimation
and Invoice Risk Flagging.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path configuration — ensure Inference package is importable
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Inference.Predict_Freight import predict_freight
from Inference.Predict_Invoice_Flag import predict_invoice_flag

# ---------------------------------------------------------------------------
# Navigation callback helper
# ---------------------------------------------------------------------------
def set_page(target_page: str):
    st.session_state["nav_page"] = target_page

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Page configuration & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Procurement Analytics System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Base Font, Layout & Colors */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #0f172a !important;
            background-color: #f8fafc !important;
            overflow-x: hidden !important;
        }

        /* Prevent Horizontal Scroll / Overflow on Mobile */
        .stApp > header, .main, .block-container {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* High-Contrast Inputs & Labels */
        label, [data-testid="stWidgetLabel"] p, .stNumberInput label, .stTextInput label, .stFileUploader label, .stSelectbox label {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
            opacity: 1 !important;
            visibility: visible !important;
            margin-bottom: 0.3rem !important;
        }

        .stNumberInput input, .stTextInput input, .stTextArea textarea {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        }

        /* Tab Header Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f1f5f9;
            padding: 6px;
            border-radius: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            white-space: pre;
            border-radius: 8px;
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
            padding: 0px 18px;
            background-color: transparent;
            border: none !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #2563eb !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
        }

        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #f8fafc;
        }

        .sidebar-brand-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .sidebar-brand-title {
            color: #38bdf8 !important;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
        }
        .sidebar-brand-sub {
            color: #94a3b8 !important;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }

        /* FIX: Expander Hover Disappearing Issue in Sidebar */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            margin-bottom: 1.2rem !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
            transition: background-color 0.2s ease, color 0.2s ease !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:active {
            background-color: #334155 !important;
            color: #38bdf8 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
            color: inherit !important;
            fill: currentColor !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover * {
            color: #38bdf8 !important;
            fill: #38bdf8 !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #1e293b !important;
            border-top: 1px solid #334155 !important;
            padding: 1rem !important;
        }

        /* Sidebar Sliders & Tooltips */
        [data-testid="stSidebar"] .stSlider label {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            color: #38bdf8 !important;
        }

        [data-testid="stSidebar"] .stSlider [role="slider"] {
            background-color: #38bdf8 !important;
            border: 2px solid #ffffff !important;
        }

        .sidebar-status-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1.5rem;
        }
        .sidebar-status-header {
            font-size: 0.75rem;
            font-weight: 700;
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.6rem;
        }
        .sidebar-status-item {
            font-size: 0.85rem;
            color: #e2e8f0 !important;
            margin-bottom: 0.3rem;
        }
        .sidebar-status-item span {
            font-weight: 700;
            color: #38bdf8 !important;
        }

        /* Fully Responsive Stats Grid for Mobile & Desktop */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
            margin-bottom: 1.8rem;
            width: 100%;
            box-sizing: border-box;
        }

        /* Metric Cards */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            box-sizing: border-box;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .metric-card .label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.25rem;
            line-height: 1.2;
        }
        .metric-card .sub {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.2rem;
            font-weight: 500;
        }

        /* Mobile Breakpoints for Responsive Grid */
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 0.85rem;
            }
            .metric-card {
                padding: 1rem 1.1rem !important;
            }
            .metric-card .value {
                font-size: 1.4rem !important;
            }
            .metric-card .label {
                font-size: 0.72rem !important;
            }
            .metric-card .sub {
                font-size: 0.75rem !important;
            }
        }

        @media (max-width: 480px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Badges */
        .badge-approve {
            background-color: #dcfce7;
            color: #166534;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-flag {
            background-color: #fee2e2;
            color: #991b1b;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-warn {
            background-color: #fef3c7;
            color: #92400e;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: inline-block;
        }

        /* Section Header */
        .section-header {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }

        /* Hero Banner */
        .page-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 14px;
            padding: 2rem;
            color: #ffffff;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .page-banner h1 {
            color: #ffffff !important;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .page-banner p {
            color: #94a3b8 !important;
            font-size: 0.95rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# SIDEBAR NAVIGATION & ENTERPRISE CONTROLS
# ===========================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-title">Procurement Analytics</div>
            <div class="sidebar-brand-sub">Enterprise AI System • v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=["Overview", "Freight Cost Prediction", "Invoice Risk Flagging"],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Enterprise Audit Controls", expanded=False):
        st.markdown("<div style='font-size:0.8rem;color:#cbd5e1;margin-bottom:0.5rem;'>Risk Sensitivity Settings</div>", unsafe_allow_html=True)
        dollar_tolerance = st.slider(
            "Max Dollar Discrepancy ($)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Maximum allowed price difference between invoice and PO before hard flagging.",
        )
        prob_cutoff = st.slider(
            "ML Anomaly Threshold (%)",
            min_value=30,
            max_value=90,
            value=50,
            step=5,
            help="Minimum ML probability required to trigger an anomaly review.",
        ) / 100.0

    st.markdown(
        """
        <div class="sidebar-status-card">
            <div class="sidebar-status-header">System Health & Engine</div>
            <div class="sidebar-status-item">Engine: <span>Random Forest</span></div>
            <div class="sidebar-status-item">Models Loaded: <span>2 Active</span></div>
            <div class="sidebar-status-item">Status: <span>Online</span></div>
        </div>
        """,
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

    # Key Statistics Bar — Responsive 5-Card Layout
    st.markdown(
        """
        <div class="stats-grid">
            <div class="metric-card">
                <div class="label">Total Invoices Analyzed</div>
                <div class="value">5,543</div>
                <div class="sub">Historical dataset (2024)</div>
            </div>
            <div class="metric-card">
                <div class="label">Freight Model Accuracy</div>
                <div class="value" style="color: #0284c7;">97.1% R²</div>
                <div class="sub">Random Forest Regressor</div>
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

    # Overview Quick Action Navigation Section
    st.markdown("<div class='section-header'>Quick Actions — Try ML Features</div>", unsafe_allow_html=True)
    act_col1, act_col2 = st.columns(2, gap="large")
    with act_col1:
        st.button(
            "Launch Freight Cost Predictor",
            key="launch_freight_btn",
            on_click=set_page,
            args=("Freight Cost Prediction",),
        )
    with act_col2:
        st.button(
            "Launch Invoice Risk Classifier",
            key="launch_flagging_btn",
            on_click=set_page,
            args=("Invoice Risk Flagging",),
        )

    st.markdown("<br>", unsafe_allow_html=True)

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
                        <strong>AI Solution:</strong> Random Forest Regression model (97.1% R² score) that predicts standard expected freight costs ($) for any invoice dollar total, giving finance teams an instant validation baseline.
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
                        Problem Statement 2 — High Risk of Overbilling & Fraud
                    </div>
                    <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;">
                        Manual Audit Bottlenecks & Leakage
                    </div>
                    <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
                        Manual auditing of every vendor invoice is slow, expensive, and error-prone. Mismatches between PO quantities, billed amounts, and shipping fees result in significant financial leakage.
                    </div>
                    <div style="margin-top:1.2rem;padding-top:0.8rem;border-top:1px solid #cbd5e1;font-size:0.83rem;color:#1e293b;">
                        <strong>AI Solution:</strong> Hybrid Decision Classifier combining hard business rules with a Random Forest model to flag high-risk invoices for targeted review (94% precision).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- TAB 2: ML MODULES & ARCHITECTURE ----
    with overview_tabs[1]:
        st.markdown("<div class='section-header'>System Architecture & Model Highlights</div>", unsafe_allow_html=True)

        m1, m2 = st.columns(2, gap="large")

        with m1:
            st.markdown(
                """
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:1.4rem;">
                    <div style="font-weight:800;font-size:1.05rem;color:#0f172a;margin-bottom:0.4rem;">
                        Module 1 — Freight Cost Regressor
                    </div>
                    <div style="font-size:0.85rem;color:#64748b;margin-bottom:0.8rem;">
                        Predicts freight charges based on invoice total ($)
                    </div>
                    <ul style="font-size:0.85rem;color:#334155;padding-left:1.2rem;line-height:1.7;">
                        <li><strong>Algorithm:</strong> Random Forest Regressor</li>
                        <li><strong>Accuracy (R²):</strong> 97.1% Variance Explained ($24.78 MAE)</li>
                        <li><strong>Input:</strong> Total Invoice Dollars</li>
                        <li><strong>Output:</strong> Predicted Freight ($) baseline</li>
                        <li><strong>UseCase:</strong> Overcharge & shipping variance audit</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                """
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:1.4rem;">
                    <div style="font-weight:800;font-size:1.05rem;color:#0f172a;margin-bottom:0.4rem;">
                        Module 2 — Invoice Risk Flagging Classifier
                    </div>
                    <div style="font-size:0.85rem;color:#64748b;margin-bottom:0.8rem;">
                        Hybrid Classifier for high-risk invoice detection
                    </div>
                    <ul style="font-size:0.85rem;color:#334155;padding-left:1.2rem;line-height:1.7;">
                        <li><strong>Algorithm:</strong> Hybrid Rule Engine + Random Forest Classifier</li>
                        <li><strong>Inputs:</strong> Quantities, Dollars, Freight, PO totals, Delays</li>
                        <li><strong>Accuracy:</strong> 89.0% Test Accuracy | 94.0% Flag Precision</li>
                        <li><strong>UseCase:</strong> Zero false alarms on exact matches; flags overcharges</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- TAB 3: DATASET & FEATURE ENGINEERING ----
    with overview_tabs[2]:
        st.markdown("<div class='section-header'>Dataset Profile & Feature Engineering</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.9rem;color:#334155;line-height:1.7;">
                The dataset is built from <strong>5,543 historical purchase and vendor invoice records</strong>.
                Advanced CTE SQL queries merge transaction header information with itemized PO lines to produce engineered features:
            </div>
            """,
            unsafe_allow_html=True,
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                """
                <div class="metric-card" style="margin-top:1rem;">
                    <div class="label">Dollar Discrepancy</div>
                    <div class="value" style="font-size:1.2rem;">abs(Inv $ - PO $)</div>
                    <div class="sub">Direct financial mismatch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                """
                <div class="metric-card" style="margin-top:1rem;">
                    <div class="label">Quantity Variance</div>
                    <div class="value" style="font-size:1.2rem;">abs(Inv Qty - PO Qty)</div>
                    <div class="sub">Item delivery mismatch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                """
                <div class="metric-card" style="margin-top:1rem;">
                    <div class="label">Freight Cost Ratio</div>
                    <div class="value" style="font-size:1.2rem;">Freight / Invoice $</div>
                    <div class="sub">Normalized shipping intensity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE: Freight Cost Prediction & Overcharge Audit
# ===========================================================================
elif page == "Freight Cost Prediction":
    st.markdown(
        """
        <div class="page-banner">
            <h1>Freight Cost Prediction & Overcharge Audit</h1>
            <p>Estimate expected freight charges and audit actual billed freight against ML baseline predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Single Invoice Audit", "Batch Freight Evaluation"])

    # ---- Single prediction ------------------------------------------------
    with tab1:
        st.markdown("<div class='section-header'>Invoice & Freight Details</div>", unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            invoice_dollars = st.number_input(
                "Invoice Dollar Amount ($)",
                min_value=0.01,
                value=214.26,
                step=50.0,
                format="%.2f",
                help="Total dollar value of the vendor invoice.",
            )
            actual_freight_in = st.number_input(
                "Actual Billed Freight ($)",
                min_value=0.00,
                value=3.47,
                step=1.0,
                format="%.2f",
                help="Actual freight charge listed on the invoice (leave as is to compare).",
            )
            predict_btn = st.button("Run Freight Audit", key="freight_single")

        with col_out:
            if predict_btn:
                with st.spinner("Calculating ML baseline..."):
                    result = predict_freight(invoice_dollars, actual_freight=actual_freight_in)
                    predicted = float(result["predicted_freight"].iloc[0])
                    actual = float(result["actual_freight"].iloc[0])
                    var_dollars = float(result["freight_variance"].iloc[0])
                    var_pct = float(result["variance_pct"].iloc[0])
                    status = str(result["freight_status"].iloc[0])

                st.markdown("<div class='section-header'>Audit Result</div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Predicted Freight</div>"
                        f"<div class='value'>${predicted:,.2f}</div>"
                        f"<div class='sub'>ML Baseline Cost</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Billed Freight</div>"
                        f"<div class='value'>${actual:,.2f}</div>"
                        f"<div class='sub'>Actual Invoice Charge</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st_color = "#dc2626" if status == "OVERCHARGED" else "#16a34a"
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='label'>Freight Variance</div>"
                        f"<div class='value' style='color:{st_color};'>${var_dollars:+,.2f}</div>"
                        f"<div class='sub'>{var_pct:+.1f}% vs baseline</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if status == "OVERCHARGED":
                    st.markdown(
                        f"<div style='background:#fef2f2;border-left:4px solid #ef4444;"
                        f"border-radius:4px;padding:0.9rem 1rem;margin-top:1rem;"
                        f"font-size:0.88rem;color:#991b1b;'>"
                        f"<strong>Overcharge Warning:</strong> Billed freight (${actual:,.2f}) is "
                        f"<strong>${var_dollars:,.2f} ({var_pct:+.1f}%) higher</strong> than the expected ML baseline (${predicted:,.2f})."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background:#f0fdf4;border-left:4px solid #22c55e;"
                        f"border-radius:4px;padding:0.9rem 1rem;margin-top:1rem;"
                        f"font-size:0.88rem;color:#166534;'>"
                        f"<strong>Freight Normal:</strong> Billed freight charge is within standard expected benchmarks."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ---- Batch prediction ------------------------------------------------
    with tab2:
        st.markdown("<div class='section-header'>Batch Freight Baseline Input</div>", unsafe_allow_html=True)
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
        batch_btn = st.button("Run Batch Evaluation", key="freight_batch")

        if batch_btn:
            lines = [l.strip() for l in raw_input.strip().split("\n") if l.strip()]
            amounts = []
            for l in lines:
                try:
                    amounts.append(float(l))
                except ValueError:
                    pass

            if not amounts:
                st.error("Please enter at least one valid numeric amount.")
            else:
                with st.spinner("Predicting freight for batch..."):
                    results = predict_freight(amounts)

                st.markdown("<div class='section-header'>Batch Predictions</div>", unsafe_allow_html=True)
                st.dataframe(
                    results.rename(columns={
                        "invoice_dollars": "Invoice ($)",
                        "predicted_freight": "Predicted Freight ($)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_out,
                    file_name="predicted_freight_batch.csv",
                    mime="text/csv",
                )


# ===========================================================================
# PAGE: Invoice Risk Flagging (Hybrid Architecture & Multi-Factor Audit)
# ===========================================================================
elif page == "Invoice Risk Flagging":
    st.markdown(
        """
        <div class="page-banner">
            <h1>Invoice Risk Flagging & Audit System</h1>
            <p>Automated multi-factor risk assessment combining hard business rules with Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Single Invoice Analysis", "Batch CSV Upload"])

    # ---- Single prediction ------------------------------------------------
    with tab1:
        st.markdown("<div class='section-header'>Invoice & PO Details</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#2563eb;margin-bottom:0.5rem;'>INVOICE DATA</div>", unsafe_allow_html=True)
            inv_qty = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=6,
                step=1,
                help="Quantity of items billed on the vendor invoice.",
            )
            inv_dollars = st.number_input(
                "Invoice Dollar Amount ($)",
                min_value=0.01,
                value=214.26,
                step=50.0,
                format="%.2f",
                help="Total dollar amount billed on the vendor invoice.",
            )
            freight = st.number_input(
                "Freight Charge ($)",
                min_value=0.0,
                value=3.47,
                step=5.0,
                format="%.2f",
                help="Freight amount billed on the vendor invoice.",
            )

        with c2:
            st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#2563eb;margin-bottom:0.5rem;'>PO PURCHASE RECORD</div>", unsafe_allow_html=True)
            total_item_qty = st.number_input(
                "Total Item Quantity (PO Level)",
                min_value=1,
                value=6,
                step=1,
                help="Total item quantity recorded on the purchase order.",
            )
            total_item_dollars = st.number_input(
                "Total Item Dollars (PO Level) ($)",
                min_value=0.01,
                value=214.26,
                step=50.0,
                format="%.2f",
                help="Total expected item dollars recorded on the purchase order.",
            )
            po_delay = st.number_input(
                "Days PO to Invoice",
                min_value=0,
                value=2,
                step=1,
                help="Days elapsed between PO placement and invoice issue.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        classify_btn = st.button("Classify Invoice", key="flag_single")

        if classify_btn:
            dollar_diff = abs(inv_dollars - total_item_dollars)
            qty_diff = abs(inv_qty - total_item_qty)

            with st.spinner("Analyzing risk indicators..."):
                result = predict_invoice_flag(
                    invoice_quantity=inv_qty,
                    invoice_dollars=inv_dollars,
                    Freight=freight,
                    total_item_quantity=total_item_qty,
                    total_item_dollars=total_item_dollars,
                    days_po_to_invoice=po_delay,
                    dollar_tolerance=dollar_tolerance,
                    prob_cutoff=prob_cutoff,
                )
                flag_val = int(result["flag_invoice"].iloc[0])
                prob_val = float(result["flag_probability"].iloc[0])
                risk_reasons = str(result["risk_reasons"].iloc[0])

            st.markdown("<div class='section-header'>Classification Result & Risk Breakdown</div>", unsafe_allow_html=True)

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

            # Audit Factor Reason Alert
            if flag_val == 1:
                st.markdown(
                    f"<div style='background:#fef2f2;border-left:4px solid #ef4444;"
                    f"border-radius:4px;padding:0.9rem 1rem;margin-top:1rem;"
                    f"font-size:0.88rem;color:#991b1b;'>"
                    f"<strong>Action Required:</strong> This invoice has been flagged for manual review.<br>"
                    f"<strong>Triggered Risk Reasons:</strong> {risk_reasons}"
                    f"</div>",
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
                df_input = pd.read_csv(uploaded, on_bad_lines="skip")
                required_cols = [
                    "invoice_quantity", "invoice_dollars", "Freight",
                    "total_item_quantity", "total_item_dollars",
                ]
                missing = [c for c in required_cols if c not in df_input.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    dpi_list = df_input["days_po_to_invoice"].tolist() if "days_po_to_invoice" in df_input.columns else 0.0

                    with st.spinner("Running batch classification..."):
                        results = predict_invoice_flag(
                            invoice_quantity=df_input["invoice_quantity"].tolist(),
                            invoice_dollars=df_input["invoice_dollars"].tolist(),
                            Freight=df_input["Freight"].tolist(),
                            total_item_quantity=df_input["total_item_quantity"].tolist(),
                            total_item_dollars=df_input["total_item_dollars"].tolist(),
                            days_po_to_invoice=dpi_list,
                            dollar_tolerance=dollar_tolerance,
                            prob_cutoff=prob_cutoff,
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

                    display_results = results.rename(columns={
                        "invoice_quantity":    "Qty",
                        "invoice_dollars":     "Invoice ($)",
                        "Freight":             "Freight ($)",
                        "total_item_quantity": "PO Qty",
                        "total_item_dollars":  "PO Dollars ($)",
                        "flag_invoice":        "Flag",
                        "flag_probability":    "Probability",
                        "risk_reasons":        "Risk Trigger",
                        "decision":            "Decision",
                    })

                    # Highlight flagged rows with backwards-compatible Styler method
                    styler = display_results.style
                    map_method = getattr(styler, "map", getattr(styler, "applymap", None))
                    styled_results = map_method(
                        lambda v: "background-color:#fee2e2;color:#991b1b;"
                        if v == 1 else "background-color:#dcfce7;color:#166534;",
                        subset=["Flag"],
                    )

                    st.dataframe(
                        styled_results,
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_out = display_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Audit Report CSV",
                        data=csv_out,
                        file_name="invoice_audit_results.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
