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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 16px;
        color: #1e293b;
        -webkit-font-smoothing: antialiased;
    }

    p, span, div {
        line-height: 1.6;
    }

    /* ---- Sidebar Typography & Styling ---- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1544g2n {
        padding-top: 1.8rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem !important;
    }

    /* ---- Main Container ---- */
    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
        max-width: 1120px;
    }

    /* ---- Page title banner ---- */
    .page-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    }
    .page-banner h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin: 0 0 0.5rem 0;
    }
    .page-banner p {
        color: #cbd5e1;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
        margin: 0;
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1.6rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .metric-card .label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }
    .metric-card .value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-card .sub {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 0.35rem;
    }

    /* ---- Result badges ---- */
    .badge-approve {
        display: inline-block;
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.45rem 1.1rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .badge-flag {
        display: inline-block;
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 0.45rem 1.1rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.015em;
        border-left: 4px solid #2563eb;
        padding-left: 0.85rem;
        margin: 1.8rem 0 1.1rem 0;
        line-height: 1.3;
    }

    /* ---- Streamlit Main Container Theme ---- */
    .main, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* ---- Tabs Typography & High Contrast Visibility ---- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8fafc !important;
        border-radius: 8px !important;
        padding: 0.3rem 0.4rem !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.3rem !important;
        border-radius: 6px !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] button * {
        color: #334155 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.06) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] * {
        color: #2563eb !important;
        font-weight: 800 !important;
    }

    /* ---- Form Inputs & Label Visibility Fix ---- */
    label,
    label p,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    div[class*="stNumberInput"] label *,
    div[class*="stTextInput"] label *,
    div[class*="stTextArea"] label *,
    div[class*="stSelectbox"] label *,
    div[class*="stFileUploader"] label * {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        opacity: 1 !important;
        visibility: visible !important;
        margin-bottom: 0.4rem !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: #f8fafc !important;
    }

    div[data-baseweb="input"], 
    div[data-baseweb="base-input"],
    input[type="number"], 
    textarea {
        background-color: #ffffff !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-baseweb="input"]:focus-within,
    textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    }

    /* ---- Highlight Boxes & Callouts ---- */
    .highlight-box {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .highlight-box strong {
        color: #0f172a;
        font-weight: 700;
    }

    .highlight-warning {
        background: #fffbeb;
        border: 1.5px solid #fde68a;
        border-left: 5px solid #d97706;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #92400e;
        font-weight: 600;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    .highlight-success {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #166534;
        font-weight: 600;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    .highlight-danger {
        background: #fef2f2;
        border: 1.5px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #991b1b;
        font-weight: 600;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: #1e3a5f;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.8rem;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        width: 100%;
        box-shadow: 0 2px 5px rgba(30, 58, 95, 0.2);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #2563eb;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Hide Streamlit default header elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ---- Professional Sidebar Styling & Navigation ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .sidebar-brand-card,
    [data-testid="stSidebar"] .sidebar-status-card {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Fix Streamlit Icon Font for Sidebar Collapse Button */
    [data-testid="stSidebarHeader"] *,
    [data-testid="stSidebarCollapseButton"] *,
    button[aria-label*="sidebar"] *,
    span[data-testid="stHeaderIcon"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    /* Sidebar Navigation Radio Styling */
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        margin-bottom: 0.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: #2563eb !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Sidebar Card Containers */
    .sidebar-brand-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.2rem 1.1rem;
        margin-bottom: 1.5rem;
    }
    .sidebar-brand-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .sidebar-status-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.9rem;
        margin-top: 1.2rem;
        font-size: 0.8rem;
    }
    .sidebar-status-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .sidebar-status-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        color: #e2e8f0;
    }
    .sidebar-status-item span {
        color: #38bdf8;
        font-weight: 600;
    }

    /* ---- Mobile Responsiveness & Scaled Typography ---- */
    @media screen and (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 15px;
        }
        .main .block-container {
            padding: 1rem 0.6rem 2.2rem 0.6rem !important;
        }
        .page-banner {
            padding: 1.4rem 1.2rem !important;
            margin-bottom: 1.4rem !important;
            border-radius: 10px !important;
        }
        .page-banner h1 {
            font-size: 1.55rem !important;
            line-height: 1.3 !important;
        }
        .page-banner p {
            font-size: 0.92rem !important;
        }
        .stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.75rem !important;
        }
        .metric-card {
            padding: 1rem 0.85rem !important;
            border-radius: 10px !important;
        }
        .metric-card .value {
            font-size: 1.6rem !important;
        }
        .metric-card .label {
            font-size: 0.72rem !important;
        }
        .section-header {
            font-size: 1.1rem !important;
            margin: 1.4rem 0 0.8rem 0 !important;
        }
        .stButton > button {
            padding: 0.65rem 1.4rem !important;
            font-size: 0.92rem !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 0.88rem !important;
            padding: 0.5rem 0.75rem !important;
        }
    }

    @media screen and (max-width: 480px) {
        .stats-grid {
            grid-template-columns: 1fr !important;
        }
        .page-banner h1 {
            font-size: 1.4rem !important;
        }
        .metric-card .value {
            font-size: 1.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation callback function
def set_page(target_page: str):
    st.session_state["nav_page"] = target_page

# Initialize Session State Navigation
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
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

    # Overview Quick Action Navigation Section
    st.markdown("<div class='section-header'>Quick Actions — Try ML Features</div>", unsafe_allow_html=True)
    act_col1, act_col2 = st.columns(2, gap="large")
    with act_col1:
        st.button(
            "Launch Freight Cost Predictor →",
            key="launch_freight_btn",
            on_click=set_page,
            args=("Freight Cost Prediction",),
        )
    with act_col2:
        st.button(
            "Launch Invoice Risk Classifier →",
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
                df_input = pd.read_csv(uploaded, on_bad_lines="skip")
                required_cols = [
                    "invoice_quantity", "invoice_dollars", "Freight",
                    "total_item_quantity", "total_item_dollars",
                ]
                missing = [c for c in required_cols if c not in df_input.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
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
