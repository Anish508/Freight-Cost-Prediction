# Procurement Analytics System — System Enhancements & Audit Report

## Executive Summary

This document details the architectural refactoring, machine learning model enhancements, enterprise audit features, and verification test suite implemented in the **Procurement Analytics & Risk Management System**.

---

## 1. Machine Learning Model Refactoring & Confounding Fix

### **The Problem Identified**
In the original model pipeline, the binary target label `flag_invoice` was generated using two rules:
1. `abs(invoice_dollars - total_item_dollars) > 5`
2. `avg_receiving_delay > 10`

However, `avg_receiving_delay` was **excluded** from the training feature set. This created **Latent Label Noise**: 515 historical invoices in `inventory.db` had a perfect $0 dollar discrepancy and 0 quantity discrepancy, but were assigned `flag_invoice = 1` solely due to delivery delays.

Because `avg_receiving_delay` was missing from the feature matrix, the unrefactored Random Forest model attempted to memorize unobservable patterns from raw dollar amounts, predicting **`FLAGGED` (68% probability)** for a perfect 0-discrepancy invoice.

### **The Fix Applied**
1. **Delta & Ratio Feature Engineering:** Added explicit relative features:
   - `dollar_discrepancy` = `abs(invoice_dollars - total_item_dollars)`
   - `qty_discrepancy` = `abs(invoice_quantity - total_item_quantity)`
   - `freight_ratio` = `Freight / (invoice_dollars + 1e-5)`
   - `dollar_diff_ratio` = `dollar_discrepancy / (total_item_dollars + 1e-5)`
2. **Clean Risk Label Generation:** Separated unobserved delivery delay noise from real-time invoice risk factors (`dollar_discrepancy > 5`, `qty_discrepancy > 0`, `days_po_to_invoice > 15`).
3. **Calibrated Model Training:** Re-trained and serialized `invoice_flagging_model.pkl` and `invoice_flagging_scaler.pkl`.

---

## 2. Enterprise Real-World Features Added

### **A. Freight Overcharge Baseline Variance Analysis**
- **Inference Module:** [`Inference/Predict_Freight.py`](file:///d:/PDS%20Project/Inference/Predict_Freight.py)
- **Capability:** Compares actual billed shipping charges against the Random Forest ML predicted baseline.
- **Metrics Calculated:**
  - `predicted_freight` ($) — Expected freight baseline cost
  - `actual_freight` ($) — Billed shipping cost
  - `freight_variance` ($) — Dollar difference (`actual - predicted`)
  - `variance_pct` (%) — Percentage variance over baseline
  - `freight_status` — `NORMAL` | `OVERCHARGED`

### **B. Multi-Factor Risk Trigger Tags**
- **Inference Module:** [`Inference/Predict_Invoice_Flag.py`](file:///d:/PDS%20Project/Inference/Predict_Invoice_Flag.py)
- **Capability:** Audits invoices against multiple risk criteria and tags exact audit failure reasons:
  - `Price Discrepancy ($650.00 > $5.00)`
  - `Quantity Mismatch (5 units)`
  - `PO Delay (28 days)`
  - `ML Anomaly Score (82.0% >= 50.0%)`

### **C. Sidebar Enterprise Risk Controls**
- **User Interface:** [`app.py`](file:///d:/PDS%20Project/app.py)
- **Capability:** Collapsible **`⚙️ Enterprise Audit Controls`** panel in the sidebar allows finance auditors to adjust:
  - **Max Dollar Discrepancy Tolerance ($):** `$1.00` to `$50.00` (default `$5.00`)
  - **ML Anomaly Sensitivity Threshold (%):** `30%` to `90%` (default `50%`)

---

## 3. Automated Test Verification Results

All inference pipelines were executed and validated against automated test suites:

```
======================================================================
          AUTOMATED TEST SUITE & VERIFICATION REPORT
======================================================================

[TEST 1] Freight Cost Baseline & Overcharge Variance Audit
 invoice_dollars  predicted_freight  actual_freight  freight_variance  variance_pct freight_status
          214.26               6.08            3.47             -2.61         -42.9         NORMAL
         1850.00              14.29          120.00            105.71         739.7    OVERCHARGED
       137483.78             694.51         2935.20           2240.69         322.6    OVERCHARGED

[TEST 2] Invoice Risk Flagging Multi-Factor Audit
 invoice_dollars  total_item_dollars  dollar_discrepancy  qty_discrepancy  flag_invoice  flag_probability                         decision                          risk_reasons
          214.26              214.26                 0.0                0             0               0.0           [APPROVE] Auto-process         None (All Audit Rules Passed)
          140.55              140.55                 0.0                0             0               0.0           [APPROVE] Auto-process         None (All Audit Rules Passed)
         5250.00             5250.00                 0.0                0             0               0.0           [APPROVE] Auto-process         None (All Audit Rules Passed)
         1850.00             1200.00               650.0                0             1               1.0 [FLAG]    Manual Review Required   Price Discrepancy ($650.00 > $5.00)
          890.00              890.00                 0.0                5             1               1.0 [FLAG]    Manual Review Required           Quantity Mismatch (5 units)
         3400.00             3400.00                 0.0                0             1               1.0 [FLAG]    Manual Review Required                    PO Delay (28 days)
        15400.00            12100.00              3300.0                0             1               1.0 [FLAG]    Manual Review Required Price Discrepancy ($3,300.00 > $5.00)
          420.00              420.00                 0.0                0             0               0.0           [APPROVE] Auto-process         None (All Audit Rules Passed)

======================================================================
              ALL TESTS COMPLETED SUCCESSFULLY (100% PASS)
======================================================================
```

---

## 4. Test Data Guide for Manual & Batch Evaluation

### **Sample CSV Dataset (`test_invoices_sample.csv`)**

File path: [`test_invoices_sample.csv`](file:///d:/PDS%20Project/test_invoices_sample.csv)

```csv
invoice_quantity,invoice_dollars,Freight,total_item_quantity,total_item_dollars,days_po_to_invoice,notes
6,214.26,3.47,6,214.26,2.0,Exact Match - Safe Invoice (Auto-Approve)
15,140.55,8.57,15,140.55,4.0,Exact Match - Normal Freight (Auto-Approve)
100,5250.00,120.00,100,5250.00,3.0,Bulk Order - Safe (Auto-Approve)
25,1850.00,45.00,25,1200.00,5.0,Dollar Discrepancy Overbilled by $650 (FLAGGED)
10,890.00,15.00,15,890.00,4.0,Quantity Mismatch Billed 10 vs PO 15 (FLAGGED)
50,3400.00,450.00,50,3400.00,28.0,High PO to Invoice Delay 28 Days (FLAGGED)
200,15400.00,850.00,200,12100.00,1.0,Major Dollar Discrepancy Overbilled by $3300 (FLAGGED)
8,420.00,9.50,8,420.00,1.0,Small Order - Perfect Match (Auto-Approve)
```

---

## 5. UI Compatibility Improvements

- **Pandas Styler Compatibility:** Fixed `Styler.applymap()` deprecation error by replacing with dynamic `map` method fallback.
- **Robust CSV Parser:** Added `on_bad_lines="skip"` to prevent dashboard crashes on malformed user uploads.
- **Mobile Responsive & High Contrast UI:** Applied CSS rules for light mode theme, dark slate input labels, active blue tab headers, and responsive metric grids.
