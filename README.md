# PDS Project - Procurement Analytics Dashboard

## Project Overview

This project is a machine learning-powered procurement analytics system built to help finance and
operations teams automate two critical tasks:

1. **Freight Cost Prediction** - Estimate the freight charge for a vendor invoice based on its total dollar value.
2. **Invoice Risk Flagging** - Automatically classify vendor invoices as safe to process or flag for manual review.

Both models are served through a professional Streamlit dashboard that supports single and batch predictions.

---

## The Problem This Application Solves

### Problem 1 - Unpredictable Freight Costs

Procurement teams deal with hundreds of vendor invoices every month. Freight charges vary significantly
across vendors, order sizes, and product types. Without a reliable way to estimate expected freight costs,
finance teams have no baseline to detect overcharges or billing errors.

**Solution:** A trained regression model predicts the expected freight cost given the invoice dollar amount.
Any invoice where the actual freight significantly exceeds the prediction can be flagged for investigation.

### Problem 2 - Manual Invoice Review at Scale

Invoice review is time-consuming. Finance teams cannot manually verify every invoice for discrepancies in
dollar amounts, quantities, or delivery timelines. Skipping review creates financial risk; reviewing everything
wastes analyst time.

**Solution:** A trained binary classification model automatically evaluates each invoice against historical
patterns and flags only the highest-risk ones for manual review.

---

## Data Source

All data is stored in a SQLite database (inventory.db) with the following key tables:

| Table | Description |
|---|---|
| vendor_invoice | One row per vendor invoice - contains quantity, dollar amount, freight, PO number, payment dates |
| purchases | Line-item purchase records - brand, quantity, price, PO number, receiving dates |
| purchase_prices | Reference prices per brand and product |
| begin_inventory | Opening inventory snapshot (January 2024) |
| end_inventory | Closing inventory snapshot (December 2024) |

The dataset covers **5,543 vendor invoices** from approximately Jan-Dec 2024.

---

## Module 1 - Freight Cost Prediction

### What It Predicts
Given the total dollar value of a vendor invoice, the model predicts the expected **freight charge in dollars ($)**.

### Feature Used

| Feature | Description |
|---|---|
| Dollars | Total invoice dollar amount (from vendor_invoice table) |

### Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline parametric model |
| Decision Tree Regressor | Non-linear, depth-limited to 5 |
| **Random Forest Regressor** | Ensemble of 100 trees, depth-limited to 6 - **saved model** |

### Evaluation Metrics

Regression models are evaluated using:
- **MAE (Mean Absolute Error)** - Average dollar error in predictions
- **RMSE (Root Mean Squared Error)** - Penalises large errors more heavily
- **R-squared** - Proportion of variance explained (higher is better)

The **Random Forest Regressor** achieved the lowest MAE and was selected as the production model.

### Saved Model

`
Freight_cost_prediction/models/predict_freight_model.pkl
`

---

## Module 2 - Invoice Risk Flagging

### What It Predicts
Given five invoice features, the model predicts:
- **flag_invoice** - Binary: 1 = Flag for manual review, 0 = Auto-approve
- **flag_probability** - Confidence score (0.0 to 1.0) that the invoice should be flagged

### SQL Feature Engineering

The dataset was created by joining vendor_invoice with an aggregated CTE from purchases:

`sql
WITH purchase_agg AS (
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand)       AS total_brands,
        SUM(p.Quantity)               AS total_item_quantity,
        SUM(p.Dollars)                AS total_item_dollars,
        AVG(julianday(p.ReceivingDate)
            - julianday(p.PODate))    AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
)
SELECT
    vi.PONumber,
    vi.Quantity                       AS invoice_quantity,
    vi.Dollars                        AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate))      AS days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate))     AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
FROM vendor_invoice vi
LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber;
`

### Risk Label Definition

| Condition | Label |
|---|---|
| abs(invoice_dollars - total_item_dollars) >  | 1 - Flag |
| avg_receiving_delay > 10 days | 1 - Flag |
| Neither condition met | 0 - Approve |

Label distribution across 5,543 invoices:
- **3,693 (66.7%)** - Auto-approve (label = 0)
- **1,850 (33.3%)** - Flagged for review (label = 1)

### Statistical Feature Selection (Welch t-test)

| Feature | Significant (p < 0.05)? |
|---|---|
| invoice_quantity | Yes |
| invoice_dollars | Yes |
| Freight | Yes |
| days_po_to_invoice | Yes |
| total_item_quantity | Yes |
| total_item_dollars | Yes |
| avg_receiving_delay | Yes |
| days_to_pay | No |
| total_brands | No |

### Final Features (Production Model)

| Feature | Description |
|---|---|
| invoice_quantity | Number of items on the invoice |
| invoice_dollars | Total dollar amount on the invoice |
| Freight | Freight/shipping charge |
| total_item_quantity | Aggregated item quantity at PO level |
| total_item_dollars | Aggregated dollar total at PO level |

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| total_item_dollars | 27.1% |
| total_item_quantity | 19.9% |
| invoice_dollars | 18.9% |
| Freight | 17.8% |
| invoice_quantity | 16.1% |

### Model Comparison

| Model | Accuracy | F1 Score (Flagged) | Notes |
|---|---|---|---|
| Logistic Regression | 65% | 0.01 | Failed - data not linearly separable |
| Decision Tree Classifier | 83% | 0.76 | Balanced but moderate |
| **Random Forest Classifier** | **89%** | **0.81** | Best model - saved |

### Detailed Classification Report (Random Forest)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 - Approve | 0.87 | 0.98 | 0.92 | 725 |
| 1 - Flag | 0.94 | 0.72 | 0.81 | 384 |
| **Overall Accuracy** | | | **89%** | 1,109 |

**Confusion Matrix:**

`
                    Predicted: Approve  Predicted: Flag
Actual: Approve            707               18
Actual: Flag               109              275
`

> Interpretation: The model has very high precision (94%) on flagged invoices.
> When it raises a flag, it is almost always correct. Recall of 72% means
> roughly 28% of risky invoices are missed - a deliberate trade-off to avoid
> overwhelming analysts with false positives.

### Preprocessing
- **Scaler:** MinMaxScaler - transforms all features to [0, 1] before classification
- **Train / Test Split:** 80% / 20% (random_state = 42)

### Saved Artifacts

`
Invoice_flagging/models/invoice_flagging_model.pkl
Invoice_flagging/models/invoice_flagging_scaler.pkl
`

---

## Inference Layer

### Predict_Freight.py

`python
from Inference.Predict_Freight import predict_freight

# Single prediction
result = predict_freight(5000.00)

# Batch prediction
result = predict_freight([214.26, 5000.00, 137483.78])
`

Returns a DataFrame with columns: invoice_dollars, predicted_freight

### Predict_Invoice_Flag.py

`python
from Inference.Predict_Invoice_Flag import predict_invoice_flag

# Single prediction
result = predict_invoice_flag(
    invoice_quantity=6,
    invoice_dollars=214.26,
    Freight=3.47,
    total_item_quantity=6,
    total_item_dollars=214.26,
)

# Batch prediction - pass equal-length lists
`

Returns a DataFrame with columns: all input features + flag_invoice, flag_probability, decision

---

## Streamlit Application

`ash
streamlit run app.py
`

Opens at: http://localhost:8501

### Pages

| Page | Features |
|---|---|
| Overview | Project summary, module descriptions, model performance table |
| Freight Cost Prediction | Single prediction with metric cards; batch input with CSV download |
| Invoice Risk Flagging | Single invoice with discrepancy warning; batch CSV upload with summary stats and results table |

---

## Project File Structure

`
PDS Project/
|-- app.py                              Streamlit dashboard (entry point)
|-- inventory.db                        Master SQLite database
|-- README.md                           This file
|
|-- Fright Prediction.ipynb             Freight cost EDA and modelling notebook
|-- Risk_Flagging.ipynb                 Invoice flagging EDA and modelling notebook
|
|-- Freight_cost_prediction/
|   |-- data_preprocessing.py
|   |-- model_evaluation.py
|   |-- train.py
|   |-- data/inventory.db
|   |-- models/predict_freight_model.pkl
|
|-- Invoice_flagging/
|   |-- model_preprocessing.py
|   |-- model_evaluation.py
|   |-- train.py
|   |-- data/inventory.db
|   |-- models/
|       |-- invoice_flagging_model.pkl
|       |-- invoice_flagging_scaler.pkl
|
|-- Inference/
    |-- Predict_Freight.py
    |-- Predict_Invoice_Flag.py
`

---

## Technology Stack

| Layer | Library / Tool |
|---|---|
| Data storage | SQLite via sqlite3 |
| Data processing | pandas, numpy |
| Statistical analysis | scipy.stats (Welch t-test) |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Dashboard | streamlit |
| Language | Python 3.12 |

---

## How to Retrain the Models

### Freight Cost Prediction

`ash
cd Freight_cost_prediction
python train.py
`

### Invoice Risk Flagging (baseline)

`ash
cd Invoice_flagging
python train.py
`

### Invoice Risk Flagging (with hyperparameter tuning)

Edit the last line of Invoice_flagging/train.py:

`python
main(tune=True)
`

GridSearchCV evaluates 216 combinations across 5 folds (1,080 total fits).

---

## Key Business Value

| Outcome | Impact |
|---|---|
| Automated freight estimation | Instant baseline for detecting overcharges |
| 89% accurate invoice classification | Dramatically reduces manual review workload |
| 94% precision on flagged invoices | Analysts can trust the flags - very few false positives |
| Batch CSV processing | Hundreds of invoices processed at once |
| Transparent probability scores | Analysts see confidence levels alongside binary decisions |
| Streamlit dashboard | No code required - accessible to non-technical finance staff |