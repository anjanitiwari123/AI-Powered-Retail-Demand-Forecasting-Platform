# 🚀 AI-Powered Retail Demand Forecasting Platform

An end-to-end **AI-powered Retail Demand Forecasting Platform** built using Machine Learning, Deep Learning, Explainable AI, and Streamlit.

This platform predicts future store-level sales using historical demand patterns, promotions, calendar effects, store characteristics, and competition information.

🌐 **Live Demo:**  
https://ai-powered-retail-demand-forecasting-platform-at.streamlit.app/

---

# 📌 Project Overview

Retail demand forecasting is a critical business problem used for:

- Inventory optimization
- Supply chain planning
- Promotion strategy
- Store operations
- Revenue forecasting

This project builds a complete forecasting pipeline that predicts daily sales for individual stores using advanced feature engineering and multiple ML/DL models.

The deployed Streamlit application allows users to:

✅ Select stores  
✅ Generate future demand forecasts  
✅ Perform promotion scenario analysis  
✅ Perform holiday scenario analysis  
✅ Visualize future sales trends  
✅ Understand model predictions using SHAP explainability  

---

# 🏗️ Project Architecture

```
Raw Retail Data
        |
        ↓
Data Cleaning & Validation
        |
        ↓
Exploratory Data Analysis
        |
        ↓
Feature Engineering
        |
        ↓
Time-Series Validation
        |
        ↓
Model Training
        |
        ├── XGBoost
        ├── LightGBM
        ├── LSTM
        ├── GRU
        └── Transformer
        |
        ↓
Model Evaluation
        |
        ↓
SHAP Explainability
        |
        ↓
Model Serialization
        |
        ↓
Streamlit Deployment
```

---

# 📊 Dataset

Dataset Used:

**Rossmann Store Sales Dataset**

The dataset contains daily sales information from multiple retail stores.

## Sales Dataset Features

| Feature | Description |
|---|---|
| Store | Store identifier |
| Date | Sales date |
| Sales | Target variable |
| Customers | Customer count |
| Open | Store operating status |
| Promo | Promotion indicator |
| StateHoliday | State holiday information |
| SchoolHoliday | School holiday indicator |

## Store Dataset Features

| Feature | Description |
|---|---|
| StoreType | Store category |
| Assortment | Product assortment type |
| CompetitionDistance | Distance from competitor |
| CompetitionOpenSince | Competitor availability |
| Promo2 | Long-term promotion indicator |

---

# 🔍 Exploratory Data Analysis

Performed detailed EDA including:

- Sales distribution analysis
- Sales trend analysis
- Store-wise performance analysis
- Promotion impact analysis
- Holiday impact analysis
- Correlation analysis
- Feature relationship analysis

---

# 📈 Skewness and Outlier Handling

Sales data showed a naturally **right-skewed distribution**.

High sales values were observed because of:

- Promotional campaigns
- Seasonal demand
- Holiday periods
- High-performing stores

## Outlier Strategy

Outliers were **not removed**.

Reason:

> In retail forecasting, extreme sales values often represent genuine business events rather than incorrect data.

Removing these values would reduce the model's ability to learn demand spikes.

Instead, the model learns these patterns using:

- Tree-based algorithms
- Lag features
- Rolling statistics
- Calendar features

---

# 🧹 Data Cleaning

The following preprocessing steps were applied:

## Closed Store Removal

Rows where:

```
Open = 0
```

were removed because closed stores do not represent actual demand.

---

## Duplicate Removal

Removed exact duplicate records.

---

## Missing Value Treatment

| Feature | Treatment |
|---|---|
| CompetitionDistance | Filled using maximum distance |
| CompetitionOpenSinceMonth | Filled with 0 |
| CompetitionOpenSinceYear | Filled with 0 |
| Promo2SinceWeek | Filled with 0 |
| Promo2SinceYear | Filled with 0 |
| PromoInterval | Filled with "None" |

---

# 🚫 Data Leakage Prevention

The `Customers` feature was removed before model training.

Reason:

```
Customers → Sales
```

Customer count is highly correlated with sales but is unavailable during future forecasting.

Using it would create unrealistic predictions.

The final model only uses features available before the forecast date.

---

# ⚙️ Feature Engineering

Created business-driven forecasting features.

## Calendar Features

Generated:

- Year
- Month
- Day
- Week of Year
- Day of Week
- Weekend indicator
- Month start indicator
- Month end indicator

---

## Cyclic Encoding

Converted periodic features:

```
Month
    ↓
Month_sin
Month_cos


DayOfWeek
    ↓
DayOfWeek_sin
DayOfWeek_cos
```

This helps models capture seasonal patterns.

---

## Lag Features

Created historical demand features:

```
Sales_lag_1
Sales_lag_7
Sales_lag_14
Sales_lag_30
```

---

## Rolling Features

Generated:

```
Rolling Mean
Rolling Standard Deviation
Exponentially Weighted Mean
```

to capture recent demand behaviour.

---

## Business Features

Created:

- Competition age
- Promo duration
- Active promotion month
- Store characteristics

---

# 🤖 Machine Learning Models

Multiple models were trained and compared.

---

# XGBoost

Gradient boosting model used for baseline forecasting.

Advantages:

- Handles nonlinear relationships
- Strong tabular performance
- Robust feature learning

---

# LightGBM

Selected production model because of:

- High forecasting performance
- Fast training
- Efficient inference
- Excellent tabular data handling

---

# Deep Learning Models

Implemented:

## LSTM

Captures long-term sequential patterns.

## GRU

Efficient recurrent architecture.

## Transformer Encoder

Attention-based sequence forecasting model.

---

# 📊 Model Evaluation

Evaluation metrics used:

## RMSPE

Primary retail forecasting metric.

Formula:

```
RMSPE =
√ Mean((Actual - Prediction) / Actual)²
```

Additional metrics:

- RMSE
- MAE
- R² Score

---

# 🧠 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was implemented to interpret model predictions.

It helps answer:

- Which features influence predictions?
- Why did demand increase or decrease?
- What business factors drive sales?

Major influencing factors include:

- Historical sales
- Promotions
- Calendar patterns
- Store characteristics

---

# 🖥️ Streamlit Dashboard

The deployed dashboard provides an interactive forecasting interface.

## Forecast Controls

Users can select:

- Store
- Forecast horizon
- Promotion status
- School holiday scenario
- State holiday scenario

---

## Forecast Output

Dashboard displays:

- Expected total sales
- Average daily sales
- Peak sales day
- Forecast trend graph
- Daily forecast table

---

## Model Insights

Includes:

- Model comparison results
- SHAP feature importance visualization

---

# 📂 Project Structure

```
AI-Powered-Retail-Demand-Forecasting-Platform/

│
├── app.py
├── demand.ipynb
├── requirements.txt
│
├── data/
│   ├── train.csv
│   └── store.csv
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── transformer_model.keras
│   ├── feature_list.pkl
│   └── SHAP files
│
└── README.md
```

---

# ⚡ Installation

Clone repository:

```bash
git clone https://github.com/anjanitiwari123/AI-Powered-Retail-Demand-Forecasting-Platform.git
```

Move into project:

```bash
cd AI-Powered-Retail-Demand-Forecasting-Platform
```

Create environment:

```bash
python -m venv venv
```

Activate environment:

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# 🛠️ Tech Stack

## Programming

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- LightGBM
- XGBoost
- Scikit-learn

## Deep Learning

- TensorFlow
- Keras
-LSTM
-GRU
-Transformer
## Explainability

- SHAP

## Visualization

- Plotly
- Matplotlib

## Deployment

- Streamlit Cloud

---

# 🎯 Business Impact

This platform helps retailers:

✅ Forecast future demand  
✅ Improve inventory planning  
✅ Reduce stockouts and overstock  
✅ Optimize promotions  
✅ Make data-driven decisions  

---

# 🚀 Future Improvements

Possible enhancements:

- Automatic holiday calendar integration
- Real-time sales pipeline
- Model monitoring dashboard
- Demand uncertainty prediction
- REST API deployment
- Cloud-based ML architecture

---

# 👨‍💻 Author

## Anjani Tiwari

GitHub:

https://github.com/anjanitiwari123
