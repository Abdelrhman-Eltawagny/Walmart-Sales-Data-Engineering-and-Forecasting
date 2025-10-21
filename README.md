# Walmart-Sales-Data-Engineering-and-Forecasting
This project is an end-to-end Data Engineering and Machine Learning pipeline built using the Walmart Sales dataset.   It covers the full lifecycle — from data extraction and transformation to data warehousing, analytics, and profit forecasting using ML.  


The goal is to analyze Walmart’s sales data, identify profit patterns, and predict future weekly profits to support data-driven decision-making.

---

## 🧩 Project Workflow

### **1. ETL (Extract, Transform, Load)**
- **Extract:** Reads raw Walmart CSV data using `pandas`.
- **Transform:** 
  - Cleans missing and duplicate records.
  - Converts data types and creates new features like `total_price`, `revenue_per_item`, and `profit`.
  - Builds **star schema tables**:
    - `DimStore` (Branch, City)
    - `DimCategory`
    - `DimPayment`
    - `DimDate`
    - `fact_walmart_sales`
- **Load:** 
  - Loads all tables into an **SQLite database (`walmart.db`)**.
  - Creates and exports:
    - `ml_data_mart.csv` → for machine learning.
    - `analytics_data_mart.csv` → for business analysis.

📁 **Technologies Used:** `Python`, `Pandas`, `SQLite`, `Logging`

---

### **2. Data Analysis**
Exploratory data analysis (EDA) was performed to understand:
- Total sales and profit per **category**.
- Sales trends **by month**.
- **Payment method** distribution (Credit Card, Ewallet, Cash).
- Relationship between **ratings, quantity, and profit**.

📊 Key Insights:
- **Home and Lifestyle** & **Fashion Accessories** categories yield the highest profits.
- **Credit Card** is the most common payment method.
- Sales peak significantly during **December**.
- Higher **customer ratings** often correlate with increased **profit**.

---

### **3. Machine Learning Model**
A forecasting model was built using **XGBoost** to predict **weekly profit**.

#### Steps:
- Resampled data to **weekly frequency**.
- Applied **log transformation** for stable variance.
- Created time-based features: `week`, `month`, `year`.
- Split into train/test sets.
- Tracked the entire ML lifecycle using **MLflow**.

📈 **Model Performance:**
- Model: `XGBRegressor`
- Evaluation metrics:
  - `R² Score (Test)` ≈ 0.92  
  - `RMSE (Test)` ≈ 3,000  
- Forecast visualization shows accurate profit prediction trends.

🧰 **Tools:**  
`Python`, `Pandas`, `NumPy`, `XGBoost`, `scikit-learn`, `MLflow`, `Matplotlib`

---
