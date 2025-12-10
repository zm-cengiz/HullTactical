# Hull Tactical - Market Prediction (Python 3.11.13)

Machine Learning Project - S&P 500 Return Prediction

## Project Overview
This project predicts daily excess returns of the S&P 500 using machine learning and develops a trading strategy with a 120% volatility constraint.

### Prerequisites
- Python Python 3.11.13+
- Git

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dependencies: `pip install -r requirements.txt`

### Data
Download data from [Kaggle Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview)
and put them in the data folder as train.csv and test.csv

## Project Structure
- `notebooks/` - Jupyter notebooks for experimentation
- `data/` - Raw and processed datasets
- `results/` - Predictions and evaluation plots
- `artifacts/` - Built to contain eventual saved models
- `extension/` - Contains the bonus work for the project (extension on the Bitcoin market)


---

## 📊 Notebooks & Modeling Summary

### **EDA & Feature Selection**
Explores:
- distribution of forward and excess returns  
- missing value patterns  
- outlier detection  
- time-varying correlation analysis  
- early baseline models for intuition  

Select features by techniques that include:
- mutual information ranking  
- rolling correlation stability  
- LightGBM importance  
- category-balanced selection (E, S, V, M, I, P groups)

Final result: **20 high-signal market features** (available in 2 versions quota/no-quota).

---

### **Step 1 — Baseline Modeling**
Initial benchmark including:
- simple lag features  
- logistic/linear baselines  
- first evaluation using the custom Kaggle metric  

---

### **Step 2 — Model Development**

This step explores different learning algorithms for forecasting excess returns and building allocation signals:

- experiments with Random Forest, LightGBM, and LSTM
- uses time-series (walk-forward) cross-validation to avoid leakage
- predicts allocation weights between 0 and 2 from return forecasts
- enforces the 120% volatility constraint during mapping and evaluation

---

### **Step 3 and 5 — Feature Engineering & Modeling**
This notebook builds our final and best performing predictive model using:

#### Feature Engineering
- Lags: **1, 3, 5 days**  
- Rolling windows: **5-day & 10-day means and volatilities**  
- Short-term momentum indicators  
- Non-leaky rolling implementation using only past data  

#### Model
- **LightGBM (GBDT)**  
- Time-series cross-validation (walk-forward)  
- Hyperparameter search around best performing config  
- Mapping predictions → portfolio allocations in `[0, 2]`  

---

### **Step 4 — Backtesting & Evaluation**
Includes a deeper evaluation of our best model (step 3/5):
- cumulative excess return curves  
- volatility ratio vs market  
- drawdown analysis  
- comparison with S&P 500 baseline  
- interpretation of volatility penalties in the metric  

---
