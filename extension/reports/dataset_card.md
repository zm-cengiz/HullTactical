# Dataset Card: BTC-USD Daily Price and Returns

## Dataset Overview

This dataset contains daily price and return data for Bitcoin (BTC-USD), collected from Yahoo Finance using the `yfinance` Python library. The data is used for training machine learning models to predict next-day returns and develop trading strategies.

## Data Source

- **Provider**: Yahoo Finance
- **Access Method**: `yfinance` Python library
- **Ticker Symbol**: BTC-USD
- **Frequency**: Daily (1d interval)
- **Date Range**: January 1, 2015 to present

## Data Collection Process

1. **Raw Download**: Data retrieved using `yf.download("BTC-USD", start="2015-01-01", end=<today>, interval="1d")`
2. **Saved to**: `data/btc_raw.csv`

## Dataset Files

### 1. btc_raw.csv
Raw data directly from Yahoo Finance download.

### 2. btc_daily.csv (Cleaned Dataset)

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date (normalized, no time component) |
| open | float | Opening price (USD) |
| high | float | Highest price during the day (USD) |
| low | float | Lowest price during the day (USD) |
| close | float | Closing price (USD) |
| adj_close | float | Adjusted closing price (USD) |
| volume | float | Trading volume |
| ret | float | Daily log return: log(close_t / close_{t-1}) |

**Preprocessing Steps:**
- Reset index to make Date a column
- Renamed columns to lowercase (Date→date, Open→open, etc.)
- Sorted by date ascending
- Removed duplicate dates
- Computed daily log returns
- Dropped rows with NaN in returns (first row)

### 3. btc_features.csv (Feature Dataset)

Contains all columns from btc_daily.csv plus engineered features:

| Feature | Description |
|---------|-------------|
| ret_lag_1 to ret_lag_10 | Lagged returns (shifted by 1-10 days) |
| vol_7 | 7-day rolling standard deviation of returns |
| mean_ret_7 | 7-day rolling mean of returns |
| ma_close_7 | 7-day rolling mean of closing price |
| log_volume | log(1 + volume) |

**Note:** Initial rows with NaN from rolling calculations are dropped.

## Data Statistics

- **Total Trading Days**: ~3,600+ (varies based on download date)
- **Return Characteristics**: High volatility typical of cryptocurrency markets
- **Missing Values**: None after preprocessing

## Train/Validation/Test Split

The data is split chronologically (no shuffling):
- **Train**: First 70% of data
- **Validation**: Next 15% of data
- **Test**: Final 15% of data

## Usage Notes

### Intended Use
- Training machine learning models for return prediction
- Backtesting trading strategies
- Research and educational purposes

### Limitations
- Historical performance does not guarantee future results
- Cryptocurrency markets are highly volatile and speculative
- Data quality depends on Yahoo Finance accuracy

## Terms of Service

This data is obtained via Yahoo Finance. Users should comply with:
- Yahoo Finance Terms of Service
- `yfinance` library usage guidelines

**Important**: This data is provided for research and educational purposes only. It should not be used as the sole basis for investment decisions.

## Citation

If using this dataset, please acknowledge:
- Yahoo Finance as the data source
- The `yfinance` Python library (https://github.com/ranaroussi/yfinance)

---

*Dataset card generated as part of the BTC Crypto Extension project.*
