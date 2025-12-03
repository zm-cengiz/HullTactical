# BTC Crypto Extension: LSTM-Based Trading Strategy

## Project Description

This project implements a cross-market cryptocurrency trading strategy extension using an LSTM (Long Short-Term Memory) neural network to predict daily Bitcoin (BTC-USD) returns. The model generates allocation weights that are constrained by volatility limits relative to a buy-and-hold benchmark.

### Key Features

- **Data Source**: Yahoo Finance via `yfinance` library
- **Target Asset**: Bitcoin (BTC-USD), daily frequency
- **Model**: PyTorch LSTM for time series prediction
- **Strategy**: Dynamic allocation weights in [0, 2] based on predicted returns
- **Risk Management**: Volatility constraint (strategy vol ≤ 1.2× benchmark vol)

## Project Structure

```
extension/
├── data/
│   ├── btc_raw.csv          # Raw data from Yahoo Finance
│   ├── btc_daily.csv        # Cleaned daily data with log returns
│   └── btc_features.csv     # Feature-engineered dataset
├── notebooks/
│   └── 01_btc_crypto_extension.ipynb   # Main analysis notebook
├── reports/
│   ├── dataset_card.md      # Dataset documentation
│   ├── appendix_crypto_extension.md    # Model & strategy appendix
│   ├── cum_returns.png      # Cumulative returns plot
│   └── weights.png          # Strategy weights plot
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Installation

### 1. Create Virtual Environment

```bash
# Navigate to the extension directory
cd extension

# Create a virtual environment
python -m venv btc_env

# Activate the environment
# On macOS/Linux:
source btc_env/bin/activate
# On Windows:
# btc_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- pandas >= 1.5.0
- numpy >= 1.21.0
- yfinance >= 0.2.0
- torch >= 2.0.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- jupyter >= 1.0.0

## Usage

### Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   Navigate to `notebooks/01_btc_crypto_extension.ipynb`

3. **Run cells in order**:
   Execute each cell sequentially from top to bottom. The notebook is organized into the following sections:

   - **Step 1**: Imports and setup
   - **Step 2**: Data download and preprocessing
   - **Step 3**: Feature engineering
   - **Step 4**: Train/val/test split and normalization
   - **Step 5**: LSTM model definition and training
   - **Step 6**: Backtesting and performance evaluation
   - **Steps 7-10**: Documentation generation

### Expected Outputs

- **Data files**: `data/btc_raw.csv`, `data/btc_daily.csv`, `data/btc_features.csv`
- **Plots**: `reports/cum_returns.png`, `reports/weights.png`
- **Metrics**: Sharpe ratio, cumulative returns, volatility ratio, max drawdown

## Model Overview

### Architecture
- **Type**: LSTM (Long Short-Term Memory)
- **Input**: 20-day sequences of 14 features
- **Hidden Size**: 64
- **Output**: Scalar (predicted next-day return)

### Features Used
- Lagged returns (1-10 days)
- 7-day rolling volatility
- 7-day rolling mean return
- 7-day moving average of close price
- Log volume

### Strategy Logic
1. Predict next-day return using LSTM
2. Map prediction to allocation weight:
   - pred ≤ 0 → weight = 0 (no position)
   - 0 < pred ≤ 1% → weight = 1 (full position)
   - pred > 1% → weight = 2 (leveraged position)
3. Apply volatility constraint if needed
4. Calculate strategy returns

## Results

Performance metrics are computed on the test set (last 15% of data) and compared against a buy-and-hold BTC benchmark. Key metrics include:

- Total Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Volatility Ratio

## Disclaimer

This project is for **research and educational purposes only**. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. This is not financial advice.

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- Yahoo Finance for providing historical price data
- `yfinance` library for easy data access
- PyTorch team for the deep learning framework
