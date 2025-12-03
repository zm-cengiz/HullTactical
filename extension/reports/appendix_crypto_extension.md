# Appendix: BTC Crypto Extension - Model and Strategy Details

## 1. Model Architecture

### LSTM Network

The model uses a Long Short-Term Memory (LSTM) neural network, which is well-suited for sequential time series data due to its ability to capture long-term dependencies.

```
Input Layer
    ↓
LSTM Layer (hidden_size=64, num_layers=1)
    ↓
Fully Connected Layer (64 → 1)
    ↓
Output (predicted return)
```

### Why LSTM?

- **Memory**: LSTMs can remember patterns over extended time periods
- **Non-linearity**: Captures complex relationships in financial data
- **Sequence handling**: Naturally processes time-ordered data
- **Gradient flow**: Avoids vanishing gradient problems in long sequences

## 2. Hyperparameters

### Data Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 20 | Days of history used per prediction |
| Train Ratio | 70% | Training data proportion |
| Validation Ratio | 15% | Validation data proportion |
| Test Ratio | 15% | Test data proportion |

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hidden Size | 64 | LSTM hidden state dimension |
| Num Layers | 1 | Stacked LSTM layers |
| Dropout | 0.0 | Dropout rate between layers |

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 64 | Samples per training batch |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Max Epochs | 30 | Maximum training epochs |
| Early Stopping | 5 | Patience (epochs without improvement) |
| Loss Function | MSE | Mean Squared Error |
| Optimizer | Adam | Adaptive moment estimation |

### Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Threshold High | 0.01 (1%) | Predicted return for max allocation |
| Volatility Constraint | 1.2 | Max strategy vol / benchmark vol |
| Weight Range | [0, 2] | Allocation weight bounds |

## 3. Feature Engineering

### Input Features (14 total)

1. **Lagged Returns** (10 features): `ret_lag_1` to `ret_lag_10`
   - Historical returns shifted by 1-10 days
   - Captures momentum and mean reversion patterns

2. **Rolling Volatility** (1 feature): `vol_7`
   - 7-day rolling standard deviation of returns
   - Measures recent market uncertainty

3. **Rolling Mean Return** (1 feature): `mean_ret_7`
   - 7-day rolling average return
   - Captures short-term trend direction

4. **Moving Average Price** (1 feature): `ma_close_7`
   - 7-day moving average of closing price
   - Price level indicator

5. **Volume Feature** (1 feature): `log_volume`
   - Log-transformed trading volume
   - Indicates market activity and liquidity

## 4. Strategy Logic

### Prediction to Weight Mapping

```python
if pred <= 0:
    weight = 0      # No position (bearish signal)
elif pred <= 0.01:
    weight = 1      # Full position (mild bullish)
else:
    weight = 2      # Leveraged position (strong bullish)
```

### Volatility Constraint

To manage risk, the strategy enforces:

$$\sigma_{strategy} \leq 1.2 \times \sigma_{benchmark}$$

If violated, weights are scaled down:

$$scale = \frac{1.2 \times \sigma_{benchmark}}{\sigma_{strategy}}$$

$$w_{final} = \text{clip}(w_{raw} \times scale, 0, 2)$$

## 5. Qualitative Analysis

### Bull Market Behavior

During sustained uptrends:
- Model tends to maintain positive predictions
- Allocations typically at 1-2 (full to leveraged)
- Strategy captures upside with amplified returns
- Risk: May stay invested too long at market tops

### Bear Market Behavior

During downtrends:
- Model predictions turn negative or near-zero
- Allocations drop to 0-1 (reduced exposure)
- Capital preservation during crashes
- Risk: May miss recovery rallies due to delayed signals

### High Volatility Regimes

During volatile periods (e.g., crypto crashes):
- Volatility constraint likely activates
- Weights scaled down to reduce risk
- Lower absolute returns but controlled drawdowns
- Trade-off: Misses some recovery gains

## 6. Limitations and Risks

### Model Limitations

1. **Look-ahead bias**: Must ensure no future information leaks into features
2. **Non-stationarity**: Financial markets change over time; model may degrade
3. **Overfitting**: LSTM can memorize training patterns that don't generalize
4. **Latency**: Daily predictions assume end-of-day execution

### Market Risks

1. **Cryptocurrency volatility**: Extreme price swings can exceed historical patterns
2. **Liquidity risk**: Large positions may face slippage
3. **Regulatory risk**: Crypto regulations can cause sudden market shifts
4. **Exchange risk**: Platform failures, hacks, or freezes

### Strategy Risks

1. **Leverage risk**: 2x allocation amplifies both gains and losses
2. **Model failure**: Predictions may become unreliable in regime changes
3. **Execution costs**: Transaction fees and slippage not modeled

## 7. Future Work Ideas

### Model Improvements

- **Attention mechanisms**: Add self-attention layers for better pattern recognition
- **Ensemble methods**: Combine LSTM with gradient boosting models
- **Hyperparameter tuning**: Systematic search (Optuna, Ray Tune)
- **Alternative architectures**: Transformers, Temporal Convolutional Networks

### Feature Enhancements

- **Sentiment data**: Twitter/Reddit sentiment scores
- **On-chain metrics**: BTC transaction volume, active addresses, hash rate
- **Cross-asset signals**: S&P 500, Gold, DXY correlations
- **Technical indicators**: RSI, MACD, Bollinger Bands

### Strategy Extensions

- **Multi-asset portfolio**: Include ETH, SOL, other cryptocurrencies
- **Risk parity weighting**: Allocate based on inverse volatility
- **Regime detection**: Separate models for bull/bear markets
- **Transaction cost modeling**: Include realistic fees and slippage

### Validation Improvements

- **Walk-forward optimization**: Rolling window retraining
- **Monte Carlo simulation**: Stress test with synthetic scenarios
- **Out-of-sample testing**: Reserve recent data for final validation

---

*Appendix generated as part of the BTC Crypto Extension project.*
