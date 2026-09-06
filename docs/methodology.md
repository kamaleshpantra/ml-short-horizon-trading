# Quantitative Methodology & Leakage Prevention

## 1. Primary Research Question

> Can market microstructure features predict the direction of the next short-horizon price movement, and does that predictive signal remain profitable after realistic transaction costs, bid-ask spread, slippage, latency, and position constraints?

---

## 2. Leakage Prevention Protocol

Data leakage is the single most common failure mode in quantitative machine learning. This codebase enforces strict mathematical and architectural boundaries:

### Feature Generation
* Every feature at time $t$ uses **only** order-book information available at or before $t$.
* No future prices, future order-book states, or forward-looking rolling windows are ever used in feature calculation.

### Scaler & Preprocessing Scope
* All parameters (e.g. `StandardScaler` mean and variance) are fitted **strictly on the training split**.
* Validation and Test splits are transformed using training parameters.

### Target Horizon Purging Math
When computing future returns over horizon $H$:
$$r_{t,H} = \frac{M_{t+H} - M_t}{M_t}$$
The observation at $T_{\text{train\_end}} - k$ (where $k < H$) computes target labels using future prices up to $T_{\text{train\_end}} - k + H$, which extends into the validation split!

To eliminate this overlap:
$$\text{Train Range} = \left[0, T_{\text{train\_end}} - H\right]$$
$$\text{Val Range} = \left[T_{\text{train\_end}} + \text{embargo}, T_{\text{val\_end}} - H\right]$$
The final $H$ observations before the fold boundary are **purged**, ensuring zero information leakage across splits.

---

## 3. Microstructure Feature Definitions

| Feature | Formula | Market Microstructure Meaning |
| :--- | :--- | :--- |
| **Mid-Price** | $M_t = \frac{Bid_1 + Ask_1}{2}$ | Unbiased estimate of instantaneous market value |
| **Spread** | $Spread_t = Ask_1 - Bid_1$ | Direct cost of immediacy / market liquidity friction |
| **Relative Spread** | $RelSpread_t = \frac{Spread_t}{M_t}$ | Scale-invariant spread cost relative to price level |
| **Order Book Imbalance (OBI)** | $OBI_L = \frac{BidDepth_L - AskDepth_L}{BidDepth_L + AskDepth_L}$ | Supply vs. demand volume asymmetry across top $L$ levels |
| **Microprice** | $MicroPrice_t = \frac{Ask_1 BidSize_1 + Bid_1 AskSize_1}{BidSize_1 + AskSize_1}$ | Queue-weighted midpoint measuring short-term inventory pressure |
| **Depth Ratio** | $DepthRatio_5 = \frac{BidDepth_5}{AskDepth_5}$ | Multi-level order book liquidity asymmetry ratio |

---

## 4. Execution Simulation & Cost Model

Traditional ML models assume execution at the midpoint $M_t$ with zero transaction costs. In high-frequency / short-horizon trading, this assumption creates fake backtest profitability.

Our backtesting engine enforces **realistic execution pricing**:

* **Buy Execution Price**:
  $$P_{\text{buy}} = Ask_1 \cdot \left(1 + \text{slippage\_rate}\right)$$
* **Sell Execution Price**:
  $$P_{\text{sell}} = Bid_1 \cdot \left(1 - \text{slippage\_rate}\right)$$
* **Exchange Transaction Fee**:
  $$\text{Fee} = P_{\text{exec}} \cdot \text{fee\_rate}$$
* **Net Position P&L**:
  $$\text{PnL}_{\text{net}} = \text{PnL}_{\text{gross}} - \text{Fees} - \text{SpreadCost} - \text{SlippageCost}$$

Unless a predictive signal exceeds the round-trip bid-ask spread plus exchange fees (e.g. 5 bps maker/taker fee), it cannot generate net trading profits.
