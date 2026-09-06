# System Architecture

## Overview

The **ML-Based Short-Horizon Market Prediction & Trading System** is designed as an end-to-end, production-grade quantitative machine learning platform. It bridges high-frequency limit order book (LOB) data engineering, quantitative microstructure modeling, realistic execution backtesting, and low-latency API deployment.

```text
                                MARKET DATA LAYER
     ┌───────────────────────────────────────────────────────────────────┐
     │  Binance Spot / FI-2010 Historical LOB Matrix                    │
     └──────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                           DATA PIPELINE & VALIDATION
     ┌───────────────────────────────────────────────────────────────────┐
     │  Raw LOB Ingestion ──► Normalization ──► Schema Validation       │
     │                            ──► Parquet Storage                    │
     └──────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                            FEATURE & TARGET LAYER
     ┌───────────────────────────────────────────────────────────────────┐
     │  Microstructure Features (OBI, Microprice, Relative Spread)       │
     │  Target Labeling (H-horizon future return, Leakage-Free Purging)  │
     └──────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                        MODEL TRAINING & EVALUATION
     ┌───────────────────────────────────────────────────────────────────┐
     │  Purged Walk-Forward CV ──► Logistic Reg / XGBoost Classifier     │
     │  Probability Calibration ──► Classification Metrics               │
     └──────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                        REALISTIC EXECUTION BACKTESTER
     ┌───────────────────────────────────────────────────────────────────┐
     │  Probability Signal Rules ──► Bid-Ask Spread Execution            │
     │  Transaction Fees (5 bps) + Slippage ──► Sharpe & Risk Metrics    │
     └──────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
                         PRODUCTION DEPLOYMENT LAYER
     ┌───────────────────────────────────────────────────────────────────┐
     │  FastAPI Prediction Service (/predict, /health, /model/info)      │
     │  Multi-stage Docker Container ──► GitHub Actions CI/CD Pipeline   │
     └───────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Ingestion & Canonical Schemas (`src/trading_ml/data/`)
* **`ingestion.py`**: Idempotent data acquisition with SHA-256 integrity verification.
* **`normalization.py`**: Converts raw FI-2010 40-column matrix into explicit bid/ask level columns.
* **`schema.py`**: Pydantic schemas (`MarketEvent` for real-time events, `HistoricalLOBRecord` for historical benchmark indexing).
* **`validation.py`**: Validates strict price positivity ($Bid_1 > 0, Ask_1 > 0$), non-crossing conditions ($Bid_1 \le Ask_1$), non-negative depth sizes, and timestamp monotonicity.
* **`parquet.py`**: High-performance PyArrow Parquet storage layer.

### 2. Feature Engineering (`src/trading_ml/features/`)
Calculates backward-looking microstructure features using information strictly available at or before time $t$:
* **Mid-Price**: $M_t = \frac{Bid_1 + Ask_1}{2}$
* **Spread & Relative Spread**: $Spread_t = Ask_1 - Bid_1, \quad RelSpread_t = \frac{Spread_t}{M_t}$
* **Order Book Imbalance (OBI)**: $OBI_L = \frac{BidDepth_L - AskDepth_L}{BidDepth_L + AskDepth_L}$ for $L \in \{1, 3, 5, 10\}$
* **Microprice**: $MicroPrice_t = \frac{Ask_1 \cdot BidSize_1 + Bid_1 \cdot AskSize_1}{BidSize_1 + AskSize_1}$
* **Depth Dynamics**: Aggregate depth across levels 1–5 and $DepthRatio_5 = \frac{BidDepth_5}{AskDepth_5}$.

### 3. Leakage-Free Target Construction (`src/trading_ml/targets/`)
* Computes $H$-horizon future returns: $r_{t,H} = \frac{M_{t+H} - M_t}{M_t}$.
* Discretizes return into direction target labels:
  * $+1.0$ (UP) if $r_{t,H} > \text{threshold}$
  * $-1.0$ (DOWN) if $r_{t,H} < -\text{threshold}$
  * $0.0$ (HOLD) if $|r_{t,H}| \le \text{threshold}$
* **Strict Boundary Rule**: Unknown future observations (the final $H$ rows) remain `NaN` and are never coerced to HOLD or 0.

### 4. Quantitative Validation & Time-Series Splits (`src/trading_ml/validation/`)
* **`temporal_train_val_test_split`**: Sequential split into Train (70%), Validation (15%), and Test (15%).
* **Purging & Embargo**: Removes $H$ observations prior to fold boundaries to prevent overlapping target label leakage.
* **`PurgedWalkForwardCV`**: Expanding-window cross-validation generator for time-series evaluation.

### 5. ML Models & Probability Calibration (`src/trading_ml/models/`)
* **Baselines**: `MajorityBaseline`, `OBIRuleBaseline` ($OBI_1$ heuristic), `LogisticRegressionBaseline` (standardized multi-class linear model).
* **Primary ML Model**: `XGBoostMarketPredictor` with internal label mapping ($-1.0 \rightarrow 0, 0.0 \rightarrow 1, 1.0 \rightarrow 2$) and feature importance ranking.
* **Calibration**: `CalibratedPredictor` using Platt Scaling (sigmoid) or Isotonic regression.

### 6. Realistic Execution Backtesting (`src/trading_ml/backtesting/`)
* **`SignalGenerator`**: Converts model probability distributions $P(\text{DOWN}), P(\text{HOLD}), P(\text{UP})$ into target position signals $\{-1, 0, 1\}$.
* **`BacktestEngine`**: Realistic event-driven simulation:
  * Buys execute at $Ask_1 \times (1 + \text{slippage})$
  * Sells execute at $Bid_1 \times (1 - \text{slippage})$
  * Configurable exchange fees (`fee_bps=5.0`) and slippage (`slippage_bps=1.0`)
  * Strict position constraints (`max_position=1.0`)
* **`metrics.py`**: Computes Total Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, Profit Factor, Total Trades, Total Fees, and Slippage Costs.

### 7. Production API & Container Deployment (`api/`, `docker/`, `.github/`)
* **FastAPI Service**: Low-latency REST API exposing `/predict`, `/health`, `/model/info`.
* **Docker Container**: Multi-stage build with non-root security context and automated container healthcheck.
* **CI/CD Pipeline**: GitHub Actions workflow running unit tests, dataset generation, model evaluation, backtesting, and Docker build validation.
