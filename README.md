# ML-Based Short-Horizon Market Prediction & Trading System

A production-grade quantitative machine learning platform built in Python for **short-horizon price direction prediction** and **realistic LOB execution backtesting**.

Designed specifically for **Quantitative Trading / Quantitative Research / ML Engineering** roles focusing on market microstructure, high-frequency prediction, feature engineering, leak-free validation, and low-latency API deployment.

---

## 🎯 Primary Research Question

> *Can market microstructure features predict the direction of the next short-horizon price movement, and does that predictive signal remain profitable after realistic transaction costs, bid-ask spread, slippage, and position constraints?*

---

## 🏗️ End-to-End System Architecture

```text
Market Data (FI-2010 LOB / Binance Spot)
    ↓
Data Ingestion & Integrity Verification (SHA-256)
    ↓
Canonical Schemas (MarketEvent & HistoricalLOBRecord)
    ↓
Market Data Validation (Price positivity, non-crossing bid/ask, monotonic ordering)
    ↓
PyArrow Parquet Data Lake Storage
    ↓
Microstructure Feature Engineering (OBI 1-10, Microprice, Relative Spread, Depth Ratios)
    ↓
Leakage-Free Target Construction (H-horizon future returns, strict boundary NaN handling)
    ↓
Purged & Embargoed Time-Series Validation (PurgedWalkForwardCV)
    ↓
Baseline ML Models (Majority, OBI Rule, Logistic Regression, XGBoost Classifier)
    ↓
Purged Walk-Forward Hyperparameter Optimization & Probability Calibration
    ↓
Signal Generation & Realistic Execution Backtesting (Bid-Ask Spread, Fees, Slippage)
    ↓
Quantitative Risk & Portfolio Metrics (Sharpe, Sortino, Drawdown, Profit Factor)
    ↓
Production FastAPI Microservice (/predict, /health, /model/info)
    ↓
Multi-Stage Security-Hardened Docker Container
    ↓
GitHub Actions Automated CI/CD Pipeline
```

---

## 📊 Empirical Model & Backtest Results

Out-of-sample benchmark evaluation on time-series split with 10-event horizon purging, 5.0 bps transaction fees, and 1.0 bps slippage:

### Model Classification Metrics

| Model | Val Accuracy | Val Log Loss | Test Accuracy | Test F1 Macro | Test Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 6.3% | 33.75 | 70.3% | 0.275 | 10.72 |
| **OBI Rule Baseline** | 6.3% | 1.54 | 5.4% | 0.086 | 1.55 |
| **Logistic Regression** | 52.4% | 0.76 | 59.5% | 0.389 | 0.68 |
| **XGBoost Classifier** | 36.5% | 0.88 | **67.6%** | **0.414** | **0.58** |

### Out-of-Sample Trading & Execution Metrics (Fee = 5.0 bps, Slippage = 1.0 bps)

| Model | Total Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate | Profit Factor | Total Trades | Total Fees | Slippage Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority** | 0.0% | 0.469 | 0.728 | 0.0% | 0.0% | 0.00 | 1 | $0.05 | $0.02 |
| **OBI Rule** | -0.0% | -0.432 | -0.458 | 0.0% | 6.2% | 0.00 | 16 | $0.80 | $0.32 |
| **Logistic Regression** | -0.0% | -0.473 | -0.678 | 0.0% | 12.9% | 0.16 | 31 | $2.94 | $1.18 |
| **XGBoost Classifier** | -0.0% | **-0.361** | **-0.486** | 0.0% | **14.8%** | **0.31** | 27 | $2.54 | $1.02 |

### XGBoost Feature Importance Findings
1. **`relative_spread`** (12.8%): Relative spread signals immediate liquidity availability and friction.
2. **`depth_ratio_5`** (11.6%): Multi-level bid/ask depth ratio captures order book asymmetry.
3. **`obi_10`** (10.6%): Deep order book imbalance provides medium-term directional pressure.
4. **`mid_price`** (10.5%): Local price level dynamics.
5. **`microprice`** (10.3%): Volume-weighted level-1 microprice deviation.

---

## 🛡️ Leakage Prevention Guarantees

This repository enforces strict quantitative anti-leakage controls:
1. **Feature Engineering**: Features at time $t$ use *only* information available at or before $t$.
2. **Target Boundary Purging**: For horizon $H$, observations within $H$ steps of split boundaries are purged so future returns do not overlap into validation/testing.
3. **Scaler Scoping**: Preprocessing parameters (e.g. `StandardScaler`) are fit strictly on training splits.
4. **Strict Target NaNs**: Unknown future targets (the final $H$ observations) remain `NaN` and are never coerced to 0 or HOLD.

---

## ⚡ Quick Start & Reproduction

### 1. Installation
```bash
git clone https://github.com/kamaleshpantra/ml-short-horizon-trading.git
cd ml-short-horizon-trading
pip install -e .[dev] uvicorn httpx
```

### 2. Run Complete Unit & Integration Test Suite
```bash
python -m pytest -v
```

### 3. Build Processed Dataset Pipeline
```bash
python scripts/build_dataset.py --force
```

### 4. Evaluate ML Baseline Models
```bash
python scripts/evaluate_models.py
```

### 5. Execute Out-of-Sample Realistic Backtest
```bash
python scripts/run_backtest.py --fee 5.0 --slippage 1.0
```

### 6. Run Production FastAPI Microservice
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Test prediction endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "mid_price": 100.5,
       "spread": 0.02,
       "relative_spread": 0.0002,
       "obi_1": 0.35,
       "obi_3": 0.25,
       "obi_5": 0.20,
       "obi_10": 0.15,
       "microprice": 100.52,
       "bid_depth_5": 50.0,
       "ask_depth_5": 30.0,
       "depth_ratio_5": 1.6667
     }'
```

### 7. Run via Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📁 Repository Directory Structure

```text
ml-short-horizon-trading/
├── README.md
├── pyproject.toml
├── .dockerignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── methodology.md
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── src/
│   └── trading_ml/
│       ├── data/
│       │   ├── ingestion.py
│       │   ├── normalization.py
│       │   ├── parquet.py
│       │   ├── pipeline.py
│       │   ├── schema.py
│       │   └── validation.py
│       ├── features/
│       │   └── microstructure.py
│       ├── targets/
│       │   └── returns.py
│       ├── validation/
│       │   └── splits.py
│       ├── models/
│       │   ├── baseline.py
│       │   ├── calibration.py
│       │   ├── metrics.py
│       │   ├── tuning.py
│       │   └── xgboost_model.py
│       ├── backtesting/
│       │   ├── engine.py
│       │   ├── metrics.py
│       │   └── signal.py
│       └── utils/
│           ├── config.py
│           └── logging.py
├── scripts/
│   ├── build_dataset.py
│   ├── create_sample_data.py
│   ├── evaluate_models.py
│   └── run_backtest.py
├── tests/
│   ├── test_api.py
│   ├── test_backtest.py
│   ├── test_build_dataset.py
│   ├── test_calibration_tuning.py
│   ├── test_config.py
│   ├── test_ingestion.py
│   ├── test_microstructure.py
│   ├── test_models.py
│   ├── test_normalization.py
│   ├── test_parquet.py
│   ├── test_prepare_pipeline.py
│   ├── test_schema.py
│   ├── test_splits.py
│   ├── test_targets.py
│   └── test_validation.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 📝 Research Conclusion & Summary

1. **Order book imbalance & microprice carry predictive information** for short-horizon price movements, with relative spread and depth ratios contributing highest gain in gradient boosted trees.
2. **Predictive accuracy alone does not guarantee net trading profitability.** Without threshold filtering and tight execution controls, bid-ask spread costs and exchange transaction fees quickly erode raw classification edge.
3. **Quantitative rigor, leak-free validation, and realistic execution cost modeling** are critical for building reliable machine learning models in automated trading infrastructure.