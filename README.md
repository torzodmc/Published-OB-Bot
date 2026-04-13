# 🤖 OB-AI-Trading-Bot (Enhanced Synthetic Portfolio)

This repository contains the completely refactored, production-ready AI pipeline for high-frequency algorithmic scalping using 15-minute Order Blocks (OBs) on `BTCUSDT`.

By leveraging Iterative Amplitude Adjusted Fourier Transforms (IAAFT) inside the encapsulated `generator/` suite, this engine expands real-world historical market data into **60 Years of mathematically identical synthetic data**. It uses this expanded reality to train a series of hyper-robust XGBoost classifiers to recognize the safest geometries for trading Order Blocks under specific Risk/Reward ratios.

## 📂 Repository Architecture

```text
Published_OB_Bot/
├── README.md               # Pipeline Documentation and Metrics
├── dataset_manager.py      # [Step 1] Automates fetch, split, and synthesis
├── train_models.py         # [Step 2] Trains the AI portfolio mapped to specific R:R targets
├── test_models.py          # [Step 3] Independent validation utility for pre-existing models
├── input_data/             # Stores the generated Authentic and Synthetic CSVs
├── models/                 # Stores the finalized .joblib XGBoost models
└── generator/              # Contains the MarketForge AI Engine and Configs
```

## 🚀 Execution Instructions

Building the full suite is fully automated. Simply execute the scripts in numerical order:

### 1. Build the Database
```bash
python dataset_manager.py
# Downloads Binance 15m OHLCV natively.
# Splits the first 80% out for Training.
# Uses ATierEngine to spawn 60 Years of Synthetic variations.
```

### 2. Train the Portfolio
```bash
python train_models.py
# Reads the massive Authentic + Synthetic Data.
# Simulates Take Profits / Stop Losses for 4 explicit R:R models.
# Compiles all 4 models and outputs them to /models/
```

### 3. Independently Verify
```bash
python test_models.py
# Can be run at ANY time independently to verify the existing models.
# Runs them strictly on the Unseen 20% authentic data and generates confidence matrices.
```

---

## 📈 Live Portfolio Evaluation Metrics
*(Evaluated strictly on Unseen 20% authentic data - Jan 2024 to Dec 2024)*

### Model 1: The Fast Scalper (1:0.33)
**Risk: 1.0 ATR | Reward: 0.33 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  87.9%    |   643   | +108.5 R |     +108.5% |     +9.0% | LIQUIDATED (-100%)
 > 20 %      |  89.0%    |   634   | +116.1 R |     +116.1% |     +9.6% | LIQUIDATED (-100%)
 > 30 %      |  89.5%    |   610   | +116.2 R |     +116.2% |     +9.6% | LIQUIDATED (-100%)
 > 40 %      |  91.8%    |   547   | +120.7 R |     +120.7% |    +10.0% | LIQUIDATED (-100%)
 > 50 %      |  92.2%    |   447   | +101.0 R |     +101.0% |     +8.3% | LIQUIDATED (-100%)
 > 60 %      |  91.9%    |   347   |  +77.3 R |      +77.3% |     +6.4% | LIQUIDATED (-100%)
 > 70 %      |  94.3%    |   122   |  +31.0 R |      +31.0% |     +2.6% | LIQUIDATED (-100%)
 > 80 %      |  93.3%    |    15   |   +3.6 R |       +3.6% |     +0.3% | LIQUIDATED (-100%)
 > 90 %      | 100.0%    |     7   |   +2.3 R |       +2.3% |     +0.2% | +636.1%
```

### Model 2: The Baseline Strategy (1:0.50)
**Risk: 1.0 ATR | Reward: 0.50 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 20 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 30 %      |  77.7%    |   641   | +106.0 R |     +106.0% |     +8.8% | LIQUIDATED (-100%)
 > 40 %      |  78.9%    |   583   | +107.0 R |     +107.0% |     +8.8% | LIQUIDATED (-100%)
 > 50 %      |  83.3%    |   336   |  +84.0 R |      +84.0% |     +6.9% | LIQUIDATED (-100%)
 > 60 %      |  85.2%    |   155   |  +43.0 R |      +43.0% |     +3.6% | LIQUIDATED (-100%)
 > 70 %      | 100.0%    |     5   |   +2.5 R |       +2.5% |     +0.2% | +659.4%
 > 80 %      | 100.0%    |     4   |   +2.0 R |       +2.0% |     +0.2% | +406.2%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```

### Model 3: Standard Balance (1:1.00)
**Risk: 1.0 ATR | Reward: 1.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 20 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 30 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 40 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 50 %      |  50.4%    |   133   |   +1.0 R |       +1.0% |     +0.1% | LIQUIDATED (-100%)
 > 60 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 70 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 80 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```

### Model 4: The Trend Rider (1:3.00)
**Risk: 1.0 ATR | Reward: 3.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  24.7%    |   643   |   -7.0 R |       -7.0% |     -0.6% | LIQUIDATED (-100%)
 > 20 %      |  24.7%    |   643   |   -7.0 R |       -7.0% |     -0.6% | LIQUIDATED (-100%)
 > 30 %      |  24.8%    |   642   |   -6.0 R |       -6.0% |     -0.5% | LIQUIDATED (-100%)
 > 40 %      |  25.4%    |   603   |   +9.0 R |       +9.0% |     +0.7% | LIQUIDATED (-100%)
 > 50 %      |  30.2%    |   212   |  +44.0 R |      +44.0% |     +3.6% | LIQUIDATED (-100%)
 > 60 %      |  42.9%    |     7   |   +5.0 R |       +5.0% |     +0.4% | LIQUIDATED (-100%)
 > 70 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 80 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```
