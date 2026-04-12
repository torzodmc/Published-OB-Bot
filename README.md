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
 > 20 %      |  87.9%    |   643   | +108.5 R |     +108.5% |     +9.0% | LIQUIDATED (-100%)
 > 30 %      |  88.2%    |   621   | +107.8 R |     +107.8% |     +8.9% | LIQUIDATED (-100%)
 > 40 %      |  89.8%    |   529   | +102.8 R |     +102.8% |     +8.5% | LIQUIDATED (-100%)
 > 50 %      |  89.6%    |   393   |  +75.2 R |      +75.2% |     +6.2% | LIQUIDATED (-100%)
 > 60 %      |  88.4%    |   155   |  +27.2 R |      +27.2% |     +2.2% | LIQUIDATED (-100%)
 > 70 %      |  88.0%    |    25   |   +4.3 R |       +4.3% |     +0.4% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |     9   |   +3.0 R |       +3.0% |     +0.2% | +1,202.2%
 > 90 %      | 100.0%    |     8   |   +2.6 R |       +2.6% |     +0.2% | +879.1%
```

### Model 2: The Baseline Strategy (1:0.50)
**Risk: 1.0 ATR | Reward: 0.50 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 20 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 30 %      |  77.8%    |   632   | +106.0 R |     +106.0% |     +8.8% | LIQUIDATED (-100%)
 > 40 %      |  77.9%    |   556   |  +93.5 R |      +93.5% |     +7.7% | LIQUIDATED (-100%)
 > 50 %      |  82.5%    |   280   |  +66.5 R |      +66.5% |     +5.5% | LIQUIDATED (-100%)
 > 60 %      |  85.7%    |    49   |  +14.0 R |      +14.0% |     +1.2% | LIQUIDATED (-100%)
 > 70 %      |  88.9%    |     9   |   +3.0 R |       +3.0% |     +0.2% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |     5   |   +2.5 R |       +2.5% |     +0.2% | +659.4%
 > 90 %      | 100.0%    |     4   |   +2.0 R |       +2.0% |     +0.2% | +406.2%
```

### Model 3: Standard Balance (1:1.00)
**Risk: 1.0 ATR | Reward: 1.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 20 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 30 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 40 %      |  52.3%    |   597   |  +27.0 R |      +27.0% |     +2.2% | LIQUIDATED (-100%)
 > 50 %      |  59.9%    |   167   |  +33.0 R |      +33.0% |     +2.7% | LIQUIDATED (-100%)
 > 60 %      |  66.7%    |    12   |   +4.0 R |       +4.0% |     +0.3% | LIQUIDATED (-100%)
 > 70 %      |  50.0%    |     4   |   +0.0 R |       +0.0% |     +0.0% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |     1   |   +1.0 R |       +1.0% |     +0.1% | +100.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```

### Model 4: The Trend Rider (1:3.00)
**Risk: 1.0 ATR | Reward: 3.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  24.7%    |   643   |   -7.0 R |       -7.0% |     -0.6% | LIQUIDATED (-100%)
 > 20 %      |  24.7%    |   643   |   -7.0 R |       -7.0% |     -0.6% | LIQUIDATED (-100%)
 > 30 %      |  24.8%    |   641   |   -5.0 R |       -5.0% |     -0.4% | LIQUIDATED (-100%)
 > 40 %      |  25.4%    |   603   |   +9.0 R |       +9.0% |     +0.7% | LIQUIDATED (-100%)
 > 50 %      |  27.7%    |   206   |  +22.0 R |      +22.0% |     +1.8% | LIQUIDATED (-100%)
 > 60 %      |  50.0%    |     6   |   +6.0 R |       +6.0% |     +0.5% | LIQUIDATED (-100%)
 > 70 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 80 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```
