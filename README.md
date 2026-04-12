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
 > 30 %      |  88.4%    |   619   | +108.5 R |     +108.5% |     +9.0% | LIQUIDATED (-100%)
 > 40 %      |  89.6%    |   538   | +103.1 R |     +103.1% |     +8.5% | LIQUIDATED (-100%)
 > 50 %      |  88.9%    |   416   |  +76.1 R |      +76.1% |     +6.3% | LIQUIDATED (-100%)
 > 60 %      |  89.5%    |   219   |  +41.7 R |      +41.7% |     +3.4% | LIQUIDATED (-100%)
 > 70 %      |  92.0%    |    25   |   +5.6 R |       +5.6% |     +0.5% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |    10   |   +3.3 R |       +3.3% |     +0.3% | +1,631.9%
 > 90 %      | 100.0%    |     7   |   +2.3 R |       +2.3% |     +0.2% | +636.1%
```

### Model 2: The Baseline Strategy (1:0.50)
**Risk: 1.0 ATR | Reward: 0.50 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 20 %      |  77.4%    |   643   | +104.0 R |     +104.0% |     +8.6% | LIQUIDATED (-100%)
 > 30 %      |  77.5%    |   632   | +103.0 R |     +103.0% |     +8.5% | LIQUIDATED (-100%)
 > 40 %      |  78.1%    |   558   |  +96.0 R |      +96.0% |     +7.9% | LIQUIDATED (-100%)
 > 50 %      |  80.9%    |   314   |  +67.0 R |      +67.0% |     +5.5% | LIQUIDATED (-100%)
 > 60 %      |  86.5%    |    52   |  +15.5 R |      +15.5% |     +1.3% | LIQUIDATED (-100%)
 > 70 %      |  91.7%    |    12   |   +4.5 R |       +4.5% |     +0.4% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |     6   |   +3.0 R |       +3.0% |     +0.2% | +1,039.1%
 > 90 %      | 100.0%    |     6   |   +3.0 R |       +3.0% |     +0.2% | +1,039.1%
```

### Model 3: Standard Balance (1:1.00)
**Risk: 1.0 ATR | Reward: 1.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 20 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 30 %      |  51.5%    |   643   |  +19.0 R |      +19.0% |     +1.6% | LIQUIDATED (-100%)
 > 40 %      |  51.8%    |   597   |  +21.0 R |      +21.0% |     +1.7% | LIQUIDATED (-100%)
 > 50 %      |  63.8%    |   141   |  +39.0 R |      +39.0% |     +3.2% | LIQUIDATED (-100%)
 > 60 %      |  61.5%    |    13   |   +3.0 R |       +3.0% |     +0.2% | LIQUIDATED (-100%)
 > 70 %      |  60.0%    |     5   |   +1.0 R |       +1.0% |     +0.1% | LIQUIDATED (-100%)
 > 80 %      | 100.0%    |     1   |   +1.0 R |       +1.0% |     +0.1% | +100.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```

### Model 4: The Trend Rider (1:3.00)
**Risk: 1.0 ATR | Reward: 3.00 ATR**

```text
CONFIDENCE   | WIN RATE  | TRADES  | NET R    | 1% RISK PnL  | 1% MO PnL  | 100% ALL-IN PnL
------------------------------------------------------------------------------------------
 > 10 %      |  24.7%    |   643   |   -7.0 R |       -7.0% |     -0.6% | LIQUIDATED (-100%)
 > 20 %      |  24.8%    |   642   |   -6.0 R |       -6.0% |     -0.5% | LIQUIDATED (-100%)
 > 30 %      |  24.8%    |   640   |   -4.0 R |       -4.0% |     -0.3% | LIQUIDATED (-100%)
 > 40 %      |  25.2%    |   618   |   +6.0 R |       +6.0% |     +0.5% | LIQUIDATED (-100%)
 > 50 %      |  28.8%    |   219   |  +33.0 R |      +33.0% |     +2.7% | LIQUIDATED (-100%)
 > 60 %      |  50.0%    |     6   |   +6.0 R |       +6.0% |     +0.5% | LIQUIDATED (-100%)
 > 70 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 80 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
 > 90 %      |   0.0%    |     0   |   +0.0 R |       +0.0% |     +0.0% | +0.0%
```
