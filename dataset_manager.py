import sys, os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "generator"))
from data_ingestion import fetch_ohlcv
from a_tier_engine import ATierEngine

def main():
    dest_dir = os.path.join(os.path.dirname(__file__), "input_data")
    os.makedirs(dest_dir, exist_ok=True)
    
    print("\n[1] Fetching Authentic Real Data (BTCUSDT 15m) 2020 -> Now...")
    print("    (Takes roughly 45 to 60 seconds downloading 75+ months of data)")
    df_real = fetch_ohlcv("BTCUSDT", "15m", "2020-01-01", "now")
    df_real.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    df_real.reset_index(drop=True, inplace=True)
    df_real.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'datetime':'Date'}, inplace=True)
    
    # Chronological Split (No lookahead bias)
    split_idx = int(len(df_real) * 0.8)
    df_train = df_real.iloc[:split_idx].copy()
    df_test = df_real.iloc[split_idx:].copy()
    
    train_path = os.path.join(dest_dir, "authentic_train_80.csv")
    test_path = os.path.join(dest_dir, "authentic_test_20.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"\n[SUCCESS] Split Authentic Market:")
    print(f"  --> {len(df_train)} rows -> {train_path}")
    print(f"  --> {len(df_test)} rows  -> {test_path} (COMPLETELY BLIND VAULT)\n")
    
    print("[2] Igniting ATier Engine - Compiling 60-Year Synthetic History...")
    print("    (Generating variations exclusively from the known 80% geometry)")
    engine = ATierEngine()
    df_train_mf = df_train.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume', 'Date':'datetime'})
    engine.train(df_train_mf)
    
    candles_60_yrs = 60 * 35064
    df_synth = engine.generate(candles_60_yrs)
    df_synth.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
    df_synth['Date'] = pd.to_datetime(df_synth['open_time'], unit='ms')
    
    synth_path = os.path.join(dest_dir, "synthetic_train_60_yrs.csv")
    df_synth.to_csv(synth_path, index=False)
    print(f"\n[SUCCESS] Master Synthetic DB Vaulted: {len(df_synth)} rows -> {synth_path}")

if __name__ == "__main__":
    main()
