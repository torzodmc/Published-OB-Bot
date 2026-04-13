import sys, os, joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "input_data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Risk Multipliers we want to train dedicated models for
MODELS_TO_TRAIN = {
    'model_1_scalp.joblib': 0.33,
    'model_2_base.joblib':  0.50,
    'model_3_1to1.joblib':  1.00,
    'model_4_1to3.joblib':  3.00
}

def calculate_ema(s, span): return s.ewm(span=span, adjust=False).mean()
def calculate_atr(h, l, c, p=14):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def calculate_rsi(s, p=14):
    d=s.diff(); g=(d.where(d>0,0)).rolling(p).mean(); l=(-d.where(d<0,0)).rolling(p).mean()
    return 100-(100/(1+g/l))

def detect_fvg(df, start_idx, end_idx, ob_type):
    if end_idx - start_idx < 2: return 0
    for k in range(start_idx+1, end_idx):
        if k+1 >= len(df): continue
        if ob_type == 'bullish':
            if df['Low'].iloc[k+1] > df['High'].iloc[k-1]: return 1
        else:
            if df['High'].iloc[k+1] < df['Low'].iloc[k-1]: return 1
    return 0

def extract_trades_vectorized(df_raw, multipliers):
    df = df_raw.copy().reset_index(drop=True)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_200'] = calculate_ema(df['Close'], 200)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['VMA_20'] = df['Volume'].rolling(20).mean()
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour
    
    SLB = 10
    active_obs = []
    last_sh = None; last_sl = None
    trades = []
    
    Highs = df['High'].values
    Lows = df['Low'].values
    Closes = df['Close'].values
    Opens = df['Open'].values
    Vols = df['Volume'].values
    VMA20 = df['VMA_20'].values
    ATRs = df['ATR'].values
    EMA50 = df['EMA_50'].values
    EMA200 = df['EMA_200'].values
    RSIs = df['RSI'].values
    Dates = df['Date'].values
    Hours = df['Hour'].values
    
    for i in range(SLB * 2, len(df)):
        cand = i - SLB
        if Highs[cand] == np.max(Highs[cand - SLB : cand + SLB + 1]):
            last_sh = {'idx': cand, 'val': Highs[cand]}
        if Lows[cand] == np.min(Lows[cand - SLB : cand + SLB + 1]):
            last_sl = {'idx': cand, 'val': Lows[cand]}
        
        row_close = Closes[i]
        
        if last_sh and row_close > last_sh['val']:
            lb_start = max(0, i - 24)
            if i > lb_start:
                idx_min = lb_start + np.argmin(Lows[lb_start:i])
                v_surge = Vols[i] / VMA20[i] if VMA20[i] > 0 else 1
                f = detect_fvg(df, idx_min, i, 'bullish')
                active_obs.append({'type':'bullish', 'top':Highs[idx_min], 'bot':Lows[idx_min], 'age':0, 'vol_surge':v_surge, 'fvg':f})
            last_sh = None
            
        if last_sl and row_close < last_sl['val']:
            lb_start = max(0, i - 24)
            if i > lb_start:
                idx_max = lb_start + np.argmax(Highs[lb_start:i])
                v_surge = Vols[i] / VMA20[i] if VMA20[i] > 0 else 1
                f = detect_fvg(df, idx_max, i, 'bearish') 
                active_obs.append({'type':'bearish', 'top':Highs[idx_max], 'bot':Lows[idx_max], 'age':0, 'vol_surge':v_surge, 'fvg':f})
            last_sl = None

        for ob in active_obs:
            ob['age'] += 1
        
        atr = ATRs[i]
        if pd.isna(atr) or atr == 0: continue
        
        hit_obs = []
        for ob in active_obs:
            if ob['type'] == 'bullish':
                if Lows[i] <= ob['top'] and Opens[i] > ob['top']: hit_obs.append(ob)
            else:
                if Highs[i] >= ob['bot'] and Opens[i] < ob['bot']: hit_obs.append(ob)
                    
        for hit_ob in hit_obs:
            if hit_ob in active_obs: active_obs.remove(hit_ob)

        new_active = []
        for ob in active_obs:
            if ob['age'] < 150: new_active.append(ob)
        active_obs = new_active
        
        for ob in hit_obs:
            ep = ob['top'] if ob['type'] == 'bullish' else ob['bot']
            ob_width = ob['top'] - ob['bot']
            if ob_width < atr * 0.1: continue
            
            stop = ob['bot'] - (atr * 1.5) if ob['type'] == 'bullish' else ob['top'] + (atr * 1.5)
            stop_dist = abs(ep - stop)
            
            try: vol_base = VMA20[i-1] if VMA20[i-1] > 0 else 1
            except: vol_base = 1
                
            outcomes = {m: None for m in multipliers}
            for j in range(i, min(i+500, len(df))): 
                l_j = Lows[j]; h_j = Highs[j]
                all_done = True
                
                for m in multipliers:
                    if outcomes[m] is None:
                        all_done = False
                        tp_m = ep + (stop_dist * m) if ob['type'] == 'bullish' else ep - (stop_dist * m)
                        if ob['type'] == 'bullish':
                            if l_j <= stop: outcomes[m] = 0
                            elif h_j >= tp_m: outcomes[m] = 1
                        else:
                            if h_j >= stop: outcomes[m] = 0
                            elif l_j <= tp_m: outcomes[m] = 1
                if all_done: break
            
            tr = {
                'Date': Dates[i], 'ob_age': ob['age'], 'atr': atr, 'rsi': RSIs[i-1],
                'ob_width_atr': ob_width / atr, 'ob_bos_vol_surge': ob['vol_surge'],
                'ob_has_fvg': ob['fvg'], 'mit_vol_surge': Vols[i] / vol_base,
                'dist_ema_50': (Closes[i-1] - EMA50[i-1]) / atr, 
                'dist_ema_200': (Closes[i-1] - EMA200[i-1]) / atr,
                'mtf_aligned': 1 if (1 if EMA50[i-1] > EMA200[i-1] else -1) == (1 if ob['type'] == 'bullish' else -1) else 0,
                'hour_of_day': Hours[i], 'ob_type': 1 if ob['type'] == 'bullish' else 0,
            }
            for m in multipliers: tr[f'outcome_{m}'] = outcomes[m] if outcomes[m] is not None else 0
            trades.append(tr)

    return pd.DataFrame(trades)

def train_model(X_train, y_train, X_test, y_test, m):
    scale_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=750, learning_rate=0.01, max_depth=4, subsample=0.7,
        colsample_bytree=0.7, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*90)
    print(f"|  MODEL EVALUATION: (Ratio -> 1 : {m})  |")
    print("="*90)
    print(f"{'CONFIDENCE':<12} | {'WIN RATE':<9} | {'TRADES':<7} | {'NET R':<8} | {'1% RISK PnL':<12} | {'100% ALL-IN PnL'}")
    print("-" * 90)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        p_thresh = (probs >= t).astype(int)
        trades = int(np.sum(p_thresh))
        if trades > 0:
            wins = int(np.sum((p_thresh == 1) & (y_test == 1)))
            wr = (wins / trades) * 100
            loss = trades - wins
            net = (wins * m) - (loss * 1.0)
            
            pnl_1 = net * 1.0
            pnl_all_str = f"LIQUIDATED (-100%)" if loss > 0 else f"+{(((1.0 + m) ** wins) - 1.0) * 100.0:,.1f}%"
            
            print(f" > {int(t*100):<3}%      | {wr:>5.1f}%    | {trades:>5}   | {net:>+6.1f} R | {pnl_1:>+10.1f}% | {pnl_all_str}")
        else:
            print(f" > {int(t*100):<3}%      | {'0.0%':>5}    | {'0':>5}   | {'+0.0':>6} R | {'+0.0':>10}% | +0.0%")
            
    print("="*90)
    return model, acc

def main():
    train_auth_path = os.path.join(INPUT_DIR, "authentic_train_80.csv")
    train_synth_path = os.path.join(INPUT_DIR, "synthetic_train_60_yrs.csv")
    # For early stopping eval mapping
    test_auth_path = os.path.join(INPUT_DIR, "authentic_test_20.csv")
    
    if not all(os.path.exists(p) for p in [train_auth_path, train_synth_path, test_auth_path]):
        print("Error: Missing input CSV datasets! Run dataset_manager.py first.")
        return
        
    print("\n[1] Loading CSV Datasets into Memory...")
    df_auth = pd.read_csv(train_auth_path)
    df_synth = pd.read_csv(train_synth_path)
    df_test = pd.read_csv(test_auth_path)
    
    multipliers = list(MODELS_TO_TRAIN.values())
    
    print("[2] High-Velocity Feature Extraction (Authentic + Synthetic) ...")
    events_auth = extract_trades_vectorized(df_auth, multipliers)
    events_synth = extract_trades_vectorized(df_synth, multipliers)
    events_test = extract_trades_vectorized(df_test, multipliers)
    
    df_train_events = pd.concat([events_auth, events_synth], ignore_index=True)
    
    print(f"\n[3] Engineering Complete. Matrix dimensions: {len(df_train_events)} total training events.")
    
    for filename, m in MODELS_TO_TRAIN.items():
        print(f"\n---> Training Portfolio Node: {filename} mapped to 1:{m} Risk:Reward")
        y_col = f'outcome_{m}'
        drop_cols = ['Date'] + [f'outcome_{x}' for x in multipliers]
        
        X_train = df_train_events.drop(columns=drop_cols)
        y_train = df_train_events[y_col].values
        
        X_test = events_test.drop(columns=drop_cols)
        y_test = events_test[y_col].values
        
        model, acc = train_model(X_train, y_train, X_test, y_test, m)
        joblib.dump(model, os.path.join(MODELS_DIR, filename))
        print(f"     [+] Saved successfully to models/{filename}")

    print("\n[SUCCESS] AI Farm execution complete. All custom models generated and vaulted.")

if __name__ == "__main__":
    main()
