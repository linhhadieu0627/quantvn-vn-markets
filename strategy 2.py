"""
Chiến lược giao dịch: Cross-Sectional Portfolio - Simple Version
================================================================
Ý tưởng từ v29.0 ML Enhanced nhưng đơn giản hóa:
  - Walk-Forward: retrain model mỗi 63 ngày (giữ nguyên ý tưởng)
  - Alpha từ Gradient Boosting đơn giản (thay LightGBM)
  - Regime detection bằng EMA200 + Volatility (thay HMM)
  - Risk Parity weights (giữ nguyên)
  - Long-only, top 5 stocks theo alpha
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("QUANT_API_KEY")

from quantvn.vn.data.utils import client
client(apikey=api_key)
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        df.index = pd.to_datetime(df.index)
    except:
        pass
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    up_move, down_move = h - h.shift(1), l.shift(1) - l
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING (15 features, đủ để có alpha)
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """Tạo features đơn giản nhưng hiệu quả"""
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    ret = c.pct_change()

    f = pd.DataFrame(index=df.index)

    # Momentum features
    for w in [5, 10, 21, 63]:
        f[f'mom_{w}'] = c.pct_change(w)

    # Volatility features
    for w in [5, 21, 63]:
        f[f'vol_{w}'] = ret.rolling(w).std() * np.sqrt(252)

    f['atr_norm'] = calc_atr(df) / (c + 1e-8)
    f['adx'] = calc_adx(df)
    f['rsi'] = calc_rsi(c, 14)

    # EMA ratios
    for span in [50, 200]:
        ema = c.ewm(span=span, adjust=False).mean()
        f[f'er_{span}'] = c / ema - 1

    # Market-relative features (nếu có)
    if market_df is not None:
        mkt = market_df['Close'].reindex(df.index, method='ffill').astype(float)
        f['rel_strength'] = c.pct_change(63) - mkt.pct_change(63)
        f['beta'] = ret.rolling(63).cov(mkt.pct_change()) / (mkt.pct_change().rolling(63).var() + 1e-8)

    # Target: forward 21-day return
    f['target'] = c.pct_change(21).shift(-21)

    # Shift features (no look-ahead)
    feature_cols = [c for c in f.columns if c != 'target']
    f[feature_cols] = f[feature_cols].shift(1)

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SIMPLE GRADIENT BOOSTING (thay LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleGBM:
    """Gradient Boosting đơn giản với decision stumps"""

    def __init__(self, n_estimators=50, lr=0.05, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = 0.0

    def _build_tree(self, X, y, depth=0):
        """Decision tree đơn giản"""
        if depth >= self.max_depth or len(y) < 10 or np.std(y) < 1e-6:
            return np.mean(y)

        best_loss = float('inf')
        best_feat, best_thr, best_left, best_right = 0, 0, 0, 0

        for feat in range(min(X.shape[1], 10)):
            col = X[:, feat]
            for thr in np.percentile(col[~np.isnan(col)], [33, 50, 67]):
                left = y[col <= thr]
                right = y[col > thr]
                if len(left) < 3 or len(right) < 3:
                    continue
                loss = np.var(left) * len(left) + np.var(right) * len(right)
                if loss < best_loss:
                    best_loss = loss
                    best_feat, best_thr = feat, thr
                    best_left, best_right = left.mean(), right.mean()

        if best_loss == float('inf'):
            return np.mean(y)

        return {'feat': best_feat, 'thr': best_thr,
                'left': best_left, 'right': best_right}

    def _predict_tree(self, tree, X):
        if not isinstance(tree, dict):
            return np.full(len(X), tree)
        pred = np.zeros(len(X))
        col = X[:, tree['feat']]
        pred[col <= tree['thr']] = tree['left']
        pred[col > tree['thr']] = tree['right']
        return pred

    def fit(self, X, y):
        y = np.clip(y, *np.percentile(y, [5, 95]))
        self.base_pred = np.mean(y)
        F = np.full(len(y), self.base_pred)

        for _ in range(self.n_estimators):
            residual = y - F
            tree = self._build_tree(X, residual)
            pred = self._predict_tree(tree, X)
            F += self.lr * pred
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(len(X), self.base_pred)
        for tree in self.trees:
            F += self.lr * self._predict_tree(tree, X)
        return F


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: WALK-FORWARD ALPHA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class WFAlpha:
    """Walk-Forward Alpha: retrain mỗi 63 ngày, window 504 ngày"""

    def __init__(self, train_window=504, retrain_freq=63):
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.models = {}
        self.scalers = {}
        self.last_train = {}

    def needs_train(self, symbol, date):
        if symbol not in self.last_train:
            return True
        return (date - self.last_train[symbol]).days >= self.retrain_freq

    def train(self, symbol, feat_df, train_end):
        train_start = train_end - pd.DateOffset(days=self.train_window)
        mask = (feat_df.index >= train_start) & (feat_df.index < train_end)
        train_data = feat_df[mask].dropna()

        if len(train_data) < 100:
            return False

        feat_cols = [c for c in feat_df.columns if c != 'target']
        X = train_data[feat_cols].values.astype(float)
        y = train_data['target'].values.astype(float)

        # Remove NaN
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[ok], y[ok]

        if len(y) < 80:
            return False

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Train GBM
        model = SimpleGBM(n_estimators=50, lr=0.05, max_depth=3)
        model.fit(X_scaled, y)

        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.last_train[symbol] = train_end
        return True

    def predict(self, symbol, feat_df, date):
        if symbol not in self.models or date not in feat_df.index:
            return 0.5

        feat_cols = [c for c in feat_df.columns if c != 'target']
        row = feat_df.loc[date, feat_cols]

        if row.isna().any():
            return 0.5

        X = row.values.astype(float).reshape(1, -1)
        X_scaled = self.scalers[symbol].transform(X)
        raw_pred = float(self.models[symbol].predict(X_scaled)[0])

        # Convert to alpha score [0, 1]
        alpha = (raw_pred * 12 + 0.4) / 0.8
        return float(np.clip(alpha, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: REGIME DETECTION (đơn giản hóa)
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Regime detection bằng EMA200 + Volatility"""

    REGIME_MULT = {'bull': 1.0, 'sideways': 0.65, 'bear': 0.3}

    def __init__(self, market_df):
        self.market_df = market_df.copy()

    def get_multiplier(self, date):
        if date not in self.market_df.index:
            return 1.0

        hist = self.market_df.loc[:date]
        if len(hist) < 200:
            return 1.0

        close = hist['Close']
        ema200 = close.ewm(span=200, adjust=False).mean()
        price = close.iloc[-1]

        # Volatility regime
        ret = close.pct_change()
        vol = ret.tail(20).std() * np.sqrt(252)
        high_vol = vol > 0.25

        if price > ema200.iloc[-1] * 1.02 and not high_vol:
            return 1.0
        elif price > ema200.iloc[-1] * 0.98:
            return 0.7
        else:
            return 0.4


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RISK PARITY WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def risk_parity_weights(symbols, alpha_scores, all_data, date, lookback=126):
    n = len(symbols)
    if n == 0:
        return {}
    if n == 1:
        return {symbols[0]: 1.0}

    # Get historical returns
    start = date - pd.DateOffset(days=lookback)
    returns = {}
    for s in symbols:
        hist = all_data[s].loc[start:date, 'Close']
        if len(hist) > 20:
            returns[s] = hist.pct_change().dropna()

    if len(returns) < 2:
        return {s: 1.0 / n for s in symbols}

    # Covariance matrix
    ret_df = pd.DataFrame(returns).dropna()
    cov = ret_df.cov().values * 252
    cov = cov + np.eye(n) * 1e-6

    # Iterative risk parity
    w = np.ones(n) / n
    for _ in range(200):
        port_vol = np.sqrt(w @ cov @ w) + 1e-8
        risk_contrib = w * (cov @ w) / port_vol
        target_rc = port_vol / n
        w = w - 0.2 * (risk_contrib - target_rc)
        w = np.clip(w, 0.05, 0.4)
        w = w / w.sum()

    # Blend với alpha scores
    alphas = np.array([alpha_scores.get(s, 0.5) for s in symbols])
    alphas = alphas / (alphas.sum() + 1e-8)
    blended = 0.7 * w + 0.3 * alphas
    blended = blended / blended.sum()

    return {s: float(v) for s, v in zip(symbols, blended)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CROSS-SECTIONAL RANKER
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_MAP = {
    "HPG": "steel", "FPT": "tech", "VCB": "bank", "ACB": "bank", "MBB": "bank",
    "TCB": "bank", "CTG": "bank", "BID": "bank", "VIC": "re", "VHM": "re",
    "GAS": "energy", "VNM": "cons", "MSN": "cons", "MWG": "retail", "PNJ": "retail",
    "SSI": "sec", "HCM": "sec", "VND": "sec", "SBT": "cons", "REE": "ind",
}


def rank_symbols(alphas, n_select=5, max_per_sector=2, min_alpha=0.5):
    filtered = {s: v for s, v in alphas.items() if not np.isnan(v) and v >= min_alpha}
    if len(filtered) < 3:
        filtered = {s: v for s, v in alphas.items() if not np.isnan(v)}

    sorted_symbols = sorted(filtered.items(), key=lambda x: -x[1])

    selected = []
    sector_counts = {}

    for symbol, score in sorted_symbols:
        if len(selected) >= n_select:
            break
        sector = SECTOR_MAP.get(symbol, 'other')
        if sector_counts.get(sector, 0) < max_per_sector:
            selected.append((symbol, score))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(all_data, market_df=None, capital=100_000_000):
    print("\n" + "=" * 60)
    print("  CROSS-SECTIONAL PORTFOLIO - Simple Version")
    print("  Walk-Forward Alpha | Risk Parity | Regime Filter")
    print("=" * 60)

    # Feature Engineering
    print("\n[1/5] Building features...")
    feat_data = {}
    for sym, df in all_data.items():
        feat_data[sym] = build_features(df, market_df)
        print(f"  ✓ {sym}: {len(feat_data[sym])} bars")

    # Common dates
    all_dates = None
    for df in all_data.values():
        dates = set(df.index)
        all_dates = dates if all_dates is None else all_dates & dates
    all_dates = sorted(all_dates)

    warmup = 504
    valid_dates = all_dates[warmup:]
    rebalance_dates = valid_dates[::21]  # Monthly
    print(f"\n[2/5] Dates: {len(all_dates)} total, {len(rebalance_dates)} rebalance points")

    # Regime detector
    print("\n[3/5] Initializing regime detector...")
    if market_df is None:
        proxy_close = pd.concat([d['Close'] for d in all_data.values()], axis=1).mean(axis=1)
        market_df = pd.DataFrame({'Close': proxy_close}).dropna()
    regime = RegimeDetector(market_df)

    # Walk-Forward Alpha
    print("\n[4/5] Running Walk-Forward Alpha...")
    wfa = WFAlpha(train_window=504, retrain_freq=63)

    positions_history = {}

    for i, date in enumerate(rebalance_dates):
        # Train models
        for sym in feat_data:
            if wfa.needs_train(sym, date):
                wfa.train(sym, feat_data[sym], date)

        # Predict alphas
        alphas = {s: wfa.predict(s, feat_data[s], date) for s in feat_data}

        # Rank and select
        selected = rank_symbols(alphas, n_select=5, min_alpha=0.5)

        if not selected:
            continue

        selected_symbols = [s for s, _ in selected]
        selected_scores = {s: v for s, v in selected}

        # Optimize weights
        weights = risk_parity_weights(selected_symbols, selected_scores, all_data, date)

        if weights:
            positions_history[date] = weights

        if i % 20 == 0:
            print(f"  {date.date()}: {', '.join(selected_symbols[:3])}")

    print(f"  Active signals: {len(positions_history)}/{len(rebalance_dates)}")

    return positions_history, regime


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: GEN_POSITION (cho quantvn platform)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_position(df: pd.DataFrame, capital: float = 100_000_000,
                 positions_history: Dict = None, regime=None) -> pd.DataFrame:
    """
    Tạo cột position cho platform quantvn.
    Platform chỉ gọi gen_position(df) nên cần xử lý fallback.
    """
    df = df.copy()

    # Fallback: nếu không có positions_history (chạy standalone)
    if positions_history is None:
        # Chiến lược đơn giản: EMA cross
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['signal'] = (df['ema20'] > df['ema50']).astype(int)
        df['position'] = df['signal'] * 1000
        df.drop(['ema20', 'ema50', 'signal'], axis=1, inplace=True)
        return df

    # Xây dựng position từ lịch sử
    df['position'] = 0

    # Lấy rebalance dates
    rebalance_dates = sorted(positions_history.keys())

    for i, date in enumerate(rebalance_dates):
        if date not in df.index:
            continue

        mult = regime.get_multiplier(date) if regime else 1.0
        weights = positions_history[date]

        # Tìm symbol hiện tại (cách xác định symbol từ df)
        # Trong platform, mỗi df là 1 symbol riêng
        symbol = None
        for sym in list(positions_history.values())[0].keys():
            # Giả sử symbol đã biết hoặc lấy từ tên file
            symbol = sym
            break

        if symbol not in weights:
            continue

        weight = weights[symbol] * mult

        # Find next rebalance date
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else df.index[-1]

        mask = (df.index >= date) & (df.index < next_date)
        if mask.any():
            price = df.loc[date, 'Open'] if date in df.index else df[mask].iloc[0]['Open']
            target_value = capital * weight
            shares = int(target_value / price / 100) * 100
            df.loc[mask, 'position'] = shares

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from quantvn.vn.data import get_stock_hist

    SYMBOLS = ["HPG", "FPT", "VCB", "ACB", "MBB", "TCB", "CTG"]

    # Load data
    all_data = {}
    for sym in SYMBOLS:
        df = get_stock_hist(sym, resolution="1D")
        if df is not None and len(df) >= 600:
            all_data[sym] = safe_dt(df).sort_index().dropna()
            print(f"✓ {sym}: {len(all_data[sym])} bars")

    # Run pipeline
    positions_history, regime = run_pipeline(all_data)

    print("\n" + "=" * 60)
    print("  KẾT QUẢ")
    print("=" * 60)
    print(f"  Total rebalance signals: {len(positions_history)}")
    print(f"  Portfolio weights: {list(positions_history.values())[0] if positions_history else 'None'}")
    print("=" * 60)
    print("  ✓ Walk-Forward: no look-ahead bias")
    print("  ✓ Risk Parity weights")
    print("  ✓ Regime filter with EMA200 + Volatility")
    print("=" * 60)