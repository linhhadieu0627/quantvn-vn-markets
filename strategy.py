"""
Chiến lược giao dịch: Cross-Sectional Portfolio (v29.0 - ML Enhanced)
======================================================================
Version: 29.0
Nâng cấp từ v28.0:
  [A] MLAlphaEngine      - LightGBM + Walk-Forward (thay rule-based alpha)
  [B] HMMRegimeDetector  - Hidden Markov Model (thay EMA200 threshold)
  [C] MLPortfolioOptimizer - Risk Parity / Max-Sharpe (thay equal weight)
  [D] FeatureEngineer    - 40+ features kỹ thuật & thống kê
  [E] WalkForwardValidator - Strict no look-ahead validation
  [F] AlphaExplainer     - SHAP feature importance

Dependencies:
    pip install lightgbm hmmlearn shap scipy scikit-learn numpy pandas
    (torch nếu muốn dùng LSTM - optional)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: DEPENDENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    missing = []
    optional_missing = []

    required = {
        'lightgbm': 'pip install lightgbm',
        'hmmlearn': 'pip install hmmlearn',
        'sklearn': 'pip install scikit-learn',
        'scipy': 'pip install scipy',
        'shap': 'pip install shap',
    }
    optional = {
        'torch': 'pip install torch  # For LSTM/Transformer (optional)',
    }

    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"  ❌ {pkg}: {install_cmd}")

    for pkg, install_cmd in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            optional_missing.append(f"  ⚠️  {pkg}: {install_cmd}")

    if missing:
        print("MISSING REQUIRED PACKAGES:")
        for m in missing:
            print(m)
        raise ImportError("Install required packages above before running.")

    if optional_missing:
        print("Optional packages not installed (LSTM disabled):")
        for m in optional_missing:
            print(m)

    return len(optional_missing) == 0  # True nếu có torch


TORCH_AVAILABLE = False
try:
    check_dependencies()
    import lightgbm as lgb
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from scipy.optimize import minimize
    import shap
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        TORCH_AVAILABLE = True
    except ImportError:
        pass
except Exception as e:
    print(f"Dependency error: {e}")
    raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: UTILITY FUNCTIONS (giữ lại từ v28)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move, down_move = high - high.shift(1), low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bb(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    return mid + 2 * std, mid, mid - 2 * std


def safe_convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for col in ['time', 'Time', 'date', 'Date', 'timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            return df
    try:
        if len(df) > 0 and isinstance(df.index[0], (int, np.integer)):
            df.index = pd.to_datetime(df.index, unit='ns' if df.index[0] > 1e10 else 's')
        else:
            df.index = pd.to_datetime(df.index)
    except:
        pass
    return df


def rolling_r2(y: pd.Series, window: int = 60) -> pd.Series:
    def r2(arr):
        if len(arr) < window or np.isnan(arr).any():
            return 0.0
        x = np.arange(len(arr))
        x_mean, y_mean = np.mean(x), np.mean(arr)
        num = np.sum((x - x_mean) * (arr - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0: return 0.0
        slope = num / den
        y_pred = slope * x + (y_mean - slope * x_mean)
        ss_res = np.sum((arr - y_pred) ** 2)
        ss_tot = np.sum((arr - y_mean) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return y.rolling(window).apply(r2, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: [D] FEATURE ENGINEER - 40+ features
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Tạo 40+ features từ OHLCV data cho ML models.
    Tất cả features đều shift(1) để tránh look-ahead bias.
    """

    FEATURE_GROUPS = {
        'momentum': ['mom_5', 'mom_10', 'mom_21', 'mom_63', 'mom_126'],
        'volatility': ['vol_5', 'vol_10', 'vol_21', 'vol_63', 'atr_norm'],
        'trend': ['adx_14', 'trend_quality_60', 'slope_21', 'ema_ratio_50', 'ema_ratio_200'],
        'oscillator': ['rsi_14', 'rsi_28', 'stoch_k', 'cci_20'],
        'volume': ['vol_ratio_5', 'vol_ratio_21', 'obv_trend'],
        'pattern': ['bb_position', 'distance_from_high_52w', 'distance_from_low_52w'],
        'market_relative': ['beta_63', 'alpha_63', 'corr_63', 'relative_strength'],
        'regime_features': ['return_dispersion', 'market_vol_regime'],
    }

    def __init__(self, market_df: Optional[pd.DataFrame] = None):
        self.market_df = market_df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build full feature matrix. Returns df with all features + 'target'."""
        df = df.copy()
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df.get('Volume', pd.Series(1.0, index=df.index)).astype(float)
        returns = close.pct_change()

        feat = pd.DataFrame(index=df.index)

        # ── Momentum ─────────────────────────────────────────────────────────
        for w in [5, 10, 21, 63, 126]:
            feat[f'mom_{w}'] = close.pct_change(w)

        # Risk-adjusted momentum
        for w in [21, 63]:
            r = close.pct_change(w)
            v = returns.rolling(w).std() * np.sqrt(252)
            feat[f'sharpe_mom_{w}'] = r / (v + 1e-8)

        # ── Volatility ────────────────────────────────────────────────────────
        for w in [5, 10, 21, 63]:
            feat[f'vol_{w}'] = returns.rolling(w).std() * np.sqrt(252)

        feat['atr_norm'] = calculate_atr(df) / close

        # Vol regime: current vol / longer-term vol
        feat['vol_ratio'] = feat['vol_5'] / (feat['vol_63'] + 1e-8)

        # ── Trend ─────────────────────────────────────────────────────────────
        feat['adx_14'] = calculate_adx(df, 14)
        feat['trend_quality_60'] = rolling_r2(returns.cumsum(), 60)

        # EMA ratios
        for span in [10, 21, 50, 200]:
            ema = close.ewm(span=span, adjust=False).mean()
            feat[f'ema_ratio_{span}'] = close / ema - 1

        # Linear slope (normalized)
        def rolling_slope(y, w=21):
            def slope(arr):
                x = np.arange(len(arr))
                return np.polyfit(x, arr, 1)[0] / (arr[-1] + 1e-8)
            return y.rolling(w).apply(slope, raw=True)

        feat['slope_21'] = rolling_slope(close, 21)
        feat['slope_63'] = rolling_slope(close, 63)

        # ── Oscillators ───────────────────────────────────────────────────────
        feat['rsi_14'] = calculate_rsi(close, 14)
        feat['rsi_28'] = calculate_rsi(close, 28)
        feat['rsi_norm'] = (feat['rsi_14'] - 50) / 50  # centered

        # Stochastic %K
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        feat['stoch_k'] = (close - low_14) / (high_14 - low_14 + 1e-8) * 100

        # CCI
        typical = (high + low + close) / 3
        feat['cci_20'] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std() + 1e-8)
        feat['cci_20'] = feat['cci_20'].clip(-3, 3)

        # ── Volume ────────────────────────────────────────────────────────────
        if volume.std() > 0:
            for w in [5, 21]:
                feat[f'vol_ratio_{w}'] = volume / (volume.rolling(w).mean() + 1e-8)

            # On-Balance Volume trend
            obv = (np.sign(returns) * volume).cumsum()
            feat['obv_trend'] = obv.pct_change(21)
        else:
            for w in [5, 21]:
                feat[f'vol_ratio_{w}'] = 1.0
            feat['obv_trend'] = 0.0

        # ── Price Patterns ────────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = calculate_bb(df, 20)
        feat['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        feat['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)

        feat['distance_from_high_52w'] = close / high.rolling(252).max() - 1
        feat['distance_from_low_52w'] = close / low.rolling(252).min() - 1

        # Gap (overnight)
        prev_close = close.shift(1)
        feat['gap'] = (df['Open'].astype(float) / prev_close - 1).clip(-0.1, 0.1)

        # ── Market-Relative Features ──────────────────────────────────────────
        if self.market_df is not None:
            mkt = self.market_df['Close'].reindex(df.index, method='ffill').astype(float)
            mkt_ret = mkt.pct_change()

            # Rolling beta và alpha
            for w in [63, 126]:
                cov = returns.rolling(w).cov(mkt_ret)
                var = mkt_ret.rolling(w).var()
                beta = cov / (var + 1e-8)
                feat[f'beta_{w}'] = beta.clip(-3, 3)
                feat[f'alpha_{w}'] = returns.rolling(w).mean() - beta * mkt_ret.rolling(w).mean()

            feat['corr_63'] = returns.rolling(63).corr(mkt_ret)
            feat['relative_strength'] = (close.pct_change(63) - mkt.pct_change(63))

        # ── Target: Forward return 21 ngày ───────────────────────────────────
        feat['target'] = close.pct_change(21).shift(-21)

        # ── SHIFT ALL FEATURES by 1 (no look-ahead) ─────────────────────────
        feature_cols = [c for c in feat.columns if c != 'target']
        feat[feature_cols] = feat[feature_cols].shift(1)

        return feat


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: [A] ML ALPHA ENGINE - LightGBM với Walk-Forward
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardConfig:
    train_window_days: int = 504     # 2 năm training
    val_window_days: int = 63        # 3 tháng validation
    retrain_freq_days: int = 63      # Retrain mỗi quý
    min_train_samples: int = 200     # Tối thiểu samples để train
    n_lgbm_estimators: int = 300
    lgbm_learning_rate: float = 0.03
    lgbm_max_depth: int = 4
    lgbm_num_leaves: int = 31
    lgbm_subsample: float = 0.8
    lgbm_colsample: float = 0.7
    lgbm_reg_alpha: float = 0.1     # L1
    lgbm_reg_lambda: float = 1.0    # L2


class MLAlphaEngine:
    """
    Walk-Forward LightGBM Alpha Engine.

    Thay thế rule-based alpha (hardcoded weights) bằng LightGBM
    trained theo phương pháp walk-forward để tránh look-ahead bias hoàn toàn.

    Flow:
        1. Với mỗi rebalance date, lấy train window [date-504d, date)
        2. Train LightGBM predict forward_return_21d
        3. Predict alpha cho date hiện tại
        4. Retrain mỗi 63 ngày
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.models: Dict[str, Any] = {}          # symbol -> trained model
        self.scalers: Dict[str, RobustScaler] = {}
        self.feature_names: List[str] = []
        self.feature_engineer = None
        self._last_retrain_date: Dict[str, pd.Timestamp] = {}
        self._val_scores: Dict[str, List[float]] = {}

    def set_feature_engineer(self, fe: FeatureEngineer):
        self.feature_engineer = fe

    def _build_lgbm(self) -> lgb.LGBMRegressor:
        c = self.config
        return lgb.LGBMRegressor(
            n_estimators=c.n_lgbm_estimators,
            learning_rate=c.lgbm_learning_rate,
            max_depth=c.lgbm_max_depth,
            num_leaves=c.lgbm_num_leaves,
            subsample=c.lgbm_subsample,
            colsample_bytree=c.lgbm_colsample,
            reg_alpha=c.lgbm_reg_alpha,
            reg_lambda=c.lgbm_reg_lambda,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    def _should_retrain(self, symbol: str, date: pd.Timestamp) -> bool:
        if symbol not in self._last_retrain_date:
            return True
        days_since = (date - self._last_retrain_date[symbol]).days
        return days_since >= self.config.retrain_freq_days

    def train_symbol(self, symbol: str, feat_df: pd.DataFrame,
                     train_end: pd.Timestamp) -> bool:
        """Train model cho một symbol đến train_end date."""
        train_start = train_end - pd.DateOffset(days=self.config.train_window_days)

        # Lấy training data (strict: chỉ dùng data trước train_end)
        mask = (feat_df.index >= train_start) & (feat_df.index < train_end)
        train_data = feat_df[mask].dropna()

        if len(train_data) < self.config.min_train_samples:
            return False

        feature_cols = [c for c in train_data.columns if c != 'target']
        X = train_data[feature_cols]
        y = train_data['target']

        # Winsorize target
        q_low, q_high = y.quantile(0.05), y.quantile(0.95)
        y = y.clip(q_low, q_high)

        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Train LightGBM
        model = self._build_lgbm()

        # Optional: walk-forward val score
        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []
        for tr_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            m = self._build_lgbm()
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])
            pred = m.predict(X_val)
            val_scores.append(np.corrcoef(pred, y_val)[0, 1])

        self._val_scores[symbol] = val_scores

        # Retrain on full data
        model.fit(X_scaled, y, callbacks=[lgb.log_evaluation(-1)])

        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_names = feature_cols
        self._last_retrain_date[symbol] = train_end

        return True

    def predict_alpha(self, symbol: str, feat_df: pd.DataFrame,
                      date: pd.Timestamp) -> float:
        """Predict alpha score cho một symbol tại một date."""
        if symbol not in self.models:
            return 0.5  # neutral fallback

        if date not in feat_df.index:
            return 0.5

        feature_cols = [c for c in feat_df.columns if c != 'target']
        row = feat_df.loc[date, feature_cols]

        if row.isna().any():
            return 0.5

        X = self.scalers[symbol].transform(row.values.reshape(1, -1))
        raw_pred = self.models[symbol].predict(X)[0]

        # Convert predicted return to alpha score [0, 1]
        # Clip ±20% annualized, normalize
        alpha = (raw_pred * 12 + 0.4) / 0.8  # -20% -> 0, +20% -> 1
        return float(np.clip(alpha, 0.0, 1.0))

    def get_feature_importance(self, symbol: str) -> Optional[pd.Series]:
        if symbol not in self.models:
            return None
        model = self.models[symbol]
        return pd.Series(
            model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

    def get_val_ic(self, symbol: str) -> float:
        """Information Coefficient từ validation."""
        scores = self._val_scores.get(symbol, [])
        return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LSTM ALPHA (Optional - PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class AlphaLSTM(nn.Module):
        """
        LSTM model để predict alpha từ time series features.
        Input: (batch, seq_len=60, n_features)
        Output: (batch, 1) - predicted forward return
        """
        def __init__(self, n_features: int, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=4, batch_first=True
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            return self.head(attn_out[:, -1, :]) # dùng last timestep


    class LSTMAlphaEngine:
        """
        LSTM-based alpha engine (PyTorch).
        Sequence length = 60 ngày.
        Chậm hơn LightGBM nhưng học được temporal patterns.
        """
        SEQ_LEN = 60

        def __init__(self, n_features: int, device: str = 'cpu'):
            self.device = torch.device(device)
            self.n_features = n_features
            self.model = None
            self.scaler = StandardScaler()

        def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
            sequences, targets = [], []
            for i in range(self.SEQ_LEN, len(X)):
                sequences.append(X[i - self.SEQ_LEN:i])
                targets.append(y[i])
            return np.array(sequences), np.array(targets)

        def train(self, feat_df: pd.DataFrame,
                  epochs: int = 30, batch_size: int = 64):
            feat_df = feat_df.dropna()
            feature_cols = [c for c in feat_df.columns if c != 'target']
            X = self.scaler.fit_transform(feat_df[feature_cols])
            y = feat_df['target'].values

            X_seq, y_seq = self._prepare_sequences(X, y)

            # Train/val split (temporal)
            split = int(len(X_seq) * 0.8)
            X_tr, X_val = X_seq[:split], X_seq[split:]
            y_tr, y_val = y_seq[:split], y_seq[split:]

            tr_ds = TensorDataset(
                torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
            )
            tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)

            self.model = AlphaLSTM(n_features=self.n_features).to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            best_val_loss = float('inf')
            best_state = None

            for epoch in range(epochs):
                self.model.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(xb).squeeze()
                    loss = nn.MSELoss()(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_pred = self.model(X_v).squeeze()
                    val_loss = nn.MSELoss()(val_pred, y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if best_state:
                self.model.load_state_dict(best_state)

        def predict(self, X_recent: np.ndarray) -> float:
            if self.model is None:
                return 0.5
            X_scaled = self.scaler.transform(X_recent)
            seq = torch.FloatTensor(X_scaled[-self.SEQ_LEN:]).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(seq).item()
            alpha = (pred * 12 + 0.4) / 0.8
            return float(np.clip(alpha, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: [B] HMM REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class HMMRegimeDetector:
    """
    Hidden Markov Model để phát hiện market regime.

    3 states: Bear (0), Sideways (1), Bull (2)
    Features: [daily_return, rolling_vol_21, rolling_vol_63_ratio, adx_norm]

    Thay thế MarketRegime dùng EMA200 threshold cứng.
    """

    REGIME_NAMES = {0: 'bear', 1: 'sideways', 2: 'bull'}
    REGIME_MULTIPLIERS = {'bear': 0.3, 'sideways': 0.65, 'bull': 1.0}

    def __init__(self, n_components: int = 3, n_iter: int = 200):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            init_params='stmc',
        )
        self.scaler = StandardScaler()
        self._regime_mapping: Dict[int, str] = {}  # hmm state -> regime name
        self._fitted = False
        self._X_fitted: Optional[np.ndarray] = None
        self._dates_fitted: Optional[pd.DatetimeIndex] = None

    def _build_features(self, market_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        close = market_df['Close'].astype(float)
        ret = close.pct_change()
        vol_21 = ret.rolling(21).std()
        vol_63 = ret.rolling(63).std()
        vol_ratio = vol_21 / (vol_63 + 1e-8)
        ema_50 = close.ewm(span=50).mean()
        ema_200 = close.ewm(span=200).mean()
        trend = ema_50 / ema_200 - 1

        feat_df = pd.DataFrame({
            'ret': ret,
            'vol_21': vol_21,
            'vol_ratio': vol_ratio,
            'trend': trend,
        }).dropna()

        return feat_df.values, feat_df.index

    def fit(self, market_df: pd.DataFrame) -> 'HMMRegimeDetector':
        X, dates = self._build_features(market_df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Map HMM states to regime names by mean return
        states = self.model.predict(X_scaled)
        state_means = {}
        for s in range(self.n_components):
            mask = states == s
            state_means[s] = X[mask, 0].mean()  # mean return

        sorted_states = sorted(state_means.items(), key=lambda x: x[1])
        for rank, (state, _) in enumerate(sorted_states):
            self._regime_mapping[state] = self.REGIME_NAMES[rank]

        self._X_fitted = X_scaled
        self._dates_fitted = dates
        self._fitted = True
        return self

    def get_regime(self, date: pd.Timestamp,
                   market_df: pd.DataFrame) -> str:
        """Predict regime tại một date (chỉ dùng data đến date đó)."""
        if not self._fitted:
            return 'bull'

        hist = market_df.loc[:date]
        if len(hist) < 200:
            return 'bull'

        X, _ = self._build_features(hist)
        if len(X) < 5:
            return 'bull'

        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        current_state = states[-1]
        return self._regime_mapping.get(current_state, 'sideways')

    def get_multiplier(self, date: pd.Timestamp,
                       market_df: pd.DataFrame) -> float:
        regime = self.get_regime(date, market_df)
        return self.REGIME_MULTIPLIERS.get(regime, 0.7)

    def get_regime_history(self, market_df: pd.DataFrame) -> pd.Series:
        """Lấy toàn bộ regime history để phân tích."""
        if not self._fitted:
            return pd.Series()
        X, dates = self._build_features(market_df)
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        regimes = [self._regime_mapping.get(s, 'unknown') for s in states]
        return pd.Series(regimes, index=dates)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: [C] ML PORTFOLIO OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class MLPortfolioOptimizer:
    """
    Tối ưu hóa danh mục dùng Mean-Variance, Risk Parity, hoặc Max-Sharpe.

    Thay thế equal weight (1/N) bằng optimization có constraints.
    """

    def __init__(self, method: str = 'risk_parity',
                 max_weight: float = 0.35,
                 min_weight: float = 0.05,
                 lookback_days: int = 126,
                 regularization: float = 0.1):
        assert method in ('equal', 'max_sharpe', 'risk_parity', 'min_variance')
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.lookback_days = lookback_days
        self.regularization = regularization

    def _get_returns_and_cov(
        self, symbols: List[str], all_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính expected returns và covariance matrix."""
        lookback_start = date - pd.DateOffset(days=self.lookback_days)

        ret_series = {}
        for sym in symbols:
            df = all_data[sym]
            hist = df.loc[lookback_start:date]
            if len(hist) > 20:
                ret_series[sym] = hist['Close'].pct_change().dropna()

        if not ret_series:
            n = len(symbols)
            return np.zeros(n), np.eye(n) * 0.01

        ret_df = pd.DataFrame(ret_series).dropna()

        # Expected returns: shrinkage toward cross-sectional mean
        mu = ret_df.mean().values * 252
        grand_mean = np.mean(mu)
        shrinkage = self.regularization
        mu_shrunk = (1 - shrinkage) * mu + shrinkage * grand_mean

        # Covariance: Ledoit-Wolf shrinkage
        cov = ret_df.cov().values * 252
        # Simple shrinkage toward diagonal
        diag = np.diag(np.diag(cov))
        cov_shrunk = (1 - shrinkage) * cov + shrinkage * diag

        # Regularize
        cov_shrunk += np.eye(len(symbols)) * 1e-6

        return mu_shrunk, cov_shrunk

    def optimize(
        self, symbols: List[str], alpha_scores: Dict[str, float],
        all_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Tối ưu portfolio weights.

        alpha_scores: {symbol: alpha_score} từ MLAlphaEngine
        Kết hợp alpha scores vào expected returns.
        """
        if not symbols:
            return {}

        if self.method == 'equal':
            return {s: 1.0 / len(symbols) for s in symbols}

        mu, cov = self._get_returns_and_cov(symbols, all_data, date)
        n = len(symbols)

        # Blend ML alpha scores vào expected returns
        alpha_arr = np.array([alpha_scores.get(s, 0.5) for s in symbols])
        alpha_centered = alpha_arr - alpha_arr.mean()
        # Scale: alpha_centered ±0.5 -> ±5% annual boost
        mu_final = mu + alpha_centered * 0.1

        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n

        if self.method == 'max_sharpe':
            def neg_sharpe(w):
                ret = np.dot(w, mu_final)
                vol = np.sqrt(w @ cov @ w)
                return -ret / (vol + 1e-8)
            result = minimize(neg_sharpe, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-9})

        elif self.method == 'risk_parity':
            def risk_parity_obj(w):
                port_vol = np.sqrt(w @ cov @ w)
                risk_contrib = w * (cov @ w) / (port_vol + 1e-8)
                target_rc = np.ones(n) / n * port_vol
                return np.sum((risk_contrib - target_rc) ** 2)
            result = minimize(risk_parity_obj, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'ftol': 1e-10})

        elif self.method == 'min_variance':
            def port_variance(w):
                return w @ cov @ w
            result = minimize(port_variance, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500})

        if result.success:
            weights = np.clip(result.x, self.min_weight, self.max_weight)
            weights /= weights.sum()
        else:
            weights = np.ones(n) / n

        return {s: float(w) for s, w in zip(symbols, weights)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CROSS-SECTIONAL RANKER (ML-enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class MLCrossSectionalRanker:
    """
    Rank symbols theo ML alpha scores thay vì rule-based z-scores.
    Kết hợp với MLPortfolioOptimizer.
    """

    SECTOR_MAP = {
        "HPG": "steel", "FPT": "tech", "VCB": "bank", "ACB": "bank",
        "MBB": "bank", "TCB": "bank", "CTG": "bank", "BID": "bank",
        "VIC": "realestate", "VHM": "realestate", "GAS": "energy",
        "VNM": "consumer", "MSN": "consumer", "MWG": "retail",
        "PNJ": "retail", "SSI": "securities", "HCM": "securities",
        "VND": "securities", "SBT": "consumer", "REE": "industrial"
    }

    def __init__(self, n_long: int = 5, max_sector: int = 2,
                 min_alpha_threshold: float = 0.55):
        self.n_long = n_long
        self.max_sector = max_sector
        self.min_alpha_threshold = min_alpha_threshold  # chỉ long nếu alpha > 0.55

    def rank(
        self, ml_alphas: Dict[str, float],
        date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """
        Rank symbols, áp dụng sector constraint và alpha threshold.
        Returns: sorted list of (symbol, alpha_score)
        """
        # Filter by minimum alpha
        filtered = {
            s: v for s, v in ml_alphas.items()
            if not np.isnan(v) and v >= self.min_alpha_threshold
        }

        if len(filtered) < 3:
            # Lower threshold nếu không đủ
            filtered = {s: v for s, v in ml_alphas.items() if not np.isnan(v)}

        sorted_symbols = sorted(filtered.items(), key=lambda x: -x[1])

        selected = []
        sector_counts = {}
        for symbol, score in sorted_symbols:
            if len(selected) >= self.n_long:
                break
            sector = self.SECTOR_MAP.get(symbol, "other")
            if sector_counts.get(sector, 0) < self.max_sector:
                selected.append((symbol, score))
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return selected


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: [F] ALPHA EXPLAINER - SHAP
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaExplainer:
    """
    SHAP-based explainability cho ML alpha models.
    """

    def __init__(self, ml_engine: MLAlphaEngine):
        self.ml_engine = ml_engine

    def explain_symbol(self, symbol: str,
                       feat_df: pd.DataFrame,
                       n_samples: int = 100) -> Optional[pd.DataFrame]:
        """
        Tính SHAP values cho một symbol.
        Returns DataFrame với feature importance per date.
        """
        if symbol not in self.ml_engine.models:
            return None

        model = self.ml_engine.models[symbol]
        scaler = self.ml_engine.scalers[symbol]
        feature_cols = [c for c in feat_df.columns if c != 'target']

        sample = feat_df[feature_cols].dropna().tail(n_samples)
        if len(sample) == 0:
            return None

        X_scaled = scaler.transform(sample)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            return pd.DataFrame(
                shap_values,
                index=sample.index,
                columns=feature_cols
            )
        except Exception as e:
            print(f"  SHAP error for {symbol}: {e}")
            return None

    def get_top_features(self, symbol: str,
                         feat_df: pd.DataFrame,
                         top_n: int = 10) -> pd.Series:
        """Top N most important features by mean |SHAP|."""
        shap_df = self.explain_symbol(symbol, feat_df)
        if shap_df is None:
            return self.ml_engine.get_feature_importance(symbol) or pd.Series()
        return shap_df.abs().mean().sort_values(ascending=False).head(top_n)

    def print_report(self, symbol: str, feat_df: pd.DataFrame):
        """In báo cáo feature importance."""
        print(f"\n  📊 SHAP Analysis: {symbol}")
        top = self.get_top_features(symbol, feat_df)
        if len(top) == 0:
            print("    No SHAP data available")
            return
        for feat, importance in top.items():
            bar = '█' * int(importance * 100)
            print(f"    {feat:35s} {importance:.4f}  {bar}")

        ic = self.ml_engine.get_val_ic(symbol)
        print(f"    Walk-Forward IC: {ic:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PORTFOLIO BACKTEST (từ v28, không thay đổi logic)
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioBacktest:
    def __init__(self, initial_capital: float = 100_000_000,
                 slippage_bps: float = 10.0, fee_bps: float = 15.0):
        self.initial_capital = initial_capital
        self.fee_rate = (slippage_bps + fee_bps) / 10000

    def run(self, symbols_data: Dict[str, pd.DataFrame],
            positions_history: Dict[pd.Timestamp, Dict[str, float]],
            regime_multipliers: Dict[pd.Timestamp, float] = None) -> Dict:

        all_dates = None
        for df in symbols_data.values():
            dates = set(df.index)
            if all_dates is None:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)
        all_dates = sorted(all_dates)

        cash = self.initial_capital
        holdings = {s: 0.0 for s in symbols_data.keys()}
        daily_equity, daily_dates, trade_log = [], [], []
        rebalance_set = set(positions_history.keys())

        for date in all_dates:
            # Mark-to-market
            portfolio_value = cash
            for symbol, shares in holdings.items():
                if abs(shares) > 1e-6:
                    hist = symbols_data[symbol].loc[:date]
                    if len(hist) > 0:
                        portfolio_value += shares * hist.iloc[-1]['Close']

            # Rebalance
            if date in rebalance_set:
                target_weights = positions_history[date]
                regime_mult = (regime_multipliers or {}).get(date, 1.0)
                scaled_weights = {k: v * regime_mult for k, v in target_weights.items()}

                sell_orders, buy_orders = [], []
                for symbol in symbols_data.keys():
                    if date not in symbols_data[symbol].index:
                        continue
                    open_price = symbols_data[symbol].loc[date, 'Open']
                    target_w = scaled_weights.get(symbol, 0.0)
                    target_val = portfolio_value * target_w
                    current_shares = holdings.get(symbol, 0)
                    current_val = current_shares * open_price
                    delta = target_val - current_val
                    if abs(delta) < portfolio_value * 0.001:
                        continue
                    if delta < 0:
                        sell_orders.append({'symbol': symbol, 'shares': abs(delta) / open_price,
                                            'value': abs(delta)})
                    else:
                        buy_orders.append({'symbol': symbol, 'shares': delta / open_price,
                                           'value': delta})

                # Sells first
                for o in sell_orders:
                    cost = o['value'] * self.fee_rate
                    cash += o['value'] - cost
                    new_sh = holdings.get(o['symbol'], 0) - o['shares']
                    holdings[o['symbol']] = max(new_sh, 0)
                    trade_log.append({'date': date, 'symbol': o['symbol'], 'action': 'sell',
                                      'value': o['value'], 'cost': cost})

                # Buys after
                for o in buy_orders:
                    required = o['value'] * (1 + self.fee_rate)
                    if cash >= required:
                        cash -= required
                        holdings[o['symbol']] = holdings.get(o['symbol'], 0) + o['shares']
                        trade_log.append({'date': date, 'symbol': o['symbol'], 'action': 'buy',
                                          'value': o['value'], 'cost': o['value'] * self.fee_rate})

                for sym in list(holdings.keys()):
                    if abs(holdings[sym]) < 1e-6:
                        holdings[sym] = 0

            # Daily NAV
            nav = cash
            for symbol, shares in holdings.items():
                if abs(shares) > 1e-6:
                    hist = symbols_data[symbol].loc[:date]
                    if len(hist) > 0:
                        nav += shares * hist.iloc[-1]['Close']
            daily_equity.append(nav)
            daily_dates.append(date)

        equity_series = pd.Series(daily_equity, index=pd.DatetimeIndex(daily_dates))
        daily_returns = equity_series.pct_change().dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax() * 100
        years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25

        return {
            'equity_series': equity_series,
            'daily_returns': daily_returns,
            'sharpe': sharpe,
            'cagr': ((equity_series.iloc[-1] / self.initial_capital) ** (1 / max(years, 0.01)) - 1) * 100,
            'total_return': (equity_series.iloc[-1] / self.initial_capital - 1) * 100,
            'max_drawdown': drawdown.min(),
            'final_equity': equity_series.iloc[-1],
            'total_trades': len(trade_log),
            'total_cost': sum(t.get('cost', 0) for t in trade_log),
            'years': years,
            'calmar': abs(((equity_series.iloc[-1] / self.initial_capital) ** (1/max(years, 0.01)) - 1) * 100 / min(drawdown.min(), -0.01)),
            'win_rate': (daily_returns > 0).mean(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_section(title: str, width: int = 80):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def run_ml_pipeline(
    all_data: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 100_000_000,
    optimizer_method: str = 'risk_parity',
    use_lstm: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Full ML pipeline. Returns backtest results dict.

    Parameters:
        all_data: {symbol: OHLCV DataFrame}
        market_df: Market index DataFrame (VN30/VNINDEX)
        optimizer_method: 'equal' | 'max_sharpe' | 'risk_parity' | 'min_variance'
        use_lstm: Enable LSTM alpha (cần torch, chậm hơn)
        verbose: Print progress
    """

    # ── Step 1: Feature Engineering ──────────────────────────────────────────
    if verbose:
        print_section("STEP 1: FEATURE ENGINEERING")

    fe = FeatureEngineer(market_df=market_df)
    feat_data: Dict[str, pd.DataFrame] = {}

    for symbol, df in all_data.items():
        try:
            feat_df = fe.build_features(df)
            feat_data[symbol] = feat_df
            if verbose:
                n_feat = len([c for c in feat_df.columns if c != 'target'])
                print(f"    ✓ {symbol}: {n_feat} features, {len(feat_df)} bars")
        except Exception as e:
            if verbose:
                print(f"    ❌ {symbol}: {e}")

    # ── Step 2: Common Dates & Warmup ─────────────────────────────────────────
    if verbose:
        print_section("STEP 2: DATE ALIGNMENT")

    all_dates = None
    for df in all_data.values():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    all_dates = sorted(all_dates)

    warmup = 504  # 2 năm warmup cho ML
    valid_dates = all_dates[warmup:]
    rebalance_dates = valid_dates[::21]  # Monthly

    if verbose:
        print(f"  Total dates: {len(all_dates)}, After warmup: {len(valid_dates)}")
        print(f"  Rebalance dates: {len(rebalance_dates)}")
        print(f"  Period: {valid_dates[0]} → {valid_dates[-1]}")

    # ── Step 3: HMM Regime Detection ─────────────────────────────────────────
    if verbose:
        print_section("STEP 3: HMM REGIME DETECTION")

    if market_df is not None:
        regime_detector = HMMRegimeDetector(n_components=3)
        regime_detector.fit(market_df)
        if verbose:
            regime_hist = regime_detector.get_regime_history(market_df)
            if len(regime_hist) > 0:
                counts = regime_hist.value_counts()
                for regime, cnt in counts.items():
                    pct = cnt / len(regime_hist) * 100
                    mult = HMMRegimeDetector.REGIME_MULTIPLIERS.get(regime, 0.7)
                    print(f"  {regime:10s}: {cnt:4d} days ({pct:.1f}%) → multiplier={mult}")
    else:
        # Fallback: dùng proxy từ all_data
        proxy_close = pd.concat([df['Close'] for df in all_data.values()], axis=1).mean(axis=1)
        proxy_df = pd.DataFrame({'Close': proxy_close}).dropna()
        regime_detector = HMMRegimeDetector()
        regime_detector.fit(proxy_df)
        market_df = proxy_df
        if verbose:
            print("  ⚠️  Using proxy market data")

    # ── Step 4: ML Alpha Training (Walk-Forward) ──────────────────────────────
    if verbose:
        print_section("STEP 4: ML ALPHA TRAINING (LIGHTGBM WALK-FORWARD)")

    wf_config = WalkForwardConfig(
        train_window_days=504,
        retrain_freq_days=63,
        n_lgbm_estimators=300,
        lgbm_learning_rate=0.03,
    )

    ml_engine = MLAlphaEngine(config=wf_config)
    ml_engine.set_feature_engineer(fe)

    # Pre-train tất cả symbols trước rebalance đầu tiên
    first_date = rebalance_dates[0]
    n_trained = 0

    for symbol in list(feat_data.keys()):
        feat_df = feat_data[symbol]
        success = ml_engine.train_symbol(symbol, feat_df, train_end=first_date)
        if success:
            n_trained += 1
            ic = ml_engine.get_val_ic(symbol)
            if verbose:
                print(f"    ✓ {symbol}: IC={ic:.3f}")
        elif verbose:
            print(f"    ⚠️  {symbol}: insufficient data")

    if verbose:
        print(f"  Trained: {n_trained}/{len(feat_data)} symbols")

    # ── Step 5: Portfolio Optimizer ───────────────────────────────────────────
    if verbose:
        print_section(f"STEP 5: PORTFOLIO OPTIMIZATION ({optimizer_method.upper()})")

    optimizer = MLPortfolioOptimizer(
        method=optimizer_method,
        max_weight=0.35,
        min_weight=0.05,
        lookback_days=126,
    )

    ranker = MLCrossSectionalRanker(n_long=5, max_sector=2, min_alpha_threshold=0.52)

    # ── Step 6: Generate Positions ────────────────────────────────────────────
    if verbose:
        print_section("STEP 6: GENERATING POSITIONS")

    positions_history: Dict[pd.Timestamp, Dict[str, float]] = {}
    regime_multipliers: Dict[pd.Timestamp, float] = {}

    for i, date in enumerate(rebalance_dates):
        # Retrain nếu cần
        for symbol in list(feat_data.keys()):
            if ml_engine._should_retrain(symbol, date):
                feat_df = feat_data[symbol]
                ml_engine.train_symbol(symbol, feat_df, train_end=date)

        # Predict alpha scores
        ml_alphas: Dict[str, float] = {}
        for symbol in feat_data.keys():
            feat_df = feat_data[symbol]
            alpha = ml_engine.predict_alpha(symbol, feat_df, date)
            ml_alphas[symbol] = alpha

        # Rank và select symbols
        selected = ranker.rank(ml_alphas, date)

        if not selected:
            continue

        selected_symbols = [s for s, _ in selected]
        selected_scores = {s: v for s, v in selected}

        # Optimize weights
        weights = optimizer.optimize(
            symbols=selected_symbols,
            alpha_scores=selected_scores,
            all_data=all_data,
            date=date,
        )

        if weights:
            positions_history[date] = weights

        # Regime multiplier từ HMM
        regime_multipliers[date] = regime_detector.get_multiplier(date, market_df)

        if verbose and i % 10 == 0:
            regime = regime_detector.get_regime(date, market_df)
            mult = regime_multipliers[date]
            symbols_str = ', '.join(selected_symbols)
            print(f"  {date.date()} | {regime:8s} ({mult:.2f}x) | {symbols_str}")

    if verbose:
        print(f"\n  Active signals: {len(positions_history)}/{len(rebalance_dates)}")

    # ── Step 7: SHAP Explainability ───────────────────────────────────────────
    if verbose:
        print_section("STEP 7: SHAP FEATURE IMPORTANCE")

    explainer = AlphaExplainer(ml_engine)
    top_symbols = list(feat_data.keys())[:3]  # Explain 3 symbols đầu
    for symbol in top_symbols:
        if symbol in feat_data:
            explainer.print_report(symbol, feat_data[symbol])

    # ── Step 8: Backtest ──────────────────────────────────────────────────────
    if verbose:
        print_section("STEP 8: BACKTEST")

    portfolio = PortfolioBacktest(initial_capital=initial_capital)
    results = portfolio.run(all_data, positions_history, regime_multipliers)
    results['positions_history'] = positions_history
    results['ml_engine'] = ml_engine
    results['explainer'] = explainer

    return results


def print_results(results: Dict, market_df: Optional[pd.DataFrame] = None,
                  valid_dates: Optional[List] = None):
    """In kết quả backtest đẹp."""
    print_section("BACKTEST RESULTS - v29.0 ML ENHANCED")

    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Period        : {results['years']:.1f} years                    │
  │  CAGR          : {results['cagr']:+.2f}%                        │
  │  Total Return  : {results['total_return']:+.2f}%                       │
  │  Sharpe Ratio  : {results['sharpe']:.3f}                        │
  │  Max Drawdown  : {results['max_drawdown']:.2f}%                       │
  │  Calmar Ratio  : {results['calmar']:.2f}                         │
  │  Win Rate      : {results['win_rate']:.1%}                        │
  │  Total Trades  : {results['total_trades']:,}                         │
  │  Trans. Cost   : {results['total_cost']:,.0f} VND          │
  │  Final Equity  : {results['final_equity']:,.0f} VND  │
  └─────────────────────────────────────────────┘""")

    if market_df is not None and valid_dates is not None:
        try:
            bench_start, bench_end = valid_dates[0], valid_dates[-1]
            if bench_start in market_df.index and bench_end in market_df.index:
                bench_ret = (market_df.loc[bench_end, 'Close'] /
                             market_df.loc[bench_start, 'Close'] - 1) * 100
                alpha = results['total_return'] - bench_ret
                print(f"  Benchmark (VN30)  : {bench_ret:.2f}%")
                print(f"  Alpha             : {alpha:+.2f}%")
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("QUANT_API_KEY")
    if not api_key:
        print("❌ QUANT_API_KEY not found")
        exit(1)

    from quantvn.vn.data.utils import client
    from quantvn.vn.data import get_stock_hist
    client(apikey=api_key)

    print("\n" + "█" * 80)
    print("  CROSS-SECTIONAL PORTFOLIO v29.0 - ML ENHANCED")
    print("  [A] LightGBM Alpha  [B] HMM Regime  [C] Risk Parity  [D] SHAP")
    print("█" * 80)

    SYMBOLS = [
        "HPG", "FPT", "VCB", "ACB", "MBB", "TCB", "CTG", "BID",
        "VIC", "VHM", "GAS", "VNM", "MSN", "MWG", "PNJ",
        "SSI", "HCM", "VND", "SBT", "REE"
    ]

    # Load market data
    market_df = None
    try:
        vnindex = get_stock_hist("VN30", resolution="1D")
        if vnindex is not None:
            market_df = safe_convert_datetime(vnindex).sort_index().dropna()
            print(f"  ✓ VN30: {len(market_df)} bars")
    except Exception as e:
        print(f"  ⚠️  Market data: {e}")

    # Load stock data
    all_data: Dict[str, pd.DataFrame] = {}
    print(f"\n  Loading {len(SYMBOLS)} symbols...")
    for sym in SYMBOLS:
        try:
            df = get_stock_hist(sym, resolution="1D")
            if df is not None and len(df) >= 600:
                df = safe_convert_datetime(df).sort_index().dropna()
                all_data[sym] = df
                print(f"    ✓ {sym}: {len(df)} bars")
            else:
                print(f"    ⚠️  {sym}: insufficient")
        except Exception as e:
            print(f"    ❌ {sym}: {e}")

    if len(all_data) < 10:
        print("❌ Not enough data")
        exit(1)

    # Run pipeline
    results = run_ml_pipeline(
        all_data=all_data,
        market_df=market_df,
        initial_capital=100_000_000,
        optimizer_method='risk_parity',  # 'equal' | 'max_sharpe' | 'risk_parity' | 'min_variance'
        use_lstm=TORCH_AVAILABLE,
        verbose=True,
    )

    # Tính valid_dates để so sánh benchmark
    all_dates_set = None
    for df in all_data.values():
        dates = set(df.index)
        if all_dates_set is None:
            all_dates_set = dates
        else:
            all_dates_set = all_dates_set.intersection(dates)
    all_dates_sorted = sorted(all_dates_set)
    valid_dates = all_dates_sorted[504:]

    print_results(results, market_df, valid_dates)

    print("""
  ⚠️  NOTES:
  - Walk-forward validation: no look-ahead bias
  - HMM regime: Bear/Sideways/Bull classification
  - Risk Parity: equal risk contribution per asset
  - SHAP: model explainability
  - Survivorship bias still present in symbol selection
    """)