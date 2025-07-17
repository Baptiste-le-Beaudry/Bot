"""
Technical Indicators Module
Bibliothèque complète d'indicateurs techniques optimisés pour le trading algorithmique.
Inclut des indicateurs classiques et avancés pour HFT, arbitrage et ML.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from numba import jit, njit
import talib
from scipy import stats, signal
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuration du logger
import logging
logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Résultat d'un indicateur avec métadonnées"""
    value: Union[pd.Series, pd.DataFrame]
    signals: Optional[pd.Series] = None  # Buy/Sell signals
    levels: Optional[Dict[str, float]] = None  # Support/Resistance levels
    metadata: Optional[Dict[str, Any]] = None  # Additional info


class TechnicalIndicators:
    """
    Collection complète d'indicateurs techniques optimisés.
    Utilise NumPy/Pandas pour la vectorisation et Numba pour l'accélération.
    """
    
    def __init__(self, use_talib: bool = True, cache_results: bool = True):
        """
        Initialisation
        
        Args:
            use_talib: Utiliser TA-Lib si disponible (plus rapide)
            cache_results: Mettre en cache les calculs coûteux
        """
        self.use_talib = use_talib and self._check_talib()
        self.cache = {} if cache_results else None
        
    def _check_talib(self) -> bool:
        """Vérifier si TA-Lib est disponible"""
        try:
            import talib
            return True
        except ImportError:
            logger.warning("TA-Lib non disponible, utilisation des implémentations Python")
            return False
    
    # ============= INDICATEURS DE TENDANCE =============
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if self.use_talib:
            return pd.Series(talib.SMA(data.values, timeperiod=period), index=data.index)
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        if self.use_talib:
            return pd.Series(talib.EMA(data.values, timeperiod=period), index=data.index)
        return data.ewm(span=period, adjust=False).mean()
    
    def wma(self, data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        if self.use_talib:
            return pd.Series(talib.WMA(data.values, timeperiod=period), index=data.index)
        
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def hma(self, data: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average - Réduit le lag"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.wma(data, half_period)
        wma_full = self.wma(data, period)
        
        raw_hma = 2 * wma_half - wma_full
        return self.wma(raw_hma, sqrt_period)
    
    def kama(self, data: pd.Series, period: int = 10, 
            fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """Kaufman's Adaptive Moving Average"""
        if self.use_talib:
            return pd.Series(talib.KAMA(data.values, timeperiod=period), index=data.index)
        
        # Efficiency Ratio
        direction = (data - data.shift(period)).abs()
        volatility = data.diff().abs().rolling(period).sum()
        er = direction / volatility.replace(0, 1)
        
        # Smoothing Constants
        fastest_sc = 2 / (fast_ema + 1)
        slowest_sc = 2 / (slow_ema + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = pd.Series(index=data.index, dtype=float)
        kama.iloc[period-1] = data.iloc[period-1]
        
        for i in range(period, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def tema(self, data: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average"""
        if self.use_talib:
            return pd.Series(talib.TEMA(data.values, timeperiod=period), index=data.index)
        
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        
        return 3 * ema1 - 3 * ema2 + ema3
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
            signal: int = 9) -> Dict[str, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        if self.use_talib:
            macd_line, signal_line, histogram = talib.MACD(
                data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return {
                'macd': pd.Series(macd_line, index=data.index),
                'signal': pd.Series(signal_line, index=data.index),
                'histogram': pd.Series(histogram, index=data.index)
            }
        
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
        """SuperTrend Indicator"""
        # ATR
        atr = self.atr(high, low, close, period)
        
        # Basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr
        
        # Initialize
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(period, len(close)):
            # Upper band
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Lower band
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # Direction
            if i == period:
                direction.iloc[i] = 1 if close.iloc[i] <= upper_band.iloc[i] else -1
            else:
                if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                    direction.iloc[i] = 1 if close.iloc[i] <= upper_band.iloc[i] else -1
                else:
                    direction.iloc[i] = -1 if close.iloc[i] >= lower_band.iloc[i] else 1
            
            # SuperTrend
            supertrend.iloc[i] = upper_band.iloc[i] if direction.iloc[i] == 1 else lower_band.iloc[i]
        
        # Generate signals
        signals = direction.diff()
        
        return IndicatorResult(
            value=supertrend,
            signals=signals,
            metadata={'direction': direction}
        )
    
    # ============= INDICATEURS DE MOMENTUM =============
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if self.use_talib:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  fastk_period: int = 14, slowk_period: int = 3, 
                  slowd_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        if self.use_talib:
            k, d = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period
            )
            return {
                'k': pd.Series(k, index=close.index),
                'd': pd.Series(d, index=close.index)
            }
        
        # Fast %K
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        
        fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        
        # Slow %K
        slow_k = fast_k.rolling(window=slowk_period).mean()
        
        # Slow %D
        slow_d = slow_k.rolling(window=slowd_period).mean()
        
        return {'k': slow_k, 'd': slow_d}
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        """Williams %R"""
        if self.use_talib:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, 
                                       timeperiod=period), index=close.index)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
           period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        if self.use_talib:
            return pd.Series(talib.CCI(high.values, low.values, close.values,
                                     timeperiod=period), index=close.index)
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        return (typical_price - sma_tp) / (0.015 * mean_deviation).replace(0, 1)
    
    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
           volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        if self.use_talib:
            return pd.Series(talib.MFI(high.values, low.values, close.values,
                                     volume.values, timeperiod=period), index=close.index)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = pd.Series(0, index=close.index)
        negative_flow = pd.Series(0, index=close.index)
        
        # Separate positive and negative flows
        mask = typical_price > typical_price.shift(1)
        positive_flow[mask] = money_flow[mask]
        negative_flow[~mask] = money_flow[~mask]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfr = positive_mf / negative_mf.replace(0, 1e-10)
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def ultimate_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series,
                          period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator"""
        if self.use_talib:
            return pd.Series(talib.ULTOSC(high.values, low.values, close.values,
                                        timeperiod1=period1, timeperiod2=period2,
                                        timeperiod3=period3), index=close.index)
        
        # Buying Pressure
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Average for each period
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        # Ultimate Oscillator
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        
        return uo
    
    def tsi(self, data: pd.Series, long_period: int = 25, 
           short_period: int = 13) -> pd.Series:
        """True Strength Index"""
        momentum = data.diff()
        
        # Double smoothed momentum
        ema_long = self.ema(momentum, long_period)
        double_smoothed_momentum = self.ema(ema_long, short_period)
        
        # Double smoothed absolute momentum
        abs_momentum = momentum.abs()
        ema_long_abs = self.ema(abs_momentum, long_period)
        double_smoothed_abs_momentum = self.ema(ema_long_abs, short_period)
        
        # TSI
        tsi = 100 * (double_smoothed_momentum / double_smoothed_abs_momentum.replace(0, 1))
        
        return tsi
    
    # ============= INDICATEURS DE VOLATILITÉ =============
    
    def bollinger_bands(self, data: pd.Series, period: int = 20,
                       std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        if self.use_talib:
            upper, middle, lower = talib.BBANDS(
                data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            return {
                'upper': pd.Series(upper, index=data.index),
                'middle': pd.Series(middle, index=data.index),
                'lower': pd.Series(lower, index=data.index)
            }
        
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + std_dev * std,
            'middle': sma,
            'lower': sma - std_dev * std
        }
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, multiplier: float = 2.0,
                        ma_type: str = 'ema') -> Dict[str, pd.Series]:
        """Keltner Channels"""
        # Middle line
        if ma_type == 'ema':
            middle = self.ema(close, period)
        else:
            middle = self.sma(close, period)
        
        # ATR for bands
        atr = self.atr(high, low, close, period)
        
        return {
            'upper': middle + multiplier * atr,
            'middle': middle,
            'lower': middle - multiplier * atr
        }
    
    def donchian_channels(self, high: pd.Series, low: pd.Series,
                         period: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channels"""
        return {
            'upper': high.rolling(window=period).max(),
            'middle': (high.rolling(window=period).max() + 
                      low.rolling(window=period).min()) / 2,
            'lower': low.rolling(window=period).min()
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
           period: int = 14) -> pd.Series:
        """Average True Range"""
        if self.use_talib:
            return pd.Series(talib.ATR(high.values, low.values, close.values,
                                     timeperiod=period), index=close.index)
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # ATR
        return tr.ewm(alpha=1/period, adjust=False).mean()
    
    def natr(self, high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """Normalized Average True Range (%)"""
        if self.use_talib:
            return pd.Series(talib.NATR(high.values, low.values, close.values,
                                      timeperiod=period), index=close.index)
        
        atr = self.atr(high, low, close, period)
        return (atr / close) * 100
    
    def historical_volatility(self, data: pd.Series, period: int = 20,
                            annualize: bool = True) -> pd.Series:
        """Historical Volatility (Standard Deviation of Returns)"""
        returns = np.log(data / data.shift(1))
        hvol = returns.rolling(window=period).std()
        
        if annualize:
            hvol *= np.sqrt(252)  # Assuming 252 trading days
        
        return hvol * 100  # Convert to percentage
    
    def garman_klass_volatility(self, high: pd.Series, low: pd.Series,
                              close: pd.Series, open_: pd.Series,
                              period: int = 20, annualize: bool = True) -> pd.Series:
        """Garman-Klass Volatility Estimator"""
        # Components
        hl_ratio = np.log(high / low) ** 2
        co_ratio = np.log(close / open_) ** 2
        
        # Garman-Klass formula
        gk = np.sqrt((0.5 * hl_ratio - 0.39 * co_ratio).rolling(period).mean())
        
        if annualize:
            gk *= np.sqrt(252)
        
        return gk * 100
    
    def chaikin_volatility(self, high: pd.Series, low: pd.Series,
                         period: int = 10, rate_of_change: int = 10) -> pd.Series:
        """Chaikin Volatility"""
        hl_spread = high - low
        ema_spread = self.ema(hl_spread, period)
        
        return ((ema_spread - ema_spread.shift(rate_of_change)) / 
                ema_spread.shift(rate_of_change)) * 100
    
    # ============= INDICATEURS DE VOLUME =============
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        if self.use_talib:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        
        return (np.sign(close.diff()) * volume).cumsum()
    
    def cmf(self, high: pd.Series, low: pd.Series, close: pd.Series,
           volume: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        mfv = ((close - low) - (high - close)) / (high - low).replace(0, 1) * volume
        return mfv.rolling(period).sum() / volume.rolling(period).sum()
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        
        if period:
            return (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
        else:
            return (typical_price * volume).cumsum() / volume.cumsum()
    
    def vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend"""
        return (volume * close.pct_change()).cumsum()
    
    def force_index(self, close: pd.Series, volume: pd.Series,
                   period: int = 13) -> pd.Series:
        """Force Index"""
        fi = close.diff() * volume
        return self.ema(fi, period)
    
    def ease_of_movement(self, high: pd.Series, low: pd.Series,
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """Ease of Movement"""
        distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        emv_ratio = distance_moved / (volume / 1e6) / (high - low)
        
        return self.sma(emv_ratio, period)
    
    def accumulation_distribution(self, high: pd.Series, low: pd.Series,
                                close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        if self.use_talib:
            return pd.Series(talib.AD(high.values, low.values, close.values,
                                    volume.values), index=close.index)
        
        clv = ((close - low) - (high - close)) / (high - low).replace(0, 1)
        return (clv * volume).cumsum()
    
    # ============= INDICATEURS DE STRUCTURE DE MARCHÉ =============
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series,
                    method: str = 'standard') -> Dict[str, pd.Series]:
        """Pivot Points (Standard, Fibonacci, Woodie, Camarilla)"""
        
        if method == 'standard':
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
        elif method == 'fibonacci':
            pivot = (high + low + close) / 3
            range_ = high - low
            r1 = pivot + 0.382 * range_
            r2 = pivot + 0.618 * range_
            r3 = pivot + 1.000 * range_
            s1 = pivot - 0.382 * range_
            s2 = pivot - 0.618 * range_
            s3 = pivot - 1.000 * range_
            
        elif method == 'woodie':
            pivot = (high + low + 2 * close) / 4
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = r1 + (high - low)
            s3 = s1 - (high - low)
            
        elif method == 'camarilla':
            pivot = (high + low + close) / 3
            range_ = high - low
            r1 = close + range_ * 1.1 / 12
            r2 = close + range_ * 1.1 / 6
            r3 = close + range_ * 1.1 / 4
            s1 = close - range_ * 1.1 / 12
            s2 = close - range_ * 1.1 / 6
            s3 = close - range_ * 1.1 / 4
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def fibonacci_retracements(self, data: pd.Series, 
                             lookback: int = 100) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        recent_data = data.tail(lookback)
        high = recent_data.max()
        low = recent_data.min()
        diff = high - low
        
        levels = {
            '0.0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.500,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100.0%': low
        }
        
        return levels
    
    def support_resistance_levels(self, high: pd.Series, low: pd.Series,
                                close: pd.Series, lookback: int = 100,
                                num_levels: int = 5) -> Dict[str, List[float]]:
        """Detect Support and Resistance Levels using multiple methods"""
        
        # Method 1: Local extrema
        highs = high.rolling(10).max() == high
        lows = low.rolling(10).min() == low
        
        resistance_levels = high[highs].tail(lookback).nlargest(num_levels).tolist()
        support_levels = low[lows].tail(lookback).nsmallest(num_levels).tolist()
        
        # Method 2: High volume areas
        # (Would need volume data for proper implementation)
        
        # Method 3: Round numbers
        current_price = close.iloc[-1]
        round_levels = []
        
        for i in range(-5, 6):
            if i != 0:
                level = round(current_price, -int(np.log10(current_price))) + i * 10 ** int(np.log10(current_price) - 1)
                round_levels.append(level)
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels),
            'round_levels': sorted(round_levels)
        }
    
    # ============= INDICATEURS AVANCÉS ET CUSTOM =============
    
    def ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
                      displacement: int = 26) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(senkou_b).max() + 
                         low.rolling(senkou_b).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def market_profile(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 20,
                      value_area_pct: float = 0.70) -> Dict[str, Any]:
        """Simplified Market Profile"""
        # Price levels
        price_range = np.linspace(low.min(), high.max(), 50)
        volume_profile = pd.Series(0, index=price_range)
        
        # Accumulate volume at price levels
        for i in range(len(close)):
            prices_in_range = price_range[
                (price_range >= low.iloc[i]) & (price_range <= high.iloc[i])
            ]
            volume_at_level = volume.iloc[i] / len(prices_in_range) if len(prices_in_range) > 0 else 0
            volume_profile[prices_in_range] += volume_at_level
        
        # Point of Control (POC)
        poc = volume_profile.idxmax()
        
        # Value Area
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum_volume = sorted_profile.cumsum()
        value_area_volume = cumsum_volume[cumsum_volume <= cumsum_volume.iloc[-1] * value_area_pct]
        
        value_area_prices = value_area_volume.index
        vah = value_area_prices.max()  # Value Area High
        val = value_area_prices.min()  # Value Area Low
        
        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'volume_profile': volume_profile
        }
    
    def vwap_bands(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, period: int = 20,
                  std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """VWAP with Bands"""
        vwap = self.vwap(high, low, close, volume, period)
        typical_price = (high + low + close) / 3
        
        # Calculate deviation
        squared_diff = ((typical_price - vwap) ** 2 * volume).rolling(period).sum()
        variance = squared_diff / volume.rolling(period).sum()
        std = np.sqrt(variance)
        
        return {
            'vwap': vwap,
            'upper': vwap + std_dev * std,
            'lower': vwap - std_dev * std
        }
    
    def order_flow_imbalance(self, bid_volume: pd.Series, 
                           ask_volume: pd.Series) -> pd.Series:
        """Order Flow Imbalance (for HFT)"""
        total_volume = bid_volume + ask_volume
        ofi = (bid_volume - ask_volume) / total_volume.replace(0, 1)
        return ofi
    
    def microstructure_features(self, bid: pd.Series, ask: pd.Series,
                              bid_size: pd.Series, ask_size: pd.Series,
                              trades: pd.DataFrame) -> Dict[str, pd.Series]:
        """Advanced Microstructure Features for HFT"""
        features = {}
        
        # Spread metrics
        features['spread'] = ask - bid
        features['spread_pct'] = (ask - bid) / ((ask + bid) / 2) * 100
        features['mid_price'] = (bid + ask) / 2
        
        # Depth imbalance
        features['depth_imbalance'] = (bid_size - ask_size) / (bid_size + ask_size)
        
        # Micro price (size-weighted)
        features['micro_price'] = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        
        # Price impact (if trades data available)
        if trades is not None and not trades.empty:
            # Simplified Kyle's Lambda
            trade_signs = np.sign(trades['price'].diff())
            volume_imbalance = (trades['volume'] * trade_signs).rolling(20).sum()
            price_change = trades['price'].pct_change().rolling(20).sum()
            
            features['kyle_lambda'] = price_change / volume_imbalance.replace(0, 1)
        
        return features
    
    def hurst_exponent(self, data: pd.Series, min_lag: int = 2,
                      max_lag: int = 20) -> float:
        """Hurst Exponent - Measure of trending vs mean-reverting"""
        lags = range(min_lag, max_lag)
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences
            differences = data.diff(lag).dropna()
            tau.append(np.std(differences))
        
        # Linear regression of log(tau) vs log(lag)
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        
        return slope  # Hurst exponent
    
    def fractal_dimension(self, data: pd.Series, 
                         method: str = 'box_counting') -> float:
        """Fractal Dimension - Complexity measure"""
        if method == 'box_counting':
            # Simplified box-counting method
            n = len(data)
            scales = [2**i for i in range(1, int(np.log2(n)))]
            counts = []
            
            normalized_data = (data - data.min()) / (data.max() - data.min())
            
            for scale in scales:
                boxes = int(n / scale)
                box_counts = 0
                
                for i in range(boxes):
                    start = i * scale
                    end = min((i + 1) * scale, n)
                    
                    if end > start:
                        segment = normalized_data.iloc[start:end]
                        if not segment.empty:
                            box_counts += 1
                
                counts.append(box_counts)
            
            # Linear regression
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            
            return -slope
        
        return 1.5  # Default
    
    def entropy_features(self, data: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Entropy-based features for regime detection"""
        features = {}
        
        # Shannon entropy
        def shannon_entropy(x):
            if len(x) < 2:
                return 0
            
            # Discretize using quantiles
            bins = pd.qcut(x, q=10, duplicates='drop', labels=False)
            counts = pd.value_counts(bins)
            probabilities = counts / len(x)
            
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        features['shannon_entropy'] = data.rolling(period).apply(shannon_entropy)
        
        # Sample entropy (simplified)
        features['sample_entropy'] = data.rolling(period).std() / data.rolling(period).mean().abs()
        
        return features
    
    # ============= PATTERN RECOGNITION =============
    
    def candlestick_patterns(self, open_: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Detect common candlestick patterns"""
        patterns = {}
        
        if self.use_talib:
            # Doji
            patterns['doji'] = pd.Series(
                talib.CDLDOJI(open_.values, high.values, low.values, close.values),
                index=close.index
            )
            
            # Hammer
            patterns['hammer'] = pd.Series(
                talib.CDLHAMMER(open_.values, high.values, low.values, close.values),
                index=close.index
            )
            
            # Engulfing
            patterns['engulfing'] = pd.Series(
                talib.CDLENGULFING(open_.values, high.values, low.values, close.values),
                index=close.index
            )
            
            # Morning Star
            patterns['morning_star'] = pd.Series(
                talib.CDLMORNINGSTAR(open_.values, high.values, low.values, close.values),
                index=close.index
            )
            
            # Evening Star
            patterns['evening_star'] = pd.Series(
                talib.CDLEVENINGSTAR(open_.values, high.values, low.values, close.values),
                index=close.index
            )
        else:
            # Manual pattern detection (simplified)
            body = close - open_
            body_size = body.abs()
            upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
            lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low
            
            # Doji: small body
            patterns['doji'] = (body_size < body_size.rolling(20).mean() * 0.1).astype(int)
            
            # Hammer: small body at top, long lower shadow
            patterns['hammer'] = (
                (body_size < body_size.rolling(20).mean() * 0.3) &
                (lower_shadow > body_size * 2) &
                (upper_shadow < body_size * 0.5)
            ).astype(int)
        
        return patterns
    
    def chart_patterns(self, data: pd.Series, 
                      min_pattern_length: int = 20) -> Dict[str, Any]:
        """Detect chart patterns (Head & Shoulders, Triangles, etc.)"""
        patterns_detected = {}
        
        # Find local peaks and troughs
        peaks = self._find_peaks(data, distance=5)
        troughs = self._find_troughs(data, distance=5)
        
        # Head and Shoulders
        hs_patterns = self._detect_head_shoulders(data, peaks, troughs)
        if hs_patterns:
            patterns_detected['head_shoulders'] = hs_patterns
        
        # Triangles
        triangle_patterns = self._detect_triangles(data, peaks, troughs)
        if triangle_patterns:
            patterns_detected['triangles'] = triangle_patterns
        
        # Double Top/Bottom
        double_patterns = self._detect_double_patterns(data, peaks, troughs)
        if double_patterns:
            patterns_detected['double_patterns'] = double_patterns
        
        return patterns_detected
    
    def _find_peaks(self, data: pd.Series, distance: int = 10) -> np.ndarray:
        """Find local peaks in data"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data.values, distance=distance)
        return peaks
    
    def _find_troughs(self, data: pd.Series, distance: int = 10) -> np.ndarray:
        """Find local troughs in data"""
        from scipy.signal import find_peaks
        troughs, _ = find_peaks(-data.values, distance=distance)
        return troughs
    
    def _detect_head_shoulders(self, data: pd.Series, peaks: np.ndarray,
                             troughs: np.ndarray) -> List[Dict]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        # Need at least 3 peaks and 2 troughs
        if len(peaks) < 3 or len(troughs) < 2:
            return patterns
        
        # Check recent formations
        for i in range(len(peaks) - 2):
            # Get three consecutive peaks
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            
            # Head should be higher than shoulders
            if data.iloc[p2] > data.iloc[p1] and data.iloc[p2] > data.iloc[p3]:
                # Shoulders should be roughly equal (within 2%)
                shoulder_ratio = data.iloc[p3] / data.iloc[p1]
                if 0.98 <= shoulder_ratio <= 1.02:
                    # Find neckline (troughs between peaks)
                    neckline_points = troughs[(troughs > p1) & (troughs < p3)]
                    if len(neckline_points) >= 2:
                        neckline_level = data.iloc[neckline_points].mean()
                        
                        patterns.append({
                            'type': 'head_and_shoulders',
                            'peaks': [p1, p2, p3],
                            'neckline': neckline_level,
                            'target': 2 * neckline_level - data.iloc[p2],
                            'confidence': 0.7
                        })
        
        return patterns
    
    def _detect_triangles(self, data: pd.Series, peaks: np.ndarray,
                        troughs: np.ndarray) -> List[Dict]:
        """Detect Triangle patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Get recent peaks and troughs
        recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
        recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            # Calculate trendlines
            peak_slope = np.polyfit(recent_peaks, data.iloc[recent_peaks].values, 1)[0]
            trough_slope = np.polyfit(recent_troughs, data.iloc[recent_troughs].values, 1)[0]
            
            # Classify triangle type
            if abs(peak_slope) < 0.001 and trough_slope > 0.001:
                triangle_type = 'ascending'
            elif peak_slope < -0.001 and abs(trough_slope) < 0.001:
                triangle_type = 'descending'
            elif peak_slope < -0.001 and trough_slope > 0.001:
                triangle_type = 'symmetric'
            else:
                triangle_type = None
            
            if triangle_type:
                patterns.append({
                    'type': f'{triangle_type}_triangle',
                    'peak_slope': peak_slope,
                    'trough_slope': trough_slope,
                    'apex': data.index[-1] + pd.Timedelta(days=10),  # Estimated
                    'confidence': 0.6
                })
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.Series, peaks: np.ndarray,
                              troughs: np.ndarray) -> List[Dict]:
        """Detect Double Top/Bottom patterns"""
        patterns = []
        
        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i+1]
                
                # Peaks should be similar height (within 1%)
                if 0.99 <= data.iloc[p2] / data.iloc[p1] <= 1.01:
                    # Valley between peaks
                    valley = troughs[(troughs > p1) & (troughs < p2)]
                    if len(valley) > 0:
                        patterns.append({
                            'type': 'double_top',
                            'peaks': [p1, p2],
                            'valley': valley[0],
                            'target': 2 * data.iloc[valley[0]] - data.iloc[p1],
                            'confidence': 0.7
                        })
        
        # Double Bottom
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                t1, t2 = troughs[i], troughs[i+1]
                
                # Troughs should be similar depth (within 1%)
                if 0.99 <= data.iloc[t2] / data.iloc[t1] <= 1.01:
                    # Peak between troughs
                    peak = peaks[(peaks > t1) & (peaks < t2)]
                    if len(peak) > 0:
                        patterns.append({
                            'type': 'double_bottom',
                            'troughs': [t1, t2],
                            'peak': peak[0],
                            'target': 2 * data.iloc[peak[0]] - data.iloc[t1],
                            'confidence': 0.7
                        })
        
        return patterns
    
    # ============= COMPOSITE INDICATORS =============
    
    def trend_strength_index(self, data: pd.Series, period: int = 20) -> pd.Series:
        """Custom Trend Strength Index combining multiple indicators"""
        # Components
        sma_slope = self.sma(data, period).diff(5) / self.sma(data, period) * 100
        adx = self.adx(data, data, data, period)  # Simplified
        
        # Price position relative to moving averages
        above_sma20 = (data > self.sma(data, 20)).astype(int)
        above_sma50 = (data > self.sma(data, 50)).astype(int)
        
        # Combine
        tsi = (sma_slope * 0.4 + adx / 100 * 0.4 + 
               (above_sma20 + above_sma50) / 2 * 0.2)
        
        return tsi
    
    def momentum_oscillator_suite(self, high: pd.Series, low: pd.Series,
                                close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Suite of momentum oscillators with composite score"""
        oscillators = {}
        
        # Individual oscillators
        oscillators['rsi'] = self.rsi(close, 14)
        oscillators['stoch_k'] = self.stochastic(high, low, close)['k']
        oscillators['cci'] = self.cci(high, low, close)
        oscillators['mfi'] = self.mfi(high, low, close, volume)
        oscillators['williams_r'] = self.williams_r(high, low, close)
        
        # Normalize to 0-100 scale
        oscillators['cci_norm'] = 50 + oscillators['cci'].clip(-100, 100) / 2
        oscillators['williams_r_norm'] = -oscillators['williams_r']
        
        # Composite score
        composite = (
            oscillators['rsi'] * 0.3 +
            oscillators['stoch_k'] * 0.2 +
            oscillators['cci_norm'] * 0.2 +
            oscillators['mfi'] * 0.2 +
            oscillators['williams_r_norm'] * 0.1
        )
        
        oscillators['composite'] = composite
        
        return oscillators
    
    def volatility_regime_indicator(self, high: pd.Series, low: pd.Series,
                                  close: pd.Series, lookback: int = 20) -> pd.Series:
        """Classify volatility regime"""
        # Multiple volatility measures
        atr = self.atr(high, low, close, lookback)
        hist_vol = self.historical_volatility(close, lookback, annualize=False)
        
        # Bollinger Band width
        bb = self.bollinger_bands(close, lookback)
        bb_width = (bb['upper'] - bb['lower']) / bb['middle']
        
        # Normalize and combine
        atr_zscore = (atr - atr.rolling(100).mean()) / atr.rolling(100).std()
        vol_zscore = (hist_vol - hist_vol.rolling(100).mean()) / hist_vol.rolling(100).std()
        bb_zscore = (bb_width - bb_width.rolling(100).mean()) / bb_width.rolling(100).std()
        
        # Composite volatility score
        vol_score = (atr_zscore + vol_zscore + bb_zscore) / 3
        
        # Classify regime
        regime = pd.Series(index=close.index, dtype=str)
        regime[vol_score < -1] = 'very_low'
        regime[(vol_score >= -1) & (vol_score < 0)] = 'low'
        regime[(vol_score >= 0) & (vol_score < 1)] = 'normal'
        regime[(vol_score >= 1) & (vol_score < 2)] = 'high'
        regime[vol_score >= 2] = 'extreme'
        
        return regime
    
    def market_strength_indicator(self, high: pd.Series, low: pd.Series,
                                close: pd.Series, volume: pd.Series) -> pd.Series:
        """Comprehensive market strength indicator"""
        # Price strength
        price_roc = close.pct_change(20) * 100
        
        # Volume strength
        volume_ratio = volume / volume.rolling(20).mean()
        
        # Breadth (simplified - would need market breadth data)
        advances = (close > close.shift(1)).rolling(20).sum()
        breadth = advances / 20
        
        # Trend persistence
        consecutive_ups = (close > close.shift(1)).astype(int)
        trend_persistence = consecutive_ups.rolling(20).sum() / 20
        
        # Combine
        strength = (
            price_roc.clip(-10, 10) / 10 * 0.3 +
            volume_ratio.clip(0, 2) / 2 * 0.2 +
            breadth * 0.25 +
            trend_persistence * 0.25
        ) * 100
        
        return strength
    
    # ============= HELPER METHODS =============
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
           period: int = 14) -> pd.Series:
        """Average Directional Index"""
        if self.use_talib:
            return pd.Series(talib.ADX(high.values, low.values, close.values,
                                     timeperiod=period), index=close.index)
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
        
        # Calculate TR
        tr = self.atr(high, low, close, 1)
        
        # Smooth
        plus_di = 100 * self.ema(plus_dm, period) / self.ema(tr, period)
        minus_di = 100 * self.ema(minus_dm, period) / self.ema(tr, period)
        
        # Calculate ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = self.ema(dx, period)
        
        return adx
    
    def all_indicators(self, ohlcv: pd.DataFrame, 
                      indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate all or selected indicators"""
        
        # Extract columns
        open_ = ohlcv['open']
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv.get('volume', pd.Series(1, index=ohlcv.index))
        
        result = ohlcv.copy()
        
        # Default indicators
        if indicators is None:
            indicators = [
                'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_20',
                'atr_14', 'adx_14', 'obv', 'mfi_14'
            ]
        
        # Calculate requested indicators
        for indicator in indicators:
            if indicator == 'sma_20':
                result['sma_20'] = self.sma(close, 20)
            elif indicator == 'ema_20':
                result['ema_20'] = self.ema(close, 20)
            elif indicator == 'rsi_14':
                result['rsi_14'] = self.rsi(close, 14)
            elif indicator == 'macd':
                macd_result = self.macd(close)
                result['macd'] = macd_result['macd']
                result['macd_signal'] = macd_result['signal']
                result['macd_hist'] = macd_result['histogram']
            elif indicator == 'bb_20':
                bb_result = self.bollinger_bands(close)
                result['bb_upper'] = bb_result['upper']
                result['bb_middle'] = bb_result['middle']
                result['bb_lower'] = bb_result['lower']
            elif indicator == 'atr_14':
                result['atr_14'] = self.atr(high, low, close, 14)
            elif indicator == 'adx_14':
                result['adx_14'] = self.adx(high, low, close, 14)
            elif indicator == 'obv':
                result['obv'] = self.obv(close, volume)
            elif indicator == 'mfi_14':
                result['mfi_14'] = self.mfi(high, low, close, volume, 14)
        
        return result


# Optimized functions using Numba for performance
@njit
def rolling_window_numba(array: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling window calculation"""
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


@njit
def sma_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized SMA calculation"""
    result = np.empty_like(values)
    result[:period-1] = np.nan
    
    for i in range(period-1, len(values)):
        result[i] = np.mean(values[i-period+1:i+1])
    
    return result


@njit
def ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized EMA calculation"""
    result = np.empty_like(values)
    result[:period-1] = np.nan
    result[period-1] = np.mean(values[:period])
    
    multiplier = 2.0 / (period + 1)
    
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
    
    return result


# Convenience functions
def calculate_indicators(data: pd.DataFrame, indicators: List[str],
                       use_parallel: bool = False) -> pd.DataFrame:
    """
    Calculate multiple indicators efficiently
    
    Args:
        data: DataFrame with OHLCV data
        indicators: List of indicator names
        use_parallel: Use parallel processing
        
    Returns:
        DataFrame with indicators added
    """
    ti = TechnicalIndicators()
    
    if use_parallel and len(indicators) > 5:
        # Parallel processing for many indicators
        from concurrent.futures import ProcessPoolExecutor
        
        def calc_indicator(ind_name):
            return ind_name, ti.all_indicators(data, [ind_name])
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(calc_indicator, indicators))
        
        # Merge results
        result_df = data.copy()
        for ind_name, ind_df in results:
            new_cols = [col for col in ind_df.columns if col not in data.columns]
            result_df[new_cols] = ind_df[new_cols]
        
        return result_df
    else:
        # Sequential processing
        return ti.all_indicators(data, indicators)


# Example usage
def main():
    """Example usage of technical indicators"""
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    # Simulate price data
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': close,
        'volume': np.random.exponential(1000000, n)
    }, index=dates)
    
    # Initialize indicators
    ti = TechnicalIndicators()
    
    # Calculate various indicators
    print("Calculating indicators...")
    
    # Trend indicators
    data['sma_20'] = ti.sma(data['close'], 20)
    data['ema_20'] = ti.ema(data['close'], 20)
    
    # Momentum
    data['rsi_14'] = ti.rsi(data['close'], 14)
    
    # Volatility
    bb = ti.bollinger_bands(data['close'])
    data['bb_upper'] = bb['upper']
    data['bb_lower'] = bb['lower']
    
    # Volume
    data['obv'] = ti.obv(data['close'], data['volume'])
    
    # Advanced
    supertrend = ti.supertrend(data['high'], data['low'], data['close'])
    data['supertrend'] = supertrend.value
    data['supertrend_signal'] = supertrend.signals
    
    print(f"\nIndicators calculated for {len(data)} periods")
    print(f"Sample (last 5 rows):\n{data.tail()}")
    
    # Detect patterns
    patterns = ti.chart_patterns(data['close'])
    print(f"\nDetected patterns: {list(patterns.keys())}")


if __name__ == "__main__":
    main()