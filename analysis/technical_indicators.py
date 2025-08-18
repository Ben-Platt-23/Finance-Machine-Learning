"""
Technical Analysis Module for Investment Decision Model

This module calculates various technical indicators that are crucial for making
investment decisions. Technical indicators help identify trends, momentum, 
volatility, and potential reversal points in stock prices.

Key Categories of Indicators:
1. Trend Indicators: Moving averages, MACD, ADX
2. Momentum Indicators: RSI, Stochastic, Williams %R
3. Volatility Indicators: Bollinger Bands, ATR
4. Volume Indicators: OBV, Volume SMA
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical analysis class that calculates various indicators
    used to analyze stock price movements and generate trading signals.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.indicators_cache = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given stock dataframe.
        
        This is the main function that orchestrates the calculation of all
        technical indicators. It takes OHLCV data and returns the same
        dataframe enriched with technical indicators.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with all technical indicators added as new columns
            
        Technical Indicators Calculated:
        - Moving Averages (SMA, EMA)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Stochastic Oscillator
        - Average True Range (ATR)
        - On-Balance Volume (OBV)
        - Williams %R
        - Commodity Channel Index (CCI)
        - Rate of Change (ROC)
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            result_df = df.copy()
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in result_df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            logger.info("Calculating technical indicators...")
            
            # 1. TREND INDICATORS
            # These help identify the overall direction of price movement
            result_df = self._add_trend_indicators(result_df)
            
            # 2. MOMENTUM INDICATORS  
            # These measure the speed and strength of price movements
            result_df = self._add_momentum_indicators(result_df)
            
            # 3. VOLATILITY INDICATORS
            # These measure the degree of price variation
            result_df = self._add_volatility_indicators(result_df)
            
            # 4. VOLUME INDICATORS
            # These analyze trading volume patterns
            result_df = self._add_volume_indicators(result_df)
            
            # 5. CUSTOM COMPOSITE INDICATORS
            # These combine multiple indicators for stronger signals
            result_df = self._add_composite_indicators(result_df)
            
            # 6. PRICE ACTION INDICATORS
            # These analyze candlestick patterns and price behavior
            result_df = self._add_price_action_indicators(result_df)
            
            logger.info(f"Successfully calculated {len(result_df.columns) - len(df.columns)} technical indicators")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-following indicators that help identify market direction.
        
        Trend indicators are lagging indicators that follow price movements
        and help confirm the direction of the trend.
        """
        # Simple Moving Averages - smooth out price data to identify trend direction
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)  # Short-term trend
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)  # Medium-term trend
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)  # Long-term trend
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)  # Very long-term trend
        
        # Exponential Moving Averages - give more weight to recent prices
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)  # Fast EMA
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)  # Slow EMA
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)  # Long EMA
        
        # MACD - Moving Average Convergence Divergence
        # Shows relationship between two moving averages and momentum
        df['macd'] = ta.trend.macd(df['close'])  # MACD line
        df['macd_signal'] = ta.trend.macd_signal(df['close'])  # Signal line
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])  # Histogram (MACD - Signal)
        
        # Average Directional Index (ADX) - measures trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)  # +DI
        df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)  # -DI
        
        # Parabolic SAR - trend-following indicator that provides stop-loss levels
        df['psar'] = ta.trend.psar_down(df['high'], df['low'], df['close'])
        
        # Trend signals based on moving average relationships
        df['ma_bullish_signal'] = (df['sma_10'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])
        df['ma_bearish_signal'] = (df['sma_10'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum oscillators that measure the speed of price changes.
        
        Momentum indicators are leading indicators that can signal potential
        reversals before they appear in the price.
        """
        # RSI - Relative Strength Index (0-100 scale)
        # Values above 70 suggest overbought, below 30 suggest oversold
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_overbought'] = df['rsi'] > 70  # Potential sell signal
        df['rsi_oversold'] = df['rsi'] < 30    # Potential buy signal
        
        # Stochastic Oscillator - compares closing price to price range
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        # Williams %R - momentum indicator similar to Stochastic
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # Rate of Change (ROC) - measures percentage change in price
        df['roc'] = ta.momentum.roc(df['close'], window=12)
        
        # Commodity Channel Index (CCI) - identifies cyclical trends
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # Money Flow Index (MFI) - volume-weighted RSI
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # Ultimate Oscillator - combines short, medium, and long-term price action
        df['ultimate_osc'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
        
        # Momentum signals
        df['momentum_bullish'] = (df['rsi'] > 50) & (df['stoch_k'] > 50) & (df['williams_r'] > -50)
        df['momentum_bearish'] = (df['rsi'] < 50) & (df['stoch_k'] < 50) & (df['williams_r'] < -50)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators that measure price dispersion.
        
        Volatility indicators help assess risk and identify potential
        breakout or breakdown points.
        """
        # Bollinger Bands - volatility bands around moving average
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=20)  # SMA 20
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        
        # Bollinger Band Width - measures volatility
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band Position - where price sits within the bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range (ATR) - measures volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Keltner Channels - trend-following bands
        df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['kc_middle'] = ta.volatility.keltner_channel_mband(df['high'], df['low'], df['close'], window=20)
        df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
        
        # Donchian Channels - breakout indicator
        df['dc_upper'] = ta.volatility.donchian_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['dc_lower'] = ta.volatility.donchian_channel_lband(df['high'], df['low'], df['close'], window=20)
        
        # Volatility signals
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.1)  # Low volatility
        df['bb_expansion'] = df['bb_width'] > df['bb_width'].rolling(20).quantile(0.9)  # High volatility
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators that analyze trading activity.
        
        Volume indicators help confirm price movements and identify
        potential reversals based on trading activity.
        """
        # On-Balance Volume (OBV) - cumulative volume based on price direction
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume SMA - smoothed volume to identify volume trends
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10)
        
        # Accumulation/Distribution Line - volume-price indicator
        df['ad_line'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow - measures buying/selling pressure
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
        
        # Volume Weighted Average Price (VWAP) approximation
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Volume signals
        df['volume_breakout'] = df['volume'] > (df['volume_sma'] * 1.5)  # High volume day
        df['volume_drying_up'] = df['volume'] < (df['volume_sma'] * 0.5)  # Low volume day
        
        return df
    
    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite indicators that combine multiple signals.
        
        These custom indicators synthesize information from multiple
        technical indicators to provide stronger, more reliable signals.
        """
        # Trend Strength Score (0-100)
        # Combines multiple trend indicators into a single score
        trend_components = []
        
        # MA alignment component
        ma_score = 0
        if 'sma_10' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            ma_score = ((df['sma_10'] > df['sma_20']).astype(int) + 
                       (df['sma_20'] > df['sma_50']).astype(int) + 
                       (df['close'] > df['sma_20']).astype(int)) / 3
        
        # MACD component
        macd_score = 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_score = (df['macd'] > df['macd_signal']).astype(int)
        
        # ADX component (trend strength)
        adx_score = 0
        if 'adx' in df.columns:
            adx_score = (df['adx'] > 25).astype(int)  # Strong trend threshold
        
        df['trend_strength'] = ((ma_score + macd_score + adx_score) / 3) * 100
        
        # Momentum Score (0-100)
        # Combines RSI, Stochastic, and Williams %R
        momentum_components = []
        
        if 'rsi' in df.columns:
            rsi_normalized = df['rsi'] / 100
            momentum_components.append(rsi_normalized)
        
        if 'stoch_k' in df.columns:
            stoch_normalized = df['stoch_k'] / 100
            momentum_components.append(stoch_normalized)
        
        if 'williams_r' in df.columns:
            williams_normalized = (df['williams_r'] + 100) / 100  # Convert from -100,0 to 0,1
            momentum_components.append(williams_normalized)
        
        if momentum_components:
            df['momentum_score'] = pd.concat(momentum_components, axis=1).mean(axis=1) * 100
        
        # Volatility Regime (Low/Medium/High)
        if 'atr' in df.columns:
            atr_percentile = df['atr'].rolling(window=50).rank(pct=True)
            df['volatility_regime'] = pd.cut(atr_percentile, 
                                           bins=[0, 0.33, 0.66, 1.0], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Overall Signal Strength (-100 to +100)
        # Positive values suggest bullish, negative suggest bearish
        trend_component = (df.get('trend_strength', 50) - 50) / 50  # Normalize to -1,1
        momentum_component = (df.get('momentum_score', 50) - 50) / 50  # Normalize to -1,1
        
        df['signal_strength'] = ((trend_component + momentum_component) / 2) * 100
        
        return df
    
    def _add_price_action_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action and candlestick pattern indicators.
        
        These indicators analyze the relationship between open, high,
        low, and close prices to identify potential reversal patterns.
        """
        # Basic candlestick properties
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Candlestick ratios
        df['body_to_range_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        
        # Candlestick types
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        df['is_doji'] = df['body_size'] < (df['total_range'] * 0.1)  # Small body relative to range
        
        # Gap analysis
        df['gap_up'] = df['low'] > df['high'].shift(1)
        df['gap_down'] = df['high'] < df['low'].shift(1)
        df['gap_size'] = np.where(df['gap_up'], 
                                 df['low'] - df['high'].shift(1),
                                 np.where(df['gap_down'], 
                                         df['low'].shift(1) - df['high'], 0))
        
        # Support and Resistance levels (simplified)
        # Find recent highs and lows as potential S/R levels
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()
        
        # Distance from S/R levels
        df['distance_from_resistance'] = (df['recent_high'] - df['close']) / df['close']
        df['distance_from_support'] = (df['close'] - df['recent_low']) / df['close']
        
        return df
    
    def get_trading_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on technical indicators.
        
        This function analyzes all the calculated technical indicators
        and provides actionable trading signals with confidence scores.
        
        Args:
            df: DataFrame with calculated technical indicators
            
        Returns:
            Dictionary containing trading signals and confidence scores
        """
        if df.empty:
            return {}
        
        # Get the latest row of data
        latest = df.iloc[-1]
        signals = {}
        
        # 1. TREND SIGNALS
        trend_signals = []
        
        # Moving Average signals
        if all(col in latest.index for col in ['sma_10', 'sma_20', 'sma_50']):
            if latest['sma_10'] > latest['sma_20'] > latest['sma_50']:
                trend_signals.append(1)  # Bullish
            elif latest['sma_10'] < latest['sma_20'] < latest['sma_50']:
                trend_signals.append(-1)  # Bearish
            else:
                trend_signals.append(0)  # Neutral
        
        # MACD signals
        if all(col in latest.index for col in ['macd', 'macd_signal']):
            if latest['macd'] > latest['macd_signal']:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
        
        signals['trend_signal'] = np.mean(trend_signals) if trend_signals else 0
        
        # 2. MOMENTUM SIGNALS
        momentum_signals = []
        
        # RSI signals
        if 'rsi' in latest.index:
            if latest['rsi'] > 70:
                momentum_signals.append(-1)  # Overbought
            elif latest['rsi'] < 30:
                momentum_signals.append(1)   # Oversold
            else:
                momentum_signals.append(0)   # Neutral
        
        # Stochastic signals
        if all(col in latest.index for col in ['stoch_k', 'stoch_d']):
            if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] > 20:
                momentum_signals.append(1)   # Bullish crossover
            elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] < 80:
                momentum_signals.append(-1)  # Bearish crossover
        
        signals['momentum_signal'] = np.mean(momentum_signals) if momentum_signals else 0
        
        # 3. VOLUME SIGNALS
        volume_signals = []
        
        if 'volume_breakout' in latest.index and latest['volume_breakout']:
            # High volume supports the current price movement
            if latest.get('is_bullish', False):
                volume_signals.append(1)
            elif latest.get('is_bearish', False):
                volume_signals.append(-1)
        
        signals['volume_signal'] = np.mean(volume_signals) if volume_signals else 0
        
        # 4. COMPOSITE SIGNALS
        if 'signal_strength' in latest.index:
            signals['composite_signal'] = latest['signal_strength'] / 100
        
        # 5. OVERALL RECOMMENDATION
        # Combine all signals with weights
        all_signals = [
            signals.get('trend_signal', 0) * 0.4,      # 40% weight to trend
            signals.get('momentum_signal', 0) * 0.3,    # 30% weight to momentum
            signals.get('volume_signal', 0) * 0.2,      # 20% weight to volume
            signals.get('composite_signal', 0) * 0.1    # 10% weight to composite
        ]
        
        overall_signal = sum(all_signals)
        signals['overall_signal'] = overall_signal
        
        # Convert to recommendation
        if overall_signal > 0.3:
            signals['recommendation'] = 'BUY'
            signals['confidence'] = min(overall_signal, 1.0)
        elif overall_signal < -0.3:
            signals['recommendation'] = 'SELL'
            signals['confidence'] = min(abs(overall_signal), 1.0)
        else:
            signals['recommendation'] = 'HOLD'
            signals['confidence'] = 1.0 - abs(overall_signal)
        
        return signals


def main():
    """Example usage of TechnicalIndicators class."""
    # This would typically be called with real stock data
    print("Technical Indicators Module")
    print("=" * 40)
    print("This module calculates comprehensive technical indicators")
    print("for investment decision making.")
    print("\nKey Features:")
    print("- Trend indicators (SMA, EMA, MACD, ADX)")
    print("- Momentum indicators (RSI, Stochastic, Williams %R)")
    print("- Volatility indicators (Bollinger Bands, ATR)")
    print("- Volume indicators (OBV, CMF)")
    print("- Composite signals combining multiple indicators")
    print("- Trading recommendations with confidence scores")


if __name__ == "__main__":
    main()
