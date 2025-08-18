"""
Investment Decision Engine

This is the core module that combines all signals from technical analysis,
machine learning models, and risk management to generate final investment
decisions. It acts as the "brain" of the investment system.

Key Components:
1. Signal Integration - Combines multiple signal sources
2. Decision Logic - Applies rules and filters to generate actions
3. Risk Filtering - Ensures decisions meet risk management criteria
4. Confidence Scoring - Provides confidence levels for each decision
5. Action Generation - Creates specific buy/sell/hold recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.technical_indicators import TechnicalIndicators
from models.ml_models import InvestmentMLModels
from risk.portfolio_optimizer import PortfolioOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InvestmentSignal:
    """Data class to represent an investment signal."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    expected_return: float  # Expected percentage return
    risk_score: float  # Risk assessment 0.0 to 1.0
    time_horizon: str  # 'short', 'medium', 'long'
    reasoning: List[str]  # List of reasons for the decision
    timestamp: datetime


class InvestmentDecisionEngine:
    """
    Advanced investment decision engine that combines multiple signal sources.
    
    This class integrates technical analysis, machine learning predictions,
    and risk management to generate comprehensive investment recommendations.
    It's designed to be the central decision-making component of the system.
    """
    
    def __init__(self, 
                 risk_tolerance: str = 'moderate',
                 max_positions: int = 10,
                 min_confidence: float = 0.6):
        """
        Initialize the decision engine.
        
        Args:
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
            max_positions: Maximum number of positions to hold
            min_confidence: Minimum confidence threshold for trades
        """
        self.risk_tolerance = risk_tolerance
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        
        # Initialize component modules
        self.technical_analyzer = TechnicalIndicators()
        self.ml_models = InvestmentMLModels()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Decision weights for different signal sources
        # These can be adjusted based on backtesting results
        self.signal_weights = {
            'technical': 0.3,      # Technical analysis weight
            'ml_models': 0.4,      # Machine learning weight  
            'risk_management': 0.2, # Risk management weight
            'market_regime': 0.1   # Market regime weight
        }
        
        # Risk tolerance settings
        self.risk_settings = {
            'conservative': {
                'max_position_size': 0.05,  # 5% max per position
                'max_portfolio_risk': 0.10, # 10% max portfolio volatility
                'min_sharpe_ratio': 1.0,
                'max_drawdown_limit': 0.10
            },
            'moderate': {
                'max_position_size': 0.10,  # 10% max per position
                'max_portfolio_risk': 0.15, # 15% max portfolio volatility
                'min_sharpe_ratio': 0.5,
                'max_drawdown_limit': 0.15
            },
            'aggressive': {
                'max_position_size': 0.20,  # 20% max per position
                'max_portfolio_risk': 0.25, # 25% max portfolio volatility
                'min_sharpe_ratio': 0.3,
                'max_drawdown_limit': 0.25
            }
        }
        
        # Current market regime assessment
        self.market_regime = 'normal'  # 'bull', 'bear', 'normal', 'volatile'
        
        logger.info(f"Decision Engine initialized with {risk_tolerance} risk tolerance")
    
    def analyze_single_asset(self, 
                           symbol: str, 
                           price_data: pd.DataFrame,
                           market_data: pd.DataFrame = None) -> InvestmentSignal:
        """
        Perform comprehensive analysis on a single asset.
        
        This function runs the complete analysis pipeline on one stock:
        1. Calculate technical indicators
        2. Generate ML predictions
        3. Assess risk metrics
        4. Combine signals into final recommendation
        
        Args:
            symbol: Stock symbol to analyze
            price_data: OHLCV data for the stock
            market_data: Market indices data for context
            
        Returns:
            InvestmentSignal with recommendation and reasoning
        """
        logger.info(f"Analyzing {symbol}...")
        
        try:
            # 1. TECHNICAL ANALYSIS
            # Add all technical indicators to the price data
            enriched_data = self.technical_analyzer.calculate_all_indicators(price_data)
            
            # Get technical analysis signals
            technical_signals = self.technical_analyzer.get_trading_signals(enriched_data)
            
            # 2. MACHINE LEARNING PREDICTIONS
            # Prepare features and make predictions
            ml_features = self.ml_models.prepare_features(enriched_data)
            ml_predictions = self.ml_models.make_predictions(ml_features)
            
            # 3. RISK ASSESSMENT
            # Calculate risk metrics for this asset
            returns = price_data['close'].pct_change().dropna()
            risk_metrics = self.portfolio_optimizer.calculate_risk_metrics(returns)
            
            # 4. MARKET REGIME ANALYSIS
            market_signals = self._analyze_market_regime(enriched_data, market_data)
            
            # 5. COMBINE ALL SIGNALS
            combined_signal = self._combine_signals(
                symbol=symbol,
                technical_signals=technical_signals,
                ml_predictions=ml_predictions,
                risk_metrics=risk_metrics,
                market_signals=market_signals,
                price_data=enriched_data
            )
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
            # Return neutral signal on error
            return InvestmentSignal(
                symbol=symbol,
                signal_type='HOLD',
                confidence=0.0,
                expected_return=0.0,
                risk_score=1.0,
                time_horizon='medium',
                reasoning=[f"Analysis error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _analyze_market_regime(self, 
                             asset_data: pd.DataFrame, 
                             market_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Analyze current market regime and conditions.
        
        Market regime affects how we interpret signals. Bull markets favor
        momentum strategies, while bear markets favor mean reversion.
        
        Args:
            asset_data: Individual asset data with indicators
            market_data: Market indices data
            
        Returns:
            Dictionary with market regime signals
        """
        market_signals = {}
        
        try:
            # Get latest data
            latest = asset_data.iloc[-1] if not asset_data.empty else None
            if latest is None:
                return {'regime_signal': 0, 'regime_confidence': 0}
            
            # 1. TREND REGIME ANALYSIS
            # Check multiple timeframe alignment
            trend_score = 0
            trend_factors = 0
            
            # Short-term vs medium-term trend
            if 'sma_10' in latest.index and 'sma_50' in latest.index:
                if latest['sma_10'] > latest['sma_50']:
                    trend_score += 1
                trend_factors += 1
            
            # Medium-term vs long-term trend
            if 'sma_50' in latest.index and 'sma_200' in latest.index:
                if latest['sma_50'] > latest['sma_200']:
                    trend_score += 1
                trend_factors += 1
            
            # Price vs moving averages
            if 'close' in latest.index and 'sma_50' in latest.index:
                if latest['close'] > latest['sma_50']:
                    trend_score += 1
                trend_factors += 1
            
            trend_strength = trend_score / trend_factors if trend_factors > 0 else 0.5
            
            # 2. VOLATILITY REGIME
            volatility_score = 0.5  # Default neutral
            
            if 'atr' in asset_data.columns:
                current_atr = latest.get('atr', 0)
                avg_atr = asset_data['atr'].rolling(50).mean().iloc[-1] if len(asset_data) >= 50 else current_atr
                
                if current_atr > avg_atr * 1.5:
                    volatility_score = 0.2  # High volatility - be cautious
                elif current_atr < avg_atr * 0.7:
                    volatility_score = 0.8  # Low volatility - favorable for trends
            
            # 3. MOMENTUM REGIME
            momentum_score = 0.5
            
            if 'rsi' in latest.index:
                rsi = latest['rsi']
                if 30 < rsi < 70:  # Healthy momentum range
                    momentum_score = 0.8
                elif rsi > 80 or rsi < 20:  # Extreme momentum
                    momentum_score = 0.3
            
            # 4. MARKET BREADTH (if market data available)
            breadth_score = 0.5
            if market_data is not None and not market_data.empty:
                # Compare asset performance to market
                try:
                    asset_returns = asset_data['close'].pct_change(20).iloc[-1]  # 20-day return
                    market_returns = market_data.iloc[:, 0].pct_change(20).iloc[-1]  # First market index
                    
                    relative_strength = asset_returns - market_returns
                    breadth_score = 0.5 + (relative_strength * 2)  # Scale to 0-1
                    breadth_score = max(0, min(1, breadth_score))  # Clip to valid range
                    
                except:
                    breadth_score = 0.5
            
            # 5. COMBINE REGIME SIGNALS
            regime_components = [trend_strength, volatility_score, momentum_score, breadth_score]
            overall_regime = np.mean(regime_components)
            
            # Determine regime type
            if overall_regime > 0.7:
                self.market_regime = 'bull'
            elif overall_regime < 0.3:
                self.market_regime = 'bear'
            elif volatility_score < 0.3:
                self.market_regime = 'volatile'
            else:
                self.market_regime = 'normal'
            
            market_signals = {
                'regime_signal': overall_regime,
                'regime_confidence': 1.0 - np.std(regime_components),  # Higher confidence if components agree
                'trend_strength': trend_strength,
                'volatility_regime': volatility_score,
                'momentum_regime': momentum_score,
                'market_breadth': breadth_score,
                'regime_type': self.market_regime
            }
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            market_signals = {'regime_signal': 0.5, 'regime_confidence': 0.5}
        
        return market_signals
    
    def _combine_signals(self,
                        symbol: str,
                        technical_signals: Dict[str, float],
                        ml_predictions: Dict[str, float],
                        risk_metrics: Dict[str, float],
                        market_signals: Dict[str, float],
                        price_data: pd.DataFrame) -> InvestmentSignal:
        """
        Combine all signal sources into a final investment decision.
        
        This is the core decision-making logic that weighs different
        signal sources and generates the final recommendation.
        
        Args:
            symbol: Stock symbol
            technical_signals: Technical analysis results
            ml_predictions: ML model predictions
            risk_metrics: Risk assessment metrics
            market_signals: Market regime analysis
            price_data: Price data with indicators
            
        Returns:
            Final InvestmentSignal with recommendation
        """
        reasoning = []
        signal_components = []
        
        # 1. PROCESS TECHNICAL SIGNALS
        technical_score = 0
        if technical_signals:
            tech_signal = technical_signals.get('overall_signal', 0)
            tech_confidence = technical_signals.get('confidence', 0.5)
            
            # Normalize technical signal to -1, +1 range
            technical_score = np.clip(tech_signal, -1, 1) * tech_confidence
            signal_components.append(technical_score * self.signal_weights['technical'])
            
            # Add reasoning
            if abs(tech_signal) > 0.3:
                direction = "bullish" if tech_signal > 0 else "bearish"
                reasoning.append(f"Technical analysis is {direction} (signal: {tech_signal:.2f})")
        
        # 2. PROCESS ML PREDICTIONS
        ml_score = 0
        if ml_predictions:
            # Direction prediction
            direction_signal = ml_predictions.get('ensemble_signal', 0)
            ml_confidence = ml_predictions.get('confidence', 0.5)
            
            # Normalize ML signal
            ml_score = np.clip(direction_signal, -1, 1) * ml_confidence
            signal_components.append(ml_score * self.signal_weights['ml_models'])
            
            # Add reasoning
            if ml_predictions.get('recommendation'):
                reasoning.append(f"ML models suggest {ml_predictions['recommendation']} "
                               f"(confidence: {ml_confidence:.2f})")
        
        # 3. PROCESS RISK METRICS
        risk_score = 0
        risk_penalty = 0
        
        if risk_metrics:
            # Risk-adjusted scoring
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            volatility = risk_metrics.get('annual_volatility', 0.2)
            
            # Risk tolerance adjustment
            risk_settings = self.risk_settings[self.risk_tolerance]
            
            # Penalize high-risk assets based on risk tolerance
            if volatility > risk_settings['max_portfolio_risk']:
                risk_penalty = -0.3  # Reduce signal strength
                reasoning.append(f"High volatility ({volatility:.1%}) exceeds risk tolerance")
            
            if max_drawdown < -risk_settings['max_drawdown_limit']:
                risk_penalty -= 0.2  # Additional penalty for high drawdown
                reasoning.append(f"High maximum drawdown ({max_drawdown:.1%})")
            
            # Reward good risk-adjusted returns
            if sharpe_ratio > risk_settings['min_sharpe_ratio']:
                risk_score = 0.2  # Boost signal for good Sharpe ratio
                reasoning.append(f"Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
            
            signal_components.append((risk_score + risk_penalty) * self.signal_weights['risk_management'])
        
        # 4. PROCESS MARKET REGIME
        regime_adjustment = 0
        if market_signals:
            regime_signal = market_signals.get('regime_signal', 0.5)
            regime_type = market_signals.get('regime_type', 'normal')
            
            # Adjust signals based on market regime
            if regime_type == 'bull':
                regime_adjustment = 0.1  # Slight bullish bias
                reasoning.append("Bullish market regime supports long positions")
            elif regime_type == 'bear':
                regime_adjustment = -0.1  # Slight bearish bias
                reasoning.append("Bearish market regime suggests caution")
            elif regime_type == 'volatile':
                regime_adjustment = -0.05  # Reduce position sizes in volatile markets
                reasoning.append("High volatility regime - reducing position size")
            
            signal_components.append(regime_adjustment * self.signal_weights['market_regime'])
        
        # 5. COMBINE ALL SIGNALS
        combined_score = sum(signal_components)
        
        # Calculate overall confidence based on signal agreement
        signal_agreement = 1.0 - np.std([comp/weight for comp, weight in 
                                       zip(signal_components, self.signal_weights.values()) 
                                       if weight > 0])
        overall_confidence = max(0.1, min(1.0, signal_agreement))
        
        # 6. GENERATE FINAL DECISION
        # Apply confidence threshold
        if overall_confidence < self.min_confidence:
            signal_type = 'HOLD'
            reasoning.append(f"Low confidence ({overall_confidence:.2f}) - holding position")
        else:
            # Determine signal type based on combined score
            if combined_score > 0.2:
                signal_type = 'BUY'
            elif combined_score < -0.2:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
        
        # 7. ESTIMATE EXPECTED RETURN
        # Base expected return on signal strength and historical volatility
        expected_return = combined_score * 10  # Scale to percentage
        
        # Adjust for market regime
        if self.market_regime == 'bull':
            expected_return *= 1.2
        elif self.market_regime == 'bear':
            expected_return *= 0.8
        
        # 8. CALCULATE RISK SCORE
        asset_risk = risk_metrics.get('annual_volatility', 0.2) if risk_metrics else 0.2
        normalized_risk = min(1.0, asset_risk / 0.5)  # Normalize to 0-1 scale
        
        # 9. DETERMINE TIME HORIZON
        # Base on signal strength and market conditions
        if abs(combined_score) > 0.5 and self.market_regime in ['bull', 'bear']:
            time_horizon = 'short'  # Strong signals in trending markets
        elif abs(combined_score) > 0.3:
            time_horizon = 'medium'  # Moderate signals
        else:
            time_horizon = 'long'   # Weak signals - longer holding period
        
        # 10. CREATE FINAL SIGNAL
        final_signal = InvestmentSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=overall_confidence,
            expected_return=expected_return,
            risk_score=normalized_risk,
            time_horizon=time_horizon,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        logger.info(f"{symbol}: {signal_type} (confidence: {overall_confidence:.2f}, "
                   f"expected return: {expected_return:.1f}%)")
        
        return final_signal
    
    def analyze_portfolio(self, 
                         symbols: List[str],
                         price_data_dict: Dict[str, pd.DataFrame],
                         market_data: pd.DataFrame = None,
                         current_positions: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Analyze multiple assets and generate portfolio recommendations.
        
        This function performs comprehensive analysis on a list of assets
        and provides portfolio-level recommendations including position
        sizing and rebalancing suggestions.
        
        Args:
            symbols: List of stock symbols to analyze
            price_data_dict: Dictionary mapping symbols to price data
            market_data: Market indices data
            current_positions: Current portfolio positions
            
        Returns:
            Dictionary with portfolio analysis and recommendations
        """
        logger.info(f"Analyzing portfolio with {len(symbols)} assets...")
        
        # Initialize results
        individual_signals = {}
        portfolio_analysis = {
            'individual_signals': {},
            'portfolio_recommendations': {},
            'risk_assessment': {},
            'rebalancing_suggestions': {},
            'summary': {}
        }
        
        # 1. ANALYZE EACH ASSET INDIVIDUALLY
        for symbol in symbols:
            if symbol in price_data_dict:
                try:
                    signal = self.analyze_single_asset(
                        symbol=symbol,
                        price_data=price_data_dict[symbol],
                        market_data=market_data
                    )
                    individual_signals[symbol] = signal
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
        
        portfolio_analysis['individual_signals'] = individual_signals
        
        # 2. GENERATE PORTFOLIO-LEVEL RECOMMENDATIONS
        if individual_signals:
            # Filter signals by confidence threshold
            high_confidence_signals = {
                symbol: signal for symbol, signal in individual_signals.items()
                if signal.confidence >= self.min_confidence
            }
            
            # Separate by signal type
            buy_signals = {k: v for k, v in high_confidence_signals.items() if v.signal_type == 'BUY'}
            sell_signals = {k: v for k, v in high_confidence_signals.items() if v.signal_type == 'SELL'}
            
            # Rank by expected return and confidence
            buy_ranked = sorted(buy_signals.items(), 
                              key=lambda x: x[1].expected_return * x[1].confidence, 
                              reverse=True)
            
            sell_ranked = sorted(sell_signals.items(),
                               key=lambda x: abs(x[1].expected_return) * x[1].confidence,
                               reverse=True)
            
            # Limit to max positions
            top_buys = buy_ranked[:self.max_positions//2]
            top_sells = sell_ranked[:self.max_positions//2]
            
            portfolio_analysis['portfolio_recommendations'] = {
                'top_buy_candidates': [symbol for symbol, _ in top_buys],
                'top_sell_candidates': [symbol for symbol, _ in top_sells],
                'buy_signals_count': len(buy_signals),
                'sell_signals_count': len(sell_signals),
                'hold_signals_count': len(individual_signals) - len(buy_signals) - len(sell_signals)
            }
        
        # 3. PORTFOLIO RISK ASSESSMENT
        if current_positions and price_data_dict:
            # Calculate portfolio returns
            position_symbols = [s for s in current_positions.keys() if s in price_data_dict]
            
            if position_symbols:
                # Create portfolio returns series
                returns_data = pd.DataFrame()
                for symbol in position_symbols:
                    if symbol in price_data_dict:
                        returns_data[symbol] = price_data_dict[symbol]['close'].pct_change()
                
                if not returns_data.empty:
                    # Calculate current portfolio weights
                    total_value = sum(abs(pos) for pos in current_positions.values())
                    if total_value > 0:
                        weights = {s: current_positions.get(s, 0)/total_value for s in position_symbols}
                        
                        # Assess portfolio risk
                        risk_assessment = self.portfolio_optimizer.assess_portfolio_risk(
                            current_positions, returns_data
                        )
                        portfolio_analysis['risk_assessment'] = risk_assessment
        
        # 4. GENERATE SUMMARY STATISTICS
        if individual_signals:
            total_signals = len(individual_signals)
            buy_count = sum(1 for s in individual_signals.values() if s.signal_type == 'BUY')
            sell_count = sum(1 for s in individual_signals.values() if s.signal_type == 'SELL')
            hold_count = total_signals - buy_count - sell_count
            
            avg_confidence = np.mean([s.confidence for s in individual_signals.values()])
            avg_expected_return = np.mean([s.expected_return for s in individual_signals.values()])
            
            portfolio_analysis['summary'] = {
                'total_assets_analyzed': total_signals,
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'hold_signals': hold_count,
                'average_confidence': avg_confidence,
                'average_expected_return': avg_expected_return,
                'market_regime': self.market_regime,
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        return portfolio_analysis
    
    def update_model_weights(self, performance_data: Dict[str, float]):
        """
        Update signal weights based on historical performance.
        
        This function allows the decision engine to learn from its
        past performance and adjust the weights given to different
        signal sources.
        
        Args:
            performance_data: Dictionary with performance metrics for each signal type
        """
        logger.info("Updating model weights based on performance...")
        
        # Simple weight adjustment based on relative performance
        total_performance = sum(performance_data.values())
        
        if total_performance > 0:
            for signal_type, performance in performance_data.items():
                if signal_type in self.signal_weights:
                    # Adjust weight based on relative performance
                    performance_ratio = performance / total_performance
                    current_weight = self.signal_weights[signal_type]
                    
                    # Gradual adjustment (don't change too quickly)
                    adjustment_factor = 0.1  # 10% adjustment per update
                    new_weight = current_weight + (performance_ratio - current_weight) * adjustment_factor
                    
                    self.signal_weights[signal_type] = max(0.05, min(0.6, new_weight))  # Keep reasonable bounds
        
        # Normalize weights to sum to 1
        total_weight = sum(self.signal_weights.values())
        self.signal_weights = {k: v/total_weight for k, v in self.signal_weights.items()}
        
        logger.info(f"Updated signal weights: {self.signal_weights}")


def main():
    """Example usage of InvestmentDecisionEngine."""
    print("Investment Decision Engine")
    print("=" * 40)
    print("This is the core decision-making module that:")
    print("- Combines technical analysis, ML predictions, and risk metrics")
    print("- Generates buy/sell/hold recommendations with confidence scores")
    print("- Provides detailed reasoning for each decision")
    print("- Adapts to different market regimes")
    print("- Manages portfolio-level risk and optimization")
    print("\nKey Features:")
    print("- Multi-signal integration with configurable weights")
    print("- Risk tolerance adjustment (conservative/moderate/aggressive)")
    print("- Market regime awareness")
    print("- Portfolio-level analysis and recommendations")
    print("- Adaptive learning from performance feedback")


if __name__ == "__main__":
    main()
