"""
Portfolio Optimization and Risk Management Module

This module handles portfolio optimization, risk assessment, and position sizing
for the investment decision system. It implements Modern Portfolio Theory (MPT),
risk metrics calculation, and dynamic position sizing based on market conditions.

Key Components:
1. Portfolio Optimization - Maximize return for given risk level
2. Risk Metrics - VaR, Sharpe ratio, maximum drawdown
3. Position Sizing - Kelly criterion, volatility-based sizing
4. Risk Management - Stop losses, portfolio limits, correlation analysis
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Advanced portfolio optimization and risk management system.
    
    This class implements various portfolio optimization techniques and
    risk management strategies to help make better investment decisions
    while controlling downside risk.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2% for US Treasury)
        """
        self.risk_free_rate = risk_free_rate
        self.portfolio_history = []
        self.risk_metrics_cache = {}
        
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        This function computes daily returns which are the foundation
        for all portfolio optimization and risk calculations.
        
        Args:
            prices: DataFrame with price data (columns = assets, rows = dates)
            
        Returns:
            DataFrame with daily returns
        """
        # Calculate daily returns (percentage change)
        returns = prices.pct_change().dropna()
        
        # Remove any infinite or extremely large values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0)
        
        # Cap extreme returns (likely data errors)
        returns = returns.clip(lower=-0.5, upper=0.5)  # Cap at ±50% daily returns
        
        return returns
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key portfolio performance metrics.
        
        This function computes essential metrics used in portfolio optimization:
        - Expected return
        - Portfolio volatility (risk)
        - Sharpe ratio (risk-adjusted return)
        - Maximum drawdown
        
        Args:
            weights: Portfolio weights (must sum to 1)
            returns: DataFrame with asset returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Ensure weights are normalized
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Expected annual return
        expected_return = portfolio_returns.mean() * 252  # Annualized
        
        # Portfolio volatility (risk)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (risk-adjusted return)
        excess_return = expected_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - 5% worst case scenario
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(expected_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'sharpe') -> Dict[str, any]:
        """
        Optimize portfolio weights using various methods.
        
        This function finds the optimal portfolio allocation that maximizes
        the chosen objective (Sharpe ratio, return, or minimum risk).
        
        Args:
            returns: DataFrame with asset returns
            method: Optimization method ('sharpe', 'min_vol', 'max_return', 'risk_parity')
            
        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        n_assets = len(returns.columns)
        
        # Initial guess - equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints - weights must sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        # Bounds - each weight between 0 and 1 (long-only portfolio)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Define objective functions
        def negative_sharpe(weights):
            """Negative Sharpe ratio for minimization."""
            metrics = self.calculate_portfolio_metrics(weights, returns)
            return -metrics['sharpe_ratio']
        
        def portfolio_volatility(weights):
            """Portfolio volatility."""
            metrics = self.calculate_portfolio_metrics(weights, returns)
            return metrics['volatility']
        
        def negative_return(weights):
            """Negative expected return for minimization."""
            metrics = self.calculate_portfolio_metrics(weights, returns)
            return -metrics['expected_return']
        
        def risk_parity_objective(weights):
            """Risk parity objective - equal risk contribution."""
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Marginal risk contribution
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            
            # Risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Minimize difference from target
            return np.sum((risk_contrib / np.sum(risk_contrib) - target_contrib) ** 2)
        
        # Choose objective function based on method
        if method == 'sharpe':
            objective = negative_sharpe
        elif method == 'min_vol':
            objective = portfolio_volatility
        elif method == 'max_return':
            objective = negative_return
        elif method == 'risk_parity':
            objective = risk_parity_objective
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        try:
            # Perform optimization
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics with optimal weights
                portfolio_metrics = self.calculate_portfolio_metrics(optimal_weights, returns)
                
                # Create results dictionary
                optimization_result = {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'metrics': portfolio_metrics,
                    'method': method,
                    'success': True,
                    'optimization_result': result
                }
                
                logger.info(f"Portfolio optimization successful ({method})")
                logger.info(f"Expected Return: {portfolio_metrics['expected_return']:.2%}")
                logger.info(f"Volatility: {portfolio_metrics['volatility']:.2%}")
                logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
                
                return optimization_result
            
            else:
                logger.error(f"Optimization failed: {result.message}")
                return self._get_equal_weight_portfolio(returns)
        
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._get_equal_weight_portfolio(returns)
    
    def _get_equal_weight_portfolio(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Fallback to equal-weight portfolio if optimization fails.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Dictionary with equal-weight portfolio
        """
        n_assets = len(returns.columns)
        equal_weights = np.array([1.0 / n_assets] * n_assets)
        
        portfolio_metrics = self.calculate_portfolio_metrics(equal_weights, returns)
        
        return {
            'weights': dict(zip(returns.columns, equal_weights)),
            'metrics': portfolio_metrics,
            'method': 'equal_weight',
            'success': True
        }
    
    def calculate_position_sizes(self, 
                               signals: Dict[str, float],
                               portfolio_value: float,
                               returns: pd.DataFrame,
                               max_position_size: float = 0.1) -> Dict[str, float]:
        """
        Calculate optimal position sizes based on signals and risk management.
        
        This function determines how much to invest in each asset based on:
        - Signal strength and confidence
        - Historical volatility
        - Kelly criterion
        - Maximum position size limits
        
        Args:
            signals: Dictionary of asset signals (-1 to 1)
            portfolio_value: Total portfolio value
            returns: Historical returns for volatility estimation
            max_position_size: Maximum position size as fraction of portfolio
            
        Returns:
            Dictionary with dollar amounts to invest in each asset
        """
        position_sizes = {}
        
        for asset, signal in signals.items():
            if asset not in returns.columns:
                continue
            
            # Get asset returns
            asset_returns = returns[asset].dropna()
            
            if len(asset_returns) < 30:  # Need minimum history
                position_sizes[asset] = 0
                continue
            
            # Calculate volatility
            volatility = asset_returns.std() * np.sqrt(252)  # Annualized
            
            # Estimate expected return based on signal
            # Stronger signals suggest higher expected returns
            expected_return = abs(signal) * 0.15  # Scale signal to reasonable return expectation
            
            # Kelly criterion for optimal position size
            # f* = (bp - q) / b, where b = odds, p = win probability, q = loss probability
            if volatility > 0:
                # Simplified Kelly: f = expected_return / variance
                kelly_fraction = expected_return / (volatility ** 2)
                
                # Cap Kelly fraction to prevent excessive leverage
                kelly_fraction = min(kelly_fraction, 0.25)  # Max 25% from Kelly
                
                # Apply signal direction and strength
                signal_adjusted_fraction = kelly_fraction * signal
                
                # Apply maximum position size constraint
                final_fraction = np.sign(signal_adjusted_fraction) * min(
                    abs(signal_adjusted_fraction), 
                    max_position_size
                )
                
                # Convert to dollar amount
                position_sizes[asset] = final_fraction * portfolio_value
            
            else:
                position_sizes[asset] = 0
        
        return position_sizes
    
    def calculate_risk_metrics(self, returns: pd.DataFrame, window: int = 252) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for the portfolio.
        
        This function computes various risk measures that help assess
        the safety and stability of the investment strategy.
        
        Args:
            returns: DataFrame with portfolio returns
            window: Rolling window for calculations (default: 1 year)
            
        Returns:
            Dictionary with risk metrics
        """
        if returns.empty:
            return {}
        
        # Convert to portfolio returns if multiple columns
        if isinstance(returns, pd.DataFrame) and len(returns.columns) > 1:
            # Assume equal weights if not specified
            portfolio_returns = returns.mean(axis=1)
        else:
            portfolio_returns = returns.squeeze() if isinstance(returns, pd.DataFrame) else returns
        
        # Basic statistics
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - multiple confidence levels
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Downside deviation (semi-volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar = abs(mean_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Rolling metrics
        rolling_vol = portfolio_returns.rolling(window=min(window, len(portfolio_returns)//2)).std() * np.sqrt(252)
        rolling_sharpe = (portfolio_returns.rolling(window=min(window, len(portfolio_returns)//2)).mean() * 252 - self.risk_free_rate) / rolling_vol
        
        # Stability metrics
        vol_of_vol = rolling_vol.std() if len(rolling_vol.dropna()) > 1 else 0
        sharpe_stability = rolling_sharpe.std() if len(rolling_sharpe.dropna()) > 1 else 0
        
        # Tail risk metrics
        skewness = stats.skew(portfolio_returns.dropna())
        kurtosis = stats.kurtosis(portfolio_returns.dropna())
        
        # Win rate
        positive_returns = portfolio_returns > 0
        win_rate = positive_returns.mean()
        
        # Average win vs average loss
        wins = portfolio_returns[portfolio_returns > 0]
        losses = portfolio_returns[portfolio_returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        risk_metrics = {
            'annual_return': mean_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'var_95_daily': var_95,
            'var_99_daily': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'downside_deviation': downside_deviation,
            'volatility_of_volatility': vol_of_vol,
            'sharpe_stability': sharpe_stability,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio
        }
        
        return risk_metrics
    
    def assess_portfolio_risk(self, 
                            current_positions: Dict[str, float],
                            returns: pd.DataFrame) -> Dict[str, any]:
        """
        Assess current portfolio risk and provide recommendations.
        
        This function analyzes the current portfolio allocation and
        identifies potential risk issues and improvement opportunities.
        
        Args:
            current_positions: Dictionary of current position sizes
            returns: Historical returns data
            
        Returns:
            Dictionary with risk assessment and recommendations
        """
        assessment = {
            'risk_level': 'UNKNOWN',
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        if not current_positions or not returns.empty:
            return assessment
        
        # Calculate portfolio weights
        total_value = sum(abs(pos) for pos in current_positions.values())
        if total_value == 0:
            assessment['risk_level'] = 'NO_POSITIONS'
            return assessment
        
        weights = {asset: pos/total_value for asset, pos in current_positions.items()}
        
        # Check concentration risk
        max_weight = max(abs(w) for w in weights.values())
        if max_weight > 0.3:
            assessment['warnings'].append(f"High concentration risk: {max_weight:.1%} in single asset")
            assessment['recommendations'].append("Consider diversifying positions")
        
        # Check correlation risk
        assets_in_portfolio = [asset for asset in weights.keys() if asset in returns.columns]
        if len(assets_in_portfolio) > 1:
            correlation_matrix = returns[assets_in_portfolio].corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            if avg_correlation > 0.7:
                assessment['warnings'].append(f"High correlation between assets: {avg_correlation:.2f}")
                assessment['recommendations'].append("Consider adding uncorrelated assets")
        
        # Calculate portfolio risk metrics
        if len(assets_in_portfolio) > 0:
            portfolio_weights = np.array([weights.get(asset, 0) for asset in assets_in_portfolio])
            portfolio_returns = (returns[assets_in_portfolio] * portfolio_weights).sum(axis=1)
            
            risk_metrics = self.calculate_risk_metrics(portfolio_returns)
            assessment['metrics'] = risk_metrics
            
            # Risk level assessment
            if risk_metrics.get('annual_volatility', 0) > 0.25:
                assessment['risk_level'] = 'HIGH'
            elif risk_metrics.get('annual_volatility', 0) > 0.15:
                assessment['risk_level'] = 'MEDIUM'
            else:
                assessment['risk_level'] = 'LOW'
            
            # Additional warnings based on metrics
            if risk_metrics.get('max_drawdown', 0) < -0.2:
                assessment['warnings'].append("High maximum drawdown risk")
            
            if risk_metrics.get('sharpe_ratio', 0) < 0.5:
                assessment['warnings'].append("Low risk-adjusted returns")
            
            if risk_metrics.get('win_rate', 0) < 0.4:
                assessment['warnings'].append("Low win rate - consider strategy adjustment")
        
        return assessment
    
    def generate_rebalancing_signals(self, 
                                   current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   threshold: float = 0.05) -> Dict[str, float]:
        """
        Generate rebalancing signals when portfolio drifts from targets.
        
        This function compares current portfolio weights with target weights
        and suggests rebalancing trades when deviations exceed thresholds.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold (default: 5%)
            
        Returns:
            Dictionary with rebalancing signals
        """
        rebalancing_signals = {}
        
        # Get all assets
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0)
            target_weight = target_weights.get(asset, 0)
            
            deviation = target_weight - current_weight
            
            # Only rebalance if deviation exceeds threshold
            if abs(deviation) > threshold:
                rebalancing_signals[asset] = deviation
        
        return rebalancing_signals


def main():
    """Example usage of PortfolioOptimizer class."""
    print("Portfolio Optimizer and Risk Management Module")
    print("=" * 50)
    print("This module provides:")
    print("- Portfolio optimization (Sharpe, min volatility, risk parity)")
    print("- Risk metrics calculation (VaR, Sharpe, max drawdown)")
    print("- Position sizing using Kelly criterion")
    print("- Risk assessment and monitoring")
    print("- Rebalancing signals")
    print("\nKey Features:")
    print("- Modern Portfolio Theory implementation")
    print("- Multiple optimization methods")
    print("- Comprehensive risk metrics")
    print("- Dynamic position sizing")
    print("- Portfolio risk monitoring")


if __name__ == "__main__":
    main()
