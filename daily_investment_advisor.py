"""
Daily Investment Advisor - Main Script

This is the main script that runs daily to analyze your investment portfolio
and provide actionable investment recommendations. It coordinates all the
modules (data fetching, technical analysis, ML models, risk management)
to give you specific buy/sell/hold decisions for each day.

Usage:
    python daily_investment_advisor.py
    
    Optional arguments:
    --symbols AAPL,GOOGL,MSFT    # Specific symbols to analyze
    --risk-tolerance moderate    # conservative, moderate, or aggressive  
    --portfolio-value 100000     # Total portfolio value for position sizing
    --output-format json         # json, csv, or console
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data.data_fetcher import DataFetcher
from data.robinhood_integration import RobinhoodPortfolioManager
from analysis.technical_indicators import TechnicalIndicators
from models.ml_models import InvestmentMLModels
from models.decision_engine import InvestmentDecisionEngine
from risk.portfolio_optimizer import PortfolioOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_advisor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DailyInvestmentAdvisor:
    """
    Main class that orchestrates the daily investment analysis.
    
    This class brings together all the components of the investment system
    to provide daily recommendations. It handles data fetching, analysis,
    and output generation.
    """
    
    def __init__(self, 
                 risk_tolerance: str = 'moderate',
                 portfolio_value: float = 100000,
                 max_positions: int = 10):
        """
        Initialize the daily investment advisor.
        
        Args:
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            portfolio_value: Total portfolio value for position sizing
            max_positions: Maximum number of positions to hold
        """
        self.risk_tolerance = risk_tolerance
        self.portfolio_value = portfolio_value
        self.max_positions = max_positions
        
        # Initialize all components
        logger.info("Initializing Daily Investment Advisor...")
        
        self.data_fetcher = DataFetcher()
        self.portfolio_manager = RobinhoodPortfolioManager()
        self.technical_analyzer = TechnicalIndicators()
        self.ml_models = InvestmentMLModels()
        self.decision_engine = InvestmentDecisionEngine(
            risk_tolerance=risk_tolerance,
            max_positions=max_positions
        )
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Default watchlist - popular stocks across different sectors
        self.default_watchlist = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV',
            # Consumer
            'KO', 'PEP', 'WMT', 'HD',
            # Energy
            'XOM', 'CVX',
            # ETFs for diversification
            'SPY', 'QQQ', 'IWM', 'VTI'
        ]
        
        # Try to load existing portfolio data
        if self.portfolio_manager.load_saved_portfolio():
            portfolio_summary = self.portfolio_manager.get_current_portfolio_summary()
            if portfolio_summary['total_positions'] > 0:
                self.portfolio_value = portfolio_summary['total_market_value']
                logger.info(f"Loaded existing portfolio: {portfolio_summary['total_positions']} positions, "
                           f"${portfolio_summary['total_market_value']:,.0f} total value")
            else:
                logger.info("No existing positions found")
        
        logger.info(f"Initialized with {risk_tolerance} risk tolerance and ${self.portfolio_value:,.0f} portfolio")
    
    def fetch_market_data(self, symbols: List[str], period: str = '2y') -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for all symbols.
        
        Args:
            symbols: List of stock symbols
            period: Data period to fetch
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        
        try:
            # Fetch stock data
            stock_data = self.data_fetcher.get_stock_data(symbols, period=period)
            
            # Filter out empty datasets
            valid_data = {symbol: df for symbol, df in stock_data.items() 
                         if not df.empty and len(df) >= 100}  # Need minimum history
            
            logger.info(f"Successfully fetched data for {len(valid_data)}/{len(symbols)} symbols")
            
            if len(valid_data) == 0:
                raise ValueError("No valid market data fetched")
            
            return valid_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise
    
    def train_models(self, price_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Train machine learning models on historical data.
        
        This function prepares features and trains ML models that will
        be used for making predictions. Should be run periodically to
        keep models updated.
        
        Args:
            price_data_dict: Dictionary of price data for training
            
        Returns:
            Dictionary with training results and performance metrics
        """
        logger.info("Training machine learning models...")
        
        training_results = {}
        
        try:
            # Combine data from multiple assets for training
            # This gives us more data points and better generalization
            combined_features = pd.DataFrame()
            
            for symbol, price_data in price_data_dict.items():
                try:
                    # Calculate technical indicators
                    enriched_data = self.technical_analyzer.calculate_all_indicators(price_data)
                    
                    # Prepare ML features
                    ml_features = self.ml_models.prepare_features(enriched_data)
                    
                    # Add symbol identifier
                    ml_features['symbol'] = symbol
                    
                    # Append to combined dataset
                    combined_features = pd.concat([combined_features, ml_features], ignore_index=True)
                    
                except Exception as e:
                    logger.warning(f"Error preparing features for {symbol}: {e}")
                    continue
            
            if combined_features.empty:
                raise ValueError("No features prepared for model training")
            
            # Remove symbol column for training
            training_data = combined_features.drop('symbol', axis=1)
            
            # Train direction classifier
            try:
                direction_performance = self.ml_models.train_direction_classifier(training_data)
                training_results['direction_classifier'] = direction_performance
                logger.info("Direction classifier trained successfully")
                
            except Exception as e:
                logger.error(f"Error training direction classifier: {e}")
                training_results['direction_classifier'] = {'error': str(e)}
            
            # Train movement regressor
            try:
                movement_performance = self.ml_models.train_movement_regressor(training_data)
                training_results['movement_regressor'] = movement_performance
                logger.info("Movement regressor trained successfully")
                
            except Exception as e:
                logger.error(f"Error training movement regressor: {e}")
                training_results['movement_regressor'] = {'error': str(e)}
            
            # Save trained models
            try:
                model_path = 'models/trained_models.joblib'
                os.makedirs('models', exist_ok=True)
                self.ml_models.save_models(model_path)
                training_results['models_saved'] = model_path
                
            except Exception as e:
                logger.error(f"Error saving models: {e}")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            training_results['error'] = str(e)
        
        return training_results
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current portfolio positions from Robinhood data.
        
        Returns:
            Dictionary mapping symbols to dollar values of current positions
        """
        try:
            portfolio_data = self.portfolio_manager.export_for_investment_advisor()
            if 'error' not in portfolio_data:
                return portfolio_data.get('current_positions', {})
            else:
                logger.warning(f"No portfolio data available: {portfolio_data.get('message', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}
    
    def analyze_investments(self, symbols: List[str]) -> Dict[str, any]:
        """
        Perform comprehensive investment analysis.
        
        This is the main analysis function that:
        1. Fetches current market data
        2. Runs technical analysis
        3. Makes ML predictions
        4. Generates investment recommendations
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"Starting investment analysis for {len(symbols)} symbols...")
        
        analysis_start_time = datetime.now()
        
        try:
            # 1. FETCH MARKET DATA
            logger.info("Step 1: Fetching market data...")
            price_data_dict = self.fetch_market_data(symbols)
            
            # Also fetch market indices for context
            market_data = self.data_fetcher.get_market_data(period='1y')
            
            # 2. LOAD OR TRAIN ML MODELS
            logger.info("Step 2: Loading ML models...")
            model_path = 'models/trained_models.joblib'
            
            if os.path.exists(model_path):
                try:
                    self.ml_models.load_models(model_path)
                    logger.info("Loaded existing ML models")
                except:
                    logger.info("Failed to load models, training new ones...")
                    self.train_models(price_data_dict)
            else:
                logger.info("No existing models found, training new ones...")
                self.train_models(price_data_dict)
            
            # 3. GET CURRENT POSITIONS
            logger.info("Step 3: Loading current portfolio positions...")
            current_positions = self.get_current_positions()
            
            if current_positions:
                logger.info(f"Found {len(current_positions)} current positions")
                # Add current position symbols to analysis if not already included
                for symbol in current_positions.keys():
                    if symbol not in price_data_dict:
                        try:
                            symbol_data = self.data_fetcher.get_stock_data([symbol], period='2y')
                            if symbol in symbol_data and not symbol_data[symbol].empty:
                                price_data_dict[symbol] = symbol_data[symbol]
                                logger.info(f"Added current position {symbol} to analysis")
                        except Exception as e:
                            logger.warning(f"Could not fetch data for current position {symbol}: {e}")
            
            # 4. ANALYZE EACH ASSET
            logger.info("Step 4: Analyzing individual assets...")
            
            portfolio_analysis = self.decision_engine.analyze_portfolio(
                symbols=list(price_data_dict.keys()),
                price_data_dict=price_data_dict,
                market_data=market_data,
                current_positions=current_positions
            )
            
            # 5. GENERATE POSITION SIZING RECOMMENDATIONS
            logger.info("Step 5: Calculating position sizes...")
            
            individual_signals = portfolio_analysis.get('individual_signals', {})
            
            # Create signals dictionary for position sizing
            signals_for_sizing = {}
            for symbol, signal in individual_signals.items():
                if signal.signal_type == 'BUY':
                    signals_for_sizing[symbol] = signal.confidence * (signal.expected_return / 10)
                elif signal.signal_type == 'SELL':
                    signals_for_sizing[symbol] = -signal.confidence * (abs(signal.expected_return) / 10)
                else:
                    signals_for_sizing[symbol] = 0
            
            # Calculate position sizes
            position_sizes = {}
            if signals_for_sizing and price_data_dict:
                # Create returns dataframe for position sizing
                returns_data = pd.DataFrame()
                for symbol in signals_for_sizing.keys():
                    if symbol in price_data_dict:
                        returns_data[symbol] = price_data_dict[symbol]['close'].pct_change()
                
                if not returns_data.empty:
                    position_sizes = self.portfolio_optimizer.calculate_position_sizes(
                        signals=signals_for_sizing,
                        portfolio_value=self.portfolio_value,
                        returns=returns_data,
                        max_position_size=0.15 if self.risk_tolerance == 'aggressive' else 0.10
                    )
            
            # 6. COMPILE FINAL RESULTS
            analysis_end_time = datetime.now()
            analysis_duration = (analysis_end_time - analysis_start_time).total_seconds()
            
            # Get portfolio summary for metadata
            portfolio_summary = self.portfolio_manager.get_current_portfolio_summary()
            
            final_results = {
                'analysis_metadata': {
                    'timestamp': analysis_end_time.isoformat(),
                    'analysis_duration_seconds': analysis_duration,
                    'symbols_analyzed': len(price_data_dict),
                    'risk_tolerance': self.risk_tolerance,
                    'portfolio_value': self.portfolio_value,
                    'current_positions_count': len(current_positions),
                    'portfolio_return_to_date': portfolio_summary.get('total_return_percent', 0)
                },
                'market_overview': {
                    'market_regime': getattr(self.decision_engine, 'market_regime', 'normal'),
                    'market_indices': market_data.iloc[-1].to_dict() if not market_data.empty else {}
                },
                'individual_analysis': portfolio_analysis,
                'position_recommendations': position_sizes,
                'portfolio_summary': self._generate_portfolio_summary(
                    individual_signals, position_sizes
                )
            }
            
            logger.info(f"Analysis completed in {analysis_duration:.1f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in investment analysis: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_portfolio_summary(self, 
                                  individual_signals: Dict,
                                  position_sizes: Dict[str, float]) -> Dict[str, any]:
        """
        Generate a summary of portfolio recommendations.
        
        Args:
            individual_signals: Individual asset signals
            position_sizes: Recommended position sizes
            
        Returns:
            Dictionary with portfolio summary
        """
        summary = {
            'action_items': [],
            'risk_warnings': [],
            'key_opportunities': [],
            'portfolio_metrics': {}
        }
        
        if not individual_signals:
            return summary
        
        # Analyze signals
        buy_signals = [s for s in individual_signals.values() if s.signal_type == 'BUY']
        sell_signals = [s for s in individual_signals.values() if s.signal_type == 'SELL']
        
        # Generate action items
        if buy_signals:
            # Sort by expected return * confidence
            top_buys = sorted(buy_signals, 
                            key=lambda x: x.expected_return * x.confidence, 
                            reverse=True)[:5]
            
            for signal in top_buys:
                position_size = position_sizes.get(signal.symbol, 0)
                if position_size > 0:
                    summary['action_items'].append({
                        'action': 'BUY',
                        'symbol': signal.symbol,
                        'amount': position_size,
                        'confidence': signal.confidence,
                        'expected_return': signal.expected_return,
                        'reasoning': signal.reasoning[:2]  # Top 2 reasons
                    })
        
        if sell_signals:
            # Sort by confidence
            top_sells = sorted(sell_signals, 
                             key=lambda x: x.confidence, 
                             reverse=True)[:3]
            
            for signal in top_sells:
                summary['action_items'].append({
                    'action': 'SELL',
                    'symbol': signal.symbol,
                    'confidence': signal.confidence,
                    'expected_return': signal.expected_return,
                    'reasoning': signal.reasoning[:2]
                })
        
        # Identify high-risk positions
        high_risk_signals = [s for s in individual_signals.values() if s.risk_score > 0.7]
        for signal in high_risk_signals:
            summary['risk_warnings'].append(f"{signal.symbol}: High risk score ({signal.risk_score:.2f})")
        
        # Identify key opportunities
        high_confidence_buys = [s for s in buy_signals if s.confidence > 0.8]
        for signal in high_confidence_buys:
            summary['key_opportunities'].append(
                f"{signal.symbol}: High confidence buy signal "
                f"({signal.confidence:.2f} confidence, {signal.expected_return:.1f}% expected return)"
            )
        
        # Calculate portfolio metrics
        if position_sizes:
            total_invested = sum(abs(pos) for pos in position_sizes.values() if pos != 0)
            cash_remaining = self.portfolio_value - total_invested
            
            summary['portfolio_metrics'] = {
                'total_positions_recommended': len([p for p in position_sizes.values() if p != 0]),
                'total_amount_to_invest': total_invested,
                'cash_remaining': cash_remaining,
                'portfolio_utilization': total_invested / self.portfolio_value,
                'avg_position_size': total_invested / len(position_sizes) if position_sizes else 0
            }
        
        return summary
    
    def output_results(self, results: Dict[str, any], format_type: str = 'console'):
        """
        Output analysis results in the specified format.
        
        Args:
            results: Analysis results dictionary
            format_type: Output format ('console', 'json', 'csv')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            # Save to JSON file
            filename = f'investment_analysis_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
            
        elif format_type == 'csv':
            # Save action items to CSV
            filename = f'investment_recommendations_{timestamp}.csv'
            
            action_items = results.get('portfolio_summary', {}).get('action_items', [])
            if action_items:
                df = pd.DataFrame(action_items)
                df.to_csv(filename, index=False)
                logger.info(f"Recommendations saved to {filename}")
            
        else:  # console output
            self._print_console_output(results)
    
    def _print_console_output(self, results: Dict[str, any]):
        """Print formatted results to console."""
        print("\n" + "="*80)
        print("DAILY INVESTMENT ADVISOR REPORT")
        print("="*80)
        
        # Metadata
        metadata = results.get('analysis_metadata', {})
        print(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
        print(f"Risk Tolerance: {metadata.get('risk_tolerance', 'Unknown').upper()}")
        print(f"Portfolio Value: ${metadata.get('portfolio_value', 0):,.0f}")
        print(f"Symbols Analyzed: {metadata.get('symbols_analyzed', 0)}")
        
        # Market overview
        market = results.get('market_overview', {})
        print(f"\nMarket Regime: {market.get('market_regime', 'Unknown').upper()}")
        
        # Portfolio summary
        summary = results.get('portfolio_summary', {})
        
        # Action items
        action_items = summary.get('action_items', [])
        if action_items:
            print("\n📈 RECOMMENDED ACTIONS:")
            print("-" * 50)
            
            for item in action_items[:10]:  # Show top 10
                action = item['action']
                symbol = item['symbol']
                confidence = item['confidence']
                
                if action == 'BUY':
                    amount = item.get('amount', 0)
                    expected_return = item.get('expected_return', 0)
                    print(f"🟢 {action} {symbol}: ${amount:,.0f} "
                          f"(Confidence: {confidence:.1%}, Expected Return: {expected_return:.1f}%)")
                else:
                    print(f"🔴 {action} {symbol} (Confidence: {confidence:.1%})")
                
                # Show top reasoning
                reasoning = item.get('reasoning', [])
                if reasoning:
                    print(f"   Reason: {reasoning[0]}")
                print()
        
        # Key opportunities
        opportunities = summary.get('key_opportunities', [])
        if opportunities:
            print("🎯 KEY OPPORTUNITIES:")
            print("-" * 50)
            for opp in opportunities:
                print(f"• {opp}")
            print()
        
        # Risk warnings
        warnings = summary.get('risk_warnings', [])
        if warnings:
            print("⚠️  RISK WARNINGS:")
            print("-" * 50)
            for warning in warnings:
                print(f"• {warning}")
            print()
        
        # Portfolio metrics
        metrics = summary.get('portfolio_metrics', {})
        if metrics:
            print("📊 PORTFOLIO METRICS:")
            print("-" * 50)
            print(f"Positions Recommended: {metrics.get('total_positions_recommended', 0)}")
            print(f"Total Investment: ${metrics.get('total_amount_to_invest', 0):,.0f}")
            print(f"Cash Remaining: ${metrics.get('cash_remaining', 0):,.0f}")
            print(f"Portfolio Utilization: {metrics.get('portfolio_utilization', 0):.1%}")
        
        print("\n" + "="*80)
        print("End of Report")
        print("="*80)


def main():
    """Main function to run the daily investment advisor."""
    parser = argparse.ArgumentParser(description='Daily Investment Advisor')
    
    parser.add_argument('--symbols', type=str, default='',
                       help='Comma-separated list of symbols to analyze')
    parser.add_argument('--risk-tolerance', type=str, default='moderate',
                       choices=['conservative', 'moderate', 'aggressive'],
                       help='Risk tolerance level')
    parser.add_argument('--portfolio-value', type=float, default=100000,
                       help='Total portfolio value')
    parser.add_argument('--output-format', type=str, default='console',
                       choices=['console', 'json', 'csv'],
                       help='Output format')
    parser.add_argument('--train-models', action='store_true',
                       help='Force retrain ML models')
    parser.add_argument('--import-portfolio', type=str, default='',
                       help='Path to CSV file to import portfolio data')
    parser.add_argument('--manual-entry', action='store_true',
                       help='Enter portfolio positions manually')
    parser.add_argument('--show-portfolio', action='store_true',
                       help='Show current portfolio summary')
    
    args = parser.parse_args()
    
    try:
        # Initialize advisor
        advisor = DailyInvestmentAdvisor(
            risk_tolerance=args.risk_tolerance,
            portfolio_value=args.portfolio_value
        )
        
        # Handle portfolio management commands
        if args.import_portfolio:
            if os.path.exists(args.import_portfolio):
                logger.info(f"Importing portfolio from {args.import_portfolio}")
                result = advisor.portfolio_manager.import_from_csv_export(args.import_portfolio)
                if 'error' not in result:
                    logger.info("Portfolio import successful!")
                else:
                    logger.error(f"Import failed: {result['error']}")
                    return
            else:
                logger.error(f"File not found: {args.import_portfolio}")
                return
        
        if args.manual_entry:
            logger.info("Starting manual portfolio entry...")
            advisor.portfolio_manager.manual_position_entry()
            
        if args.show_portfolio:
            summary = advisor.portfolio_manager.get_current_portfolio_summary()
            print(f"\nCurrent Portfolio Summary:")
            print(f"Total Positions: {summary['total_positions']}")
            print(f"Total Value: ${summary['total_market_value']:,.2f}")
            print(f"Total Return: {summary['total_return_percent']:.2f}%")
            
            if summary['positions']:
                print("\nCurrent Positions:")
                for symbol, pos in summary['positions'].items():
                    weight = pos['weight'] * 100
                    value = pos['market_value']
                    print(f"  {symbol}: ${value:,.2f} ({weight:.1f}%)")
            return
        
        # Determine symbols to analyze
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        else:
            symbols = advisor.default_watchlist
        
        logger.info(f"Starting analysis with {len(symbols)} symbols")
        
        # Force model retraining if requested
        if args.train_models:
            logger.info("Force retraining models...")
            price_data = advisor.fetch_market_data(symbols)
            advisor.train_models(price_data)
        
        # Run analysis
        results = advisor.analyze_investments(symbols)
        
        # Output results
        advisor.output_results(results, args.output_format)
        
        logger.info("Daily investment analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
