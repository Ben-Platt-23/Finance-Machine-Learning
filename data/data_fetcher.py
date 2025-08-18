"""
Data fetching module for stock prices, market indicators, and economic data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches financial data from various sources."""
    
    def __init__(self):
        self.market_indices = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq ETF', 
            'IWM': 'Russell 2000 ETF',
            'VIX': 'Volatility Index',
            'TLT': '20+ Year Treasury Bond ETF',
            'GLD': 'Gold ETF',
            'DXY': 'US Dollar Index'
        }
        
    def get_stock_data(self, 
                      symbols: List[str], 
                      period: str = "2y",
                      interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for given symbols.
        
        Args:
            symbols: List of stock symbols
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Clean column names
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    data[symbol] = df
                    logger.info(f"Fetched data for {symbol}: {len(df)} records")
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                
        return data
    
    def get_market_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch market indices data.
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with market indices data
        """
        indices_data = self.get_stock_data(list(self.market_indices.keys()), period)
        
        # Combine closing prices
        market_df = pd.DataFrame()
        for symbol, df in indices_data.items():
            if not df.empty:
                market_df[symbol] = df['close']
                
        return market_df
    
    def get_sector_etfs(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch sector ETF data for market analysis.
        
        Args:
            period: Data period
            
        Returns:
            Dictionary of sector ETF data
        """
        sector_etfs = [
            'XLK',  # Technology
            'XLF',  # Financial
            'XLV',  # Healthcare
            'XLE',  # Energy
            'XLI',  # Industrial
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLB',  # Materials
            'XLU',  # Utilities
            'XLRE'  # Real Estate
        ]
        
        return self.get_stock_data(sector_etfs, period)
    
    def get_economic_indicators(self) -> Dict[str, float]:
        """
        Fetch key economic indicators (using free sources).
        
        Returns:
            Dictionary of economic indicators
        """
        indicators = {}
        
        try:
            # VIX (Fear & Greed indicator)
            vix_data = self.get_stock_data(['VIX'], period='5d')
            if 'VIX' in vix_data and not vix_data['VIX'].empty:
                indicators['vix'] = vix_data['VIX']['close'].iloc[-1]
            
            # USD Index
            dxy_data = self.get_stock_data(['DX-Y.NYB'], period='5d')
            if 'DX-Y.NYB' in dxy_data and not dxy_data['DX-Y.NYB'].empty:
                indicators['usd_index'] = dxy_data['DX-Y.NYB']['close'].iloc[-1]
                
            # 10-Year Treasury Yield (using TLT as proxy)
            tlt_data = self.get_stock_data(['TLT'], period='5d')
            if 'TLT' in tlt_data and not tlt_data['TLT'].empty:
                indicators['treasury_proxy'] = tlt_data['TLT']['close'].iloc[-1]
                
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            
        return indicators
    
    def get_crypto_data(self, symbols: List[str] = None, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data.
        
        Args:
            symbols: List of crypto symbols (with -USD suffix)
            period: Data period
            
        Returns:
            Dictionary of crypto data
        """
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD']
            
        return self.get_stock_data(symbols, period)
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information and key metrics.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            key_metrics = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'profit_margin': info.get('profitMargins'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            return key_metrics
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    def get_insider_trading(self, symbol: str) -> pd.DataFrame:
        """
        Get insider trading data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with insider trading data
        """
        try:
            ticker = yf.Ticker(symbol)
            insider_trades = ticker.insider_transactions
            return insider_trades if insider_trades is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching insider trading for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_earnings_calendar(self, symbol: str) -> pd.DataFrame:
        """
        Get earnings calendar data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with earnings data
        """
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.calendar
            return earnings if earnings is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
            return pd.DataFrame()


def main():
    """Example usage of DataFetcher."""
    fetcher = DataFetcher()
    
    # Test with some popular stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print("Fetching stock data...")
    stock_data = fetcher.get_stock_data(symbols, period='1mo')
    
    for symbol, df in stock_data.items():
        print(f"\n{symbol}: {len(df)} records")
        print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
    
    print("\nFetching market data...")
    market_data = fetcher.get_market_data(period='1mo')
    print(f"Market indices: {list(market_data.columns)}")
    
    print("\nFetching economic indicators...")
    econ_data = fetcher.get_economic_indicators()
    for indicator, value in econ_data.items():
        print(f"{indicator}: {value:.2f}")


if __name__ == "__main__":
    main()
