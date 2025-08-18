# Investment Decision Model

An intelligent investment decision system that analyzes market data and provides daily investment recommendations using machine learning.

## Features

- **Real-time Data Analysis**: Fetches and analyzes stock prices, market indicators, and economic data
- **Technical Indicators**: Calculates RSI, MACD, Bollinger Bands, and other technical signals
- **Machine Learning Models**: Uses ensemble methods to predict price movements and trends
- **Risk Management**: Implements portfolio optimization and risk assessment
- **Daily Recommendations**: Provides actionable buy/sell/hold decisions with confidence scores
- **Backtesting**: Validates strategy performance with historical data

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Run the daily analysis:
```bash
python daily_investment_advisor.py
```

2. For backtesting:
```bash
python backtest_strategy.py --start-date 2023-01-01 --end-date 2024-01-01
```

3. Launch the web interface:
```bash
streamlit run dashboard.py
```

## Project Structure

- `data/` - Data fetching and processing modules
- `models/` - Machine learning models and training scripts  
- `analysis/` - Technical analysis and feature engineering
- `risk/` - Risk management and portfolio optimization
- `backtesting/` - Strategy validation and performance metrics
- `notebooks/` - Jupyter notebooks for research and exploration

## Configuration

Create a `.env` file with any API keys for premium data sources (optional - works with free data by default).

## Disclaimer

This is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.
