# Investment Decision Model

An intelligent investment decision system that analyzes market data and provides daily investment recommendations using machine learning. Designed to work with your Robinhood portfolio for personalized investment advice.

## 🚀 Features

- **📊 Robinhood Integration**: Import your current portfolio positions directly from Robinhood CSV exports
- **🤖 Machine Learning**: Advanced ML models predict price movements and market trends
- **📈 Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, and custom signals
- **⚖️ Risk Management**: Portfolio optimization, position sizing, and risk assessment
- **🎯 Daily Recommendations**: Specific buy/sell/hold decisions with confidence scores and dollar amounts
- **📉 Backtesting**: Validate strategy performance with comprehensive historical analysis
- **🔄 Automated Pipeline**: Complete end-to-end analysis from data fetching to actionable recommendations

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Finance-Machine-Learning

# Install dependencies
pip install -r requirements.txt

# Run the interactive setup
python setup_portfolio.py
```

## 🏃‍♂️ Quick Start

### Option 1: Interactive Setup (Recommended)
```bash
python setup_portfolio.py
```
This will guide you through:
- Importing your Robinhood portfolio data
- Setting your risk tolerance
- Testing the system
- Configuring for daily use

### Option 2: Manual Setup

1. **Import your portfolio** (choose one):
   ```bash
   # From Robinhood CSV export
   python daily_investment_advisor.py --import-portfolio robinhood_export.csv
   
   # Manual entry
   python daily_investment_advisor.py --manual-entry
   ```

2. **Run daily analysis**:
   ```bash
   python daily_investment_advisor.py
   ```

3. **View your portfolio**:
   ```bash
   python daily_investment_advisor.py --show-portfolio
   ```

## 📋 Getting Your Robinhood Data

### Method 1: CSV Export (Recommended)
1. Go to [robinhood.com](https://robinhood.com) and log in
2. Click **Account** → **Settings**
3. Scroll to **Account Information**
4. Click **Export Account Data**
5. Wait for email with download link
6. Download and import the CSV file

### Method 2: Manual Entry
- Use the `--manual-entry` flag to input positions manually
- Good for small portfolios or quick testing

## 🎯 Daily Usage

### Get Investment Recommendations
```bash
# Basic analysis with your portfolio
python daily_investment_advisor.py

# Analyze specific stocks
python daily_investment_advisor.py --symbols AAPL,GOOGL,MSFT

# Adjust risk tolerance
python daily_investment_advisor.py --risk-tolerance aggressive

# Save results to file
python daily_investment_advisor.py --output-format json
```

### Portfolio Management
```bash
# View current portfolio
python daily_investment_advisor.py --show-portfolio

# Import new data
python daily_investment_advisor.py --import-portfolio new_data.csv

# Force model retraining
python daily_investment_advisor.py --train-models
```

### Backtesting
```bash
# Test strategy performance
python backtesting/strategy_backtester.py --start-date 2023-01-01 --end-date 2024-01-01

# Test with your symbols
python backtesting/strategy_backtester.py --symbols AAPL,GOOGL,MSFT --capital 50000
```

## 📊 Example Output

```
================================================================================
DAILY INVESTMENT ADVISOR REPORT
================================================================================
Analysis Date: 2024-01-15T09:30:00
Risk Tolerance: MODERATE
Portfolio Value: $125,430
Symbols Analyzed: 15

Market Regime: BULL

📈 RECOMMENDED ACTIONS:
──────────────────────────────────────────────────────────────────────────────
🟢 BUY NVDA: $8,762 (Confidence: 85%, Expected Return: 12.3%)
   Reason: Strong momentum with ML models showing bullish divergence

🟢 BUY GOOGL: $6,234 (Confidence: 78%, Expected Return: 8.7%)
   Reason: Technical breakout above resistance with high volume

🔴 SELL TSLA (Confidence: 82%)
   Reason: Overbought conditions with bearish MACD divergence

🎯 KEY OPPORTUNITIES:
──────────────────────────────────────────────────────────────────────────────
• AAPL: High confidence buy signal (0.89 confidence, 15.2% expected return)
• META: Technical breakout with volume confirmation

📊 PORTFOLIO METRICS:
──────────────────────────────────────────────────────────────────────────────
Positions Recommended: 8
Total Investment: $45,230
Cash Remaining: $80,200
Portfolio Utilization: 36.1%
```

## 🏗️ Project Structure

```
Finance-Machine-Learning/
├── data/                          # Data fetching and processing
│   ├── data_fetcher.py           # Market data retrieval
│   ├── robinhood_integration.py  # Portfolio import/export
│   └── __init__.py
├── analysis/                      # Technical analysis
│   └── technical_indicators.py   # 20+ technical indicators
├── models/                        # Machine learning
│   ├── ml_models.py              # Price prediction models
│   ├── decision_engine.py        # Signal combination logic
│   └── __init__.py
├── risk/                          # Risk management
│   └── portfolio_optimizer.py    # Position sizing & optimization
├── backtesting/                   # Strategy validation
│   └── strategy_backtester.py    # Historical performance testing
├── config/                        # Configuration files
├── notebooks/                     # Jupyter notebooks for research
├── daily_investment_advisor.py   # Main analysis script
├── setup_portfolio.py           # Interactive setup
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## ⚙️ Configuration Options

### Risk Tolerance Levels
- **Conservative**: 5% max position size, focus on stability
- **Moderate**: 10% max position size, balanced approach
- **Aggressive**: 20% max position size, growth-focused

### Command Line Options
```bash
--symbols AAPL,GOOGL          # Specific symbols to analyze
--risk-tolerance moderate     # conservative/moderate/aggressive
--portfolio-value 100000      # Override portfolio value
--output-format json          # json/csv/console
--import-portfolio file.csv   # Import portfolio data
--manual-entry               # Enter positions manually
--show-portfolio            # Display current positions
--train-models             # Force model retraining
```

## 🧪 Testing and Validation

### Backtest Your Strategy
```bash
# Full backtest
python backtesting/strategy_backtester.py \
  --start-date 2022-01-01 \
  --end-date 2023-12-31 \
  --symbols AAPL,GOOGL,MSFT,AMZN,TSLA \
  --capital 100000 \
  --frequency weekly
```

### Model Performance
The system tracks and reports:
- Prediction accuracy
- Sharpe ratio
- Maximum drawdown
- Win rate
- Risk-adjusted returns

## 🔒 Security and Privacy

- **No API keys required**: Uses free data sources by default
- **Local data storage**: All portfolio data stored locally
- **No credentials stored**: Never stores login information
- **Encrypted backups**: Optional encryption for sensitive data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📈 Performance Notes

- **Data Sources**: Yahoo Finance (free, reliable)
- **Update Frequency**: Daily analysis recommended
- **Processing Time**: ~30-60 seconds for 20 stocks
- **Memory Usage**: ~500MB during analysis
- **Storage**: ~50MB for 2 years of data

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. It is NOT financial advice. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.

## 📞 Support

- Check the [Issues](../../issues) page for common problems
- Review the code comments for detailed explanations
- Run `python setup_portfolio.py` for interactive troubleshooting

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
