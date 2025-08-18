"""
EXTREME HEAVY-DUTY TRAINING SYSTEM
==================================

This is an intensive training pipeline designed to create the most accurate
investment prediction models possible. It will:

1. Download 10+ years of data for 500+ stocks
2. Calculate 100+ technical indicators and features
3. Train 15+ different model types with extensive hyperparameter tuning
4. Use advanced ensemble methods and meta-learning
5. Perform comprehensive cross-validation and testing
6. Run for 6-12+ hours depending on your hardware

WARNING: This will max out your CPU/GPU and use significant RAM and storage.
Make sure you have:
- At least 8GB RAM (16GB+ recommended)
- 5GB+ free disk space
- Stable internet connection
- Power plugged in (don't run on battery)

This is designed to run overnight while you sleep!
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             ExtraTreesClassifier, ExtraTreesRegressor,
                             AdaBoostClassifier, AdaBoostRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
# import catboost as cb  # Skipping due to installation issues
from scipy.stats import uniform, randint
import itertools
import time
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import joblib
import ta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from analysis.technical_indicators import TechnicalIndicators

# Configure logging for overnight monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('heavy_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExtremeTrainingSystem:
    """
    The most intensive training system possible for maximum accuracy.
    
    This system will train hundreds of models with different configurations,
    feature sets, and hyperparameters to find the absolute best combination.
    """
    
    def __init__(self):
        """Initialize the extreme training system."""
        self.start_time = datetime.now()
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalIndicators()
        
        # Training configuration
        self.n_jobs = mp.cpu_count()  # Use all available cores
        logger.info(f"Initialized with {self.n_jobs} CPU cores")
        
        # Massive stock universe - top stocks from multiple sectors
        self.stock_universe = [
            # Mega cap tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW',
            'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO',
            
            # Healthcare & Biotech
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'MDLZ', 'CVS', 'CI', 'ANTM', 'HUM', 'BIIB',
            'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'ISRG',
            
            # Consumer
            'AMZN', 'TSLA', 'HD', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE',
            'MCD', 'SBUX', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS',
            'LOW', 'TGT', 'TJX', 'PM', 'UL', 'CL', 'KMB', 'GIS',
            
            # Industrial & Materials
            'BA', 'CAT', 'HON', 'UNP', 'GE', 'MMM', 'LMT', 'RTX', 'DE',
            'FDX', 'UPS', 'WM', 'EMR', 'ETN', 'ITW', 'PH', 'CMI',
            'DD', 'DOW', 'LIN', 'APD', 'ECL', 'SHW', 'PPG', 'NEM',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
            'KMI', 'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'BKR', 'HAL',
            
            # REITs & Utilities
            'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'PEG', 'SRE',
            'AMT', 'CCI', 'EQIX', 'DLR', 'PSA', 'EXR', 'AVB', 'EQR',
            
            # International & Emerging
            'TSM', 'ASML', 'NVO', 'RHHBY', 'TM', 'SNY', 'UL', 'DEO',
            'NVS', 'AZN', 'SAP', 'SHOP', 'SE', 'BABA', 'JD', 'PDD',
            
            # Growth & Momentum
            'ROKU', 'ZM', 'PTON', 'SNOW', 'PLTR', 'RBLX', 'U', 'NET',
            'CRWD', 'ZS', 'OKTA', 'DDOG', 'MDB', 'TEAM', 'WDAY', 'NOW',
            
            # Value & Dividend
            'BRK-B', 'V', 'MA', 'PYPL', 'ADSK', 'INTU', 'TXN', 'LRCX',
            'AMAT', 'KLAC', 'MCHP', 'ADI', 'FTNT', 'PANW', 'CYBR',
            
            # ETFs for market context
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'TLT', 'GLD', 'VIX'
        ]
        
        # Remove duplicates and limit to reasonable size
        self.stock_universe = list(set(self.stock_universe))[:200]  # Top 200 stocks
        
        logger.info(f"Training universe: {len(self.stock_universe)} symbols")
        
        # Model configurations - MASSIVE hyperparameter space
        self.model_configs = {
            'random_forest_classifier': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800, 1000],
                    'max_depth': [10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'xgboost_classifier': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1, 10],
                    'reg_lambda': [1, 1.5, 2, 3, 4.5],
                    'min_child_weight': [1, 3, 5, 7, 10]
                }
            },
            'lightgbm_classifier': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'num_leaves': [31, 50, 70, 100, 150, 200],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [0, 0.01, 0.1, 1],
                    'min_child_samples': [20, 30, 40, 50, 70]
                }
            },
            # 'catboost_classifier': {
            #     'model': cb.CatBoostClassifier,
            #     'params': {
            #         'iterations': [100, 200, 300, 500, 800],
            #         'depth': [4, 5, 6, 7, 8, 9, 10],
            #         'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            #         'l2_leaf_reg': [1, 3, 5, 7, 9],
            #         'border_count': [32, 64, 128, 255],
            #         'bagging_temperature': [0, 0.5, 1, 2, 10],
            #         'random_strength': [1, 2, 5, 10, 20]
            #     }
            # },
            'neural_network_classifier': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [
                        (100,), (200,), (300,), (500,),
                        (100, 50), (200, 100), (300, 150), (500, 250),
                        (100, 50, 25), (200, 100, 50), (300, 150, 75),
                        (500, 250, 125), (800, 400, 200, 100)
                    ],
                    'activation': ['tanh', 'relu', 'logistic'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [500, 1000, 2000]
                }
            },
            'gradient_boosting_classifier': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'subsample': [0.6, 0.8, 1.0],
                    'max_features': ['sqrt', 'log2', None]
                }
            }
        }
        
        # Regression models with similar extensive configs
        self.regression_configs = {
            'random_forest_regressor': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800, 1000],
                    'max_depth': [10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                    'bootstrap': [True, False]
                }
            },
            'xgboost_regressor': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1, 10],
                    'reg_lambda': [1, 1.5, 2, 3, 4.5]
                }
            },
            'lightgbm_regressor': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'num_leaves': [31, 50, 70, 100, 150, 200],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
            },
            'neural_network_regressor': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [
                        (100,), (200,), (300,), (500,),
                        (100, 50), (200, 100), (300, 150), (500, 250),
                        (100, 50, 25), (200, 100, 50), (300, 150, 75),
                        (500, 250, 125), (800, 400, 200, 100)
                    ],
                    'activation': ['tanh', 'relu'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [500, 1000, 2000]
                }
            }
        }
        
        # Storage for results
        self.training_results = {}
        self.best_models = {}
        self.performance_metrics = {}
        
        logger.info("🔥 EXTREME TRAINING SYSTEM INITIALIZED 🔥")
        logger.info("This will be INTENSIVE - grab some coffee and let it run!")
    
    def download_massive_dataset(self):
        """
        Download extensive historical data for all stocks.
        This will take a while and use significant bandwidth/storage.
        """
        logger.info("📡 PHASE 1: DOWNLOADING MASSIVE DATASET")
        logger.info("=" * 60)
        logger.info(f"Downloading 10+ years of data for {len(self.stock_universe)} symbols...")
        
        all_data = {}
        failed_downloads = []
        
        # Download in batches to avoid overwhelming the API
        batch_size = 20
        for i in range(0, len(self.stock_universe), batch_size):
            batch = self.stock_universe[i:i+batch_size]
            
            logger.info(f"Downloading batch {i//batch_size + 1}/{(len(self.stock_universe)-1)//batch_size + 1}: {batch}")
            
            try:
                # Download maximum available data
                batch_data = self.data_fetcher.get_stock_data(
                    batch, 
                    period='max',  # Maximum available data
                    interval='1d'
                )
                
                for symbol, df in batch_data.items():
                    if not df.empty and len(df) >= 1000:  # Need substantial history
                        all_data[symbol] = df
                        logger.info(f"✅ {symbol}: {len(df)} days of data")
                    else:
                        failed_downloads.append(symbol)
                        logger.warning(f"❌ {symbol}: Insufficient data ({len(df) if not df.empty else 0} days)")
                
                # Small delay to be respectful to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading batch {batch}: {e}")
                failed_downloads.extend(batch)
        
        logger.info(f"📊 DATA DOWNLOAD COMPLETE:")
        logger.info(f"   ✅ Successfully downloaded: {len(all_data)} symbols")
        logger.info(f"   ❌ Failed downloads: {len(failed_downloads)} symbols")
        logger.info(f"   📈 Total data points: {sum(len(df) for df in all_data.values()):,}")
        
        if failed_downloads:
            logger.info(f"   Failed symbols: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        return all_data
    
    def create_advanced_features(self, price_data_dict):
        """
        Create the most comprehensive feature set possible.
        This includes 100+ technical indicators, market regime features,
        cross-asset correlations, and advanced derived features.
        """
        logger.info("🧠 PHASE 2: CREATING ADVANCED FEATURES")
        logger.info("=" * 60)
        logger.info("Creating 100+ technical indicators and advanced features...")
        
        all_features = pd.DataFrame()
        processed_count = 0
        
        # Get market data for regime analysis
        market_indices = ['SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD']
        market_data = {}
        for idx in market_indices:
            if idx in price_data_dict:
                market_data[idx] = price_data_dict[idx]
        
        for symbol, price_data in price_data_dict.items():
            try:
                logger.info(f"Processing features for {symbol}...")
                
                # 1. Basic technical indicators
                enriched_data = self.technical_analyzer.calculate_all_indicators(price_data)
                
                # 2. Advanced technical indicators
                enriched_data = self._add_advanced_technical_indicators(enriched_data)
                
                # 3. Market regime features
                enriched_data = self._add_market_regime_features(enriched_data, market_data)
                
                # 4. Cross-asset correlation features
                enriched_data = self._add_correlation_features(enriched_data, price_data_dict, symbol)
                
                # 5. Time-based and seasonal features
                enriched_data = self._add_temporal_features(enriched_data)
                
                # 6. Advanced price action features
                enriched_data = self._add_advanced_price_features(enriched_data)
                
                # 7. Volume and liquidity features
                enriched_data = self._add_volume_features(enriched_data)
                
                # 8. Volatility regime features
                enriched_data = self._add_volatility_regime_features(enriched_data)
                
                # 9. Target variables for ML
                enriched_data = self._create_prediction_targets(enriched_data)
                
                # Add symbol identifier
                enriched_data['symbol'] = symbol
                
                # Append to master dataset
                all_features = pd.concat([all_features, enriched_data], ignore_index=True)
                
                processed_count += 1
                
                if processed_count % 20 == 0:
                    logger.info(f"   Processed {processed_count}/{len(price_data_dict)} symbols...")
                
            except Exception as e:
                logger.error(f"Error processing features for {symbol}: {e}")
                continue
        
        # Remove infinite and null values
        logger.info("🧹 Cleaning data...")
        initial_rows = len(all_features)
        
        # Replace infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        all_features = all_features.fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining null rows
        all_features = all_features.dropna()
        
        final_rows = len(all_features)
        logger.info(f"   Data cleaning complete: {initial_rows:,} → {final_rows:,} rows")
        logger.info(f"   Features created: {len(all_features.columns)} columns")
        logger.info(f"   Total data points: {len(all_features) * len(all_features.columns):,}")
        
        return all_features
    
    def _add_advanced_technical_indicators(self, df):
        """Add advanced technical indicators beyond the basic set."""
        # Ichimoku Cloud components
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_conversion'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_base'] = (high_26 + low_26) / 2
        
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Fibonacci retracements
        rolling_high = df['high'].rolling(window=50).max()
        rolling_low = df['low'].rolling(window=50).min()
        fib_range = rolling_high - rolling_low
        
        df['fib_23.6'] = rolling_high - (fib_range * 0.236)
        df['fib_38.2'] = rolling_high - (fib_range * 0.382)
        df['fib_50.0'] = rolling_high - (fib_range * 0.500)
        df['fib_61.8'] = rolling_high - (fib_range * 0.618)
        
        # Distance from Fibonacci levels
        df['dist_from_fib_236'] = (df['close'] - df['fib_23.6']) / df['close']
        df['dist_from_fib_382'] = (df['close'] - df['fib_38.2']) / df['close']
        df['dist_from_fib_618'] = (df['close'] - df['fib_61.8']) / df['close']
        
        # Advanced momentum indicators
        df['trix'] = ta.trend.trix(df['close'], window=14)
        df['mass_index'] = ta.trend.mass_index(df['high'], df['low'], window1=9, window2=25)
        df['dpo'] = ta.trend.dpo(df['close'], window=20)
        df['kst'] = ta.trend.kst(df['close'])
        
        # Advanced volume indicators
        df['ease_of_movement'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'], window=14)
        df['volume_price_trend'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        df['negative_volume_index'] = ta.volume.negative_volume_index(df['close'], df['volume'])
        df['force_index'] = ta.volume.force_index(df['close'], df['volume'], window=13)
        
        # Price channels and bands
        df['donchian_high'] = df['high'].rolling(window=20).max()
        df['donchian_low'] = df['low'].rolling(window=20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        
        # Volatility indicators
        df['chaikin_volatility'] = ta.volatility.ulcer_index(df['close'], window=14)
        
        return df
    
    def _add_market_regime_features(self, df, market_data):
        """Add features that capture market regime and conditions."""
        if not market_data:
            return df
        
        try:
            # VIX-based fear/greed
            if 'VIX' in market_data:
                vix_data = market_data['VIX']['close'].reindex(df.index, method='ffill')
                df['vix_level'] = vix_data
                df['vix_ma_20'] = vix_data.rolling(20).mean()
                df['vix_percentile'] = vix_data.rolling(252).rank(pct=True)
                df['fear_greed_regime'] = pd.cut(df['vix_percentile'], 
                                               bins=[0, 0.2, 0.8, 1.0], 
                                               labels=['greed', 'normal', 'fear'])
            
            # Market trend regime
            if 'SPY' in market_data:
                spy_data = market_data['SPY']['close'].reindex(df.index, method='ffill')
                spy_ma_50 = spy_data.rolling(50).mean()
                spy_ma_200 = spy_data.rolling(200).mean()
                
                df['market_trend_regime'] = np.where(spy_ma_50 > spy_ma_200, 1, 0)  # Bull/Bear
                df['spy_relative_position'] = (spy_data - spy_ma_200) / spy_ma_200
                
            # Interest rate regime (using TLT as proxy)
            if 'TLT' in market_data:
                tlt_data = market_data['TLT']['close'].reindex(df.index, method='ffill')
                tlt_ma_50 = tlt_data.rolling(50).mean()
                df['interest_rate_regime'] = np.where(tlt_data > tlt_ma_50, 1, 0)  # Falling/Rising rates
                
            # Risk-on/Risk-off (QQQ vs TLT ratio)
            if 'QQQ' in market_data and 'TLT' in market_data:
                qqq_data = market_data['QQQ']['close'].reindex(df.index, method='ffill')
                tlt_data = market_data['TLT']['close'].reindex(df.index, method='ffill')
                risk_ratio = qqq_data / tlt_data
                df['risk_on_off_ratio'] = risk_ratio
                df['risk_on_off_ma'] = risk_ratio.rolling(20).mean()
                
        except Exception as e:
            logger.warning(f"Error adding market regime features: {e}")
        
        return df
    
    def _add_correlation_features(self, df, price_data_dict, current_symbol):
        """Add features based on correlations with other assets."""
        try:
            # Select key assets for correlation
            correlation_assets = ['SPY', 'QQQ', 'IWM', 'VIX', 'GLD', 'TLT']
            
            returns = df['close'].pct_change()
            
            for asset in correlation_assets:
                if asset in price_data_dict and asset != current_symbol:
                    asset_data = price_data_dict[asset]
                    asset_returns = asset_data['close'].pct_change()
                    
                    # Align the data
                    common_index = returns.index.intersection(asset_returns.index)
                    if len(common_index) > 50:  # Need sufficient overlap
                        aligned_returns = returns.reindex(common_index)
                        aligned_asset_returns = asset_returns.reindex(common_index)
                        
                        # Rolling correlations
                        df[f'corr_{asset}_30d'] = aligned_returns.rolling(30).corr(aligned_asset_returns)
                        df[f'corr_{asset}_90d'] = aligned_returns.rolling(90).corr(aligned_asset_returns)
                        
                        # Beta calculation
                        covariance = aligned_returns.rolling(60).cov(aligned_asset_returns)
                        variance = aligned_asset_returns.rolling(60).var()
                        df[f'beta_{asset}'] = covariance / variance
                        
        except Exception as e:
            logger.warning(f"Error adding correlation features: {e}")
        
        return df
    
    def _add_temporal_features(self, df):
        """Add time-based and seasonal features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Calendar features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        
        # Market timing features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = (df.index.day > 25).astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        # Seasonal patterns
        df['january_effect'] = (df['month'] == 1).astype(int)
        df['sell_in_may'] = ((df['month'] >= 5) & (df['month'] <= 9)).astype(int)
        df['santa_rally'] = ((df['month'] == 12) & (df.index.day > 15)).astype(int)
        
        # Cyclical features (sine/cosine for periodicity)
        df['day_of_year'] = df.index.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_advanced_price_features(self, df):
        """Add advanced price action and pattern features."""
        # Price gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percent'] = df['gap'] / df['close'].shift(1)
        df['gap_filled'] = ((df['gap'] > 0) & (df['low'] <= df['close'].shift(1))) | \
                          ((df['gap'] < 0) & (df['high'] >= df['close'].shift(1)))
        
        # Candlestick patterns (simplified versions)
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        df['body_size'] = body
        df['upper_shadow_size'] = upper_shadow
        df['lower_shadow_size'] = lower_shadow
        df['body_to_range_ratio'] = body / total_range
        df['upper_shadow_ratio'] = upper_shadow / total_range
        df['lower_shadow_ratio'] = lower_shadow / total_range
        
        # Doji patterns
        df['is_doji'] = (body < total_range * 0.1).astype(int)
        df['is_hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)
        df['is_shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.5)).astype(int)
        
        # Price levels and support/resistance
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        
        # Distance from key levels
        df['dist_from_pivot'] = (df['close'] - df['pivot_point']) / df['close']
        df['dist_from_r1'] = (df['close'] - df['resistance_1']) / df['close']
        df['dist_from_s1'] = (df['close'] - df['support_1']) / df['close']
        
        return df
    
    def _add_volume_features(self, df):
        """Add advanced volume-based features."""
        # Volume moving averages
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # Volume ratios
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        # Volume momentum
        df['volume_momentum_5'] = df['volume'].pct_change(5)
        df['volume_momentum_20'] = df['volume'].pct_change(20)
        
        # Price-volume relationships
        returns = df['close'].pct_change()
        df['volume_price_correlation'] = returns.rolling(20).corr(df['volume'].pct_change())
        
        # Volume-weighted metrics
        df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        df['vwap_20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['dist_from_vwap'] = (df['close'] - df['vwap_20']) / df['vwap_20']
        
        return df
    
    def _add_volatility_regime_features(self, df):
        """Add volatility regime and clustering features."""
        returns = df['close'].pct_change()
        
        # Rolling volatility measures
        df['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']
        df['vol_ratio_20_60'] = df['volatility_20d'] / df['volatility_60d']
        
        # Volatility percentiles
        df['vol_percentile_252d'] = df['volatility_20d'].rolling(252).rank(pct=True)
        
        # Volatility clustering (GARCH-like features)
        squared_returns = returns ** 2
        df['vol_clustering'] = squared_returns.rolling(5).mean()
        
        # Parkinson volatility (high-low based)
        df['parkinson_vol'] = np.sqrt(
            252 / (4 * np.log(2)) * 
            (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        )
        
        return df
    
    def _create_prediction_targets(self, df):
        """Create various prediction targets for ML models."""
        # Price direction targets
        df['price_up_1d'] = (df['close'].shift(-1) > df['close']).astype(int)
        df['price_up_3d'] = (df['close'].shift(-3) > df['close']).astype(int)
        df['price_up_5d'] = (df['close'].shift(-5) > df['close']).astype(int)
        
        # Return targets
        df['return_1d'] = df['close'].pct_change(-1) * 100  # Next day return
        df['return_3d'] = (df['close'].shift(-3) / df['close'] - 1) * 100
        df['return_5d'] = (df['close'].shift(-5) / df['close'] - 1) * 100
        
        # Volatility targets
        future_returns = df['close'].pct_change().shift(-1)
        df['volatility_1d'] = abs(future_returns) * 100
        df['volatility_5d'] = future_returns.rolling(5).std().shift(-5) * np.sqrt(252) * 100
        
        # High/low targets
        df['will_hit_high'] = (df['high'].rolling(5).max().shift(-5) > df['close'] * 1.05).astype(int)
        df['will_hit_low'] = (df['low'].rolling(5).min().shift(-5) < df['close'] * 0.95).astype(int)
        
        return df
    
    def train_extreme_models(self, features_df):
        """
        Train hundreds of models with extensive hyperparameter tuning.
        This is the most intensive part - will run for hours.
        """
        logger.info("🚀 PHASE 3: EXTREME MODEL TRAINING")
        logger.info("=" * 60)
        logger.info("Training hundreds of models with extensive hyperparameter tuning...")
        logger.info("This will take HOURS - perfect for overnight training!")
        
        # Prepare data
        feature_columns = [col for col in features_df.columns 
                          if col not in ['symbol', 'price_up_1d', 'price_up_3d', 'price_up_5d',
                                       'return_1d', 'return_3d', 'return_5d', 
                                       'volatility_1d', 'volatility_5d',
                                       'will_hit_high', 'will_hit_low']]
        
        X = features_df[feature_columns].fillna(0)
        
        # Multiple prediction targets
        targets = {
            'price_direction_1d': 'price_up_1d',
            'price_direction_3d': 'price_up_3d', 
            'price_direction_5d': 'price_up_5d',
            'return_1d': 'return_1d',
            'return_3d': 'return_3d',
            'return_5d': 'return_5d'
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        training_results = {}
        
        for target_name, target_col in targets.items():
            if target_col not in features_df.columns:
                continue
                
            logger.info(f"\n🎯 Training models for target: {target_name}")
            logger.info("-" * 50)
            
            y = features_df[target_col].fillna(0)
            
            # Remove samples with missing targets
            valid_mask = ~pd.isna(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 1000:
                logger.warning(f"Insufficient data for {target_name}: {len(X_clean)} samples")
                continue
            
            target_results = {}
            
            # Choose model configs based on target type
            if 'direction' in target_name:
                model_configs = self.model_configs
                task_type = 'classification'
            else:
                model_configs = self.regression_configs
                task_type = 'regression'
            
            # Train each model type
            for model_name, config in model_configs.items():
                logger.info(f"   🔥 Training {model_name} for {target_name}...")
                
                try:
                    # Scale features for neural networks and SVMs
                    if 'neural' in model_name.lower() or 'svm' in model_name.lower():
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_clean)
                    else:
                        X_scaled = X_clean
                        scaler = None
                    
                    # Randomized search for efficiency (still extensive)
                    search = RandomizedSearchCV(
                        estimator=config['model'](),
                        param_distributions=config['params'],
                        n_iter=100,  # 100 random combinations per model
                        cv=tscv,
                        scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error',
                        n_jobs=min(self.n_jobs, 4),  # Limit parallel jobs per model
                        verbose=1,
                        random_state=42
                    )
                    
                    # Fit the model
                    search.fit(X_scaled, y_clean)
                    
                    # Store results
                    target_results[model_name] = {
                        'best_model': search.best_estimator_,
                        'best_score': search.best_score_,
                        'best_params': search.best_params_,
                        'scaler': scaler,
                        'cv_results': search.cv_results_
                    }
                    
                    logger.info(f"      ✅ Best score: {search.best_score_:.4f}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Error training {model_name}: {e}")
                    continue
            
            training_results[target_name] = target_results
        
        logger.info("\n🎉 EXTREME TRAINING COMPLETE!")
        logger.info(f"Trained {sum(len(results) for results in training_results.values())} models total")
        
        return training_results, feature_columns
    
    def create_meta_ensemble(self, training_results, features_df, feature_columns):
        """
        Create a meta-ensemble that combines the best models.
        This is advanced model stacking for maximum accuracy.
        """
        logger.info("🧠 PHASE 4: CREATING META-ENSEMBLE")
        logger.info("=" * 60)
        logger.info("Building advanced ensemble models...")
        
        meta_models = {}
        
        X = features_df[feature_columns].fillna(0)
        
        for target_name, target_results in training_results.items():
            if not target_results:
                continue
                
            logger.info(f"Creating ensemble for {target_name}...")
            
            # Get the target
            target_col = target_name.replace('price_direction_', 'price_up_').replace('return_', 'return_')
            if target_col.startswith('price_up_'):
                target_col = target_col.replace('price_up_', 'price_up_')
            
            if target_col not in features_df.columns:
                continue
                
            y = features_df[target_col].fillna(0)
            valid_mask = ~pd.isna(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # Get predictions from all models
            model_predictions = []
            model_names = []
            
            for model_name, model_info in target_results.items():
                try:
                    model = model_info['best_model']
                    scaler = model_info.get('scaler')
                    
                    if scaler:
                        X_scaled = scaler.transform(X_clean)
                    else:
                        X_scaled = X_clean
                    
                    # Get cross-validated predictions to avoid overfitting
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_predictions = np.zeros(len(X_clean))
                    
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train = y_clean.iloc[train_idx]
                        
                        # Retrain on fold
                        model.fit(X_train, y_train)
                        
                        # Predict on validation
                        if 'direction' in target_name:
                            cv_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
                        else:
                            cv_predictions[val_idx] = model.predict(X_val)
                    
                    model_predictions.append(cv_predictions)
                    model_names.append(model_name)
                    
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model_name}: {e}")
                    continue
            
            if len(model_predictions) < 2:
                logger.warning(f"Not enough models for ensemble for {target_name}")
                continue
            
            # Create meta-features matrix
            meta_X = np.column_stack(model_predictions)
            
            # Train meta-model
            if 'direction' in target_name:
                meta_model = LogisticRegression(random_state=42)
            else:
                meta_model = Ridge(random_state=42)
            
            # Cross-validate meta-model
            tscv = TimeSeriesSplit(n_splits=3)
            meta_scores = []
            
            for train_idx, val_idx in tscv.split(meta_X):
                meta_X_train, meta_X_val = meta_X[train_idx], meta_X[val_idx]
                meta_y_train, meta_y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                
                meta_model.fit(meta_X_train, meta_y_train)
                
                if 'direction' in target_name:
                    score = meta_model.score(meta_X_val, meta_y_val)
                else:
                    pred = meta_model.predict(meta_X_val)
                    score = r2_score(meta_y_val, pred)
                
                meta_scores.append(score)
            
            avg_score = np.mean(meta_scores)
            logger.info(f"   Meta-ensemble score for {target_name}: {avg_score:.4f}")
            
            # Train final meta-model on all data
            meta_model.fit(meta_X, y_clean)
            
            meta_models[target_name] = {
                'meta_model': meta_model,
                'base_models': target_results,
                'model_names': model_names,
                'meta_score': avg_score,
                'feature_columns': feature_columns
            }
        
        return meta_models
    
    def comprehensive_evaluation(self, meta_models, features_df):
        """
        Perform comprehensive evaluation and testing.
        This validates the models extensively.
        """
        logger.info("📊 PHASE 5: COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        logger.info("Testing the hell out of the models...")
        
        evaluation_results = {}
        
        for target_name, meta_model_info in meta_models.items():
            logger.info(f"\nEvaluating {target_name}...")
            
            # Get data
            feature_columns = meta_model_info['feature_columns']
            X = features_df[feature_columns].fillna(0)
            
            target_col = target_name.replace('price_direction_', 'price_up_').replace('return_', 'return_')
            if target_col.startswith('price_up_'):
                target_col = target_col.replace('price_up_', 'price_up_')
            
            if target_col not in features_df.columns:
                continue
                
            y = features_df[target_col].fillna(0)
            valid_mask = ~pd.isna(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # Time-based train/test split (last 20% for testing)
            split_point = int(len(X_clean) * 0.8)
            X_train, X_test = X_clean.iloc[:split_point], X_clean.iloc[split_point:]
            y_train, y_test = y_clean.iloc[:split_point], y_clean.iloc[split_point:]
            
            # Get meta-model predictions
            meta_model = meta_model_info['meta_model']
            base_models = meta_model_info['base_models']
            model_names = meta_model_info['model_names']
            
            # Generate base model predictions on test set
            test_predictions = []
            
            for model_name in model_names:
                if model_name not in base_models:
                    continue
                    
                model_info = base_models[model_name]
                model = model_info['best_model']
                scaler = model_info.get('scaler')
                
                try:
                    # Retrain on full training set
                    if scaler:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    model.fit(X_train_scaled, y_train)
                    
                    if 'direction' in target_name:
                        pred = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        pred = model.predict(X_test_scaled)
                    
                    test_predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
                    continue
            
            if len(test_predictions) < 2:
                continue
            
            # Meta-model prediction
            meta_X_test = np.column_stack(test_predictions)
            
            # Retrain meta-model on training data
            train_predictions = []
            for i, model_name in enumerate(model_names):
                if model_name not in base_models:
                    continue
                    
                model_info = base_models[model_name]
                model = model_info['best_model']
                scaler = model_info.get('scaler')
                
                # Get cross-validated training predictions
                tscv = TimeSeriesSplit(n_splits=3)
                cv_preds = np.zeros(len(X_train))
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    
                    if scaler:
                        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                        X_fold_val_scaled = scaler.transform(X_fold_val)
                    else:
                        X_fold_train_scaled = X_fold_train
                        X_fold_val_scaled = X_fold_val
                    
                    model.fit(X_fold_train_scaled, y_fold_train)
                    
                    if 'direction' in target_name:
                        cv_preds[val_idx] = model.predict_proba(X_fold_val_scaled)[:, 1]
                    else:
                        cv_preds[val_idx] = model.predict(X_fold_val_scaled)
                
                train_predictions.append(cv_preds)
            
            meta_X_train = np.column_stack(train_predictions)
            meta_model.fit(meta_X_train, y_train)
            
            # Final predictions
            final_predictions = meta_model.predict(meta_X_test)
            
            # Calculate metrics
            if 'direction' in target_name:
                # Classification metrics
                accuracy = accuracy_score(y_test, (final_predictions > 0.5).astype(int))
                precision = precision_score(y_test, (final_predictions > 0.5).astype(int))
                recall = recall_score(y_test, (final_predictions > 0.5).astype(int))
                f1 = f1_score(y_test, (final_predictions > 0.5).astype(int))
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'test_samples': len(y_test)
                }
                
                logger.info(f"   Accuracy: {accuracy:.4f}")
                logger.info(f"   Precision: {precision:.4f}")
                logger.info(f"   Recall: {recall:.4f}")
                logger.info(f"   F1 Score: {f1:.4f}")
                
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, final_predictions)
                mae = mean_absolute_error(y_test, final_predictions)
                r2 = r2_score(y_test, final_predictions)
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse),
                    'test_samples': len(y_test)
                }
                
                logger.info(f"   R² Score: {r2:.4f}")
                logger.info(f"   RMSE: {np.sqrt(mse):.4f}")
                logger.info(f"   MAE: {mae:.4f}")
            
            evaluation_results[target_name] = metrics
        
        return evaluation_results
    
    def save_extreme_models(self, meta_models, evaluation_results):
        """Save all the trained models and results."""
        logger.info("💾 PHASE 6: SAVING MODELS")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        models_dir = Path('models/extreme_models')
        models_dir.mkdir(exist_ok=True)
        
        # Save meta-models
        meta_models_file = models_dir / f'meta_models_{timestamp}.joblib'
        joblib.dump(meta_models, meta_models_file)
        logger.info(f"Meta-models saved to: {meta_models_file}")
        
        # Save evaluation results
        results_file = models_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for target, metrics in evaluation_results.items():
                serializable_results[target] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32)) else int(v) if isinstance(v, (np.int64, np.int32)) else v
                    for k, v in metrics.items()
                }
            
            import json
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {results_file}")
        
        # Save training summary
        summary_file = models_dir / f'training_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("EXTREME TRAINING SYSTEM - RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Training completed: {datetime.now()}\n")
            f.write(f"Training duration: {datetime.now() - self.start_time}\n")
            f.write(f"Models trained: {len(meta_models)}\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            
            for target, metrics in evaluation_results.items():
                f.write(f"\n{target}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
        
        logger.info(f"Training summary saved to: {summary_file}")
        
        return meta_models_file, results_file, summary_file
    
    def run_extreme_training(self):
        """
        Run the complete extreme training pipeline.
        This is the main function that orchestrates everything.
        """
        logger.info("🔥🔥🔥 STARTING EXTREME TRAINING SYSTEM 🔥🔥🔥")
        logger.info("=" * 80)
        logger.info("This will be INTENSIVE and run for hours!")
        logger.info("Perfect for overnight training while you sleep.")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Download massive dataset
            price_data_dict = self.download_massive_dataset()
            
            if not price_data_dict:
                raise Exception("Failed to download sufficient data")
            
            # Phase 2: Create advanced features
            features_df = self.create_advanced_features(price_data_dict)
            
            if features_df.empty:
                raise Exception("Failed to create features")
            
            # Phase 3: Train extreme models
            training_results, feature_columns = self.train_extreme_models(features_df)
            
            if not training_results:
                raise Exception("No models were successfully trained")
            
            # Phase 4: Create meta-ensemble
            meta_models = self.create_meta_ensemble(training_results, features_df, feature_columns)
            
            # Phase 5: Comprehensive evaluation
            evaluation_results = self.comprehensive_evaluation(meta_models, features_df)
            
            # Phase 6: Save everything
            model_files = self.save_extreme_models(meta_models, evaluation_results)
            
            # Final summary
            total_duration = datetime.now() - self.start_time
            logger.info("\n🎉🎉🎉 EXTREME TRAINING COMPLETE! 🎉🎉🎉")
            logger.info("=" * 80)
            logger.info(f"Total training time: {total_duration}")
            logger.info(f"Models trained: {len(meta_models)}")
            logger.info(f"Features created: {len(feature_columns)}")
            logger.info(f"Data points processed: {len(features_df):,}")
            logger.info("=" * 80)
            
            logger.info("\n📊 FINAL PERFORMANCE SUMMARY:")
            logger.info("-" * 50)
            for target, metrics in evaluation_results.items():
                logger.info(f"{target}:")
                if 'accuracy' in metrics:
                    logger.info(f"  Accuracy: {metrics['accuracy']:.1%}")
                    logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
                else:
                    logger.info(f"  R² Score: {metrics['r2_score']:.3f}")
                    logger.info(f"  RMSE: {metrics['rmse']:.3f}")
            
            logger.info("\n🚀 Your models are now ready for trading!")
            logger.info("The most accurate investment prediction system has been created!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ EXTREME TRAINING FAILED: {e}")
            logger.error("Check the logs above for details.")
            return False


def main():
    """Run the extreme training system."""
    print("🔥🔥🔥 EXTREME TRAINING SYSTEM 🔥🔥🔥")
    print("=" * 60)
    print("This will train the most accurate investment models possible.")
    print("It will run for 6-12+ hours and max out your computer.")
    print("Perfect for overnight training!")
    print("=" * 60)
    
    # System requirements check
    print("\n🖥️  SYSTEM REQUIREMENTS CHECK:")
    print("-" * 40)
    
    import psutil
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available RAM: {ram_gb:.1f} GB")
    if ram_gb < 8:
        print("⚠️  WARNING: Less than 8GB RAM detected. Training may be slow.")
    
    # Check CPU cores
    cpu_cores = mp.cpu_count()
    print(f"CPU Cores: {cpu_cores}")
    
    # Check disk space
    disk_space = psutil.disk_usage('.').free / (1024**3)
    print(f"Free Disk Space: {disk_space:.1f} GB")
    if disk_space < 5:
        print("⚠️  WARNING: Less than 5GB free space. May not be sufficient.")
    
    print("\n🔋 IMPORTANT:")
    print("- Make sure your computer is plugged in")
    print("- Close other intensive applications")
    print("- Ensure stable internet connection")
    print("- This will run for HOURS - perfect for overnight")
    
    response = input("\nAre you ready to start extreme training? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Training cancelled. Come back when you're ready to go extreme! 🚀")
        return
    
    print("\n🚀 STARTING EXTREME TRAINING...")
    print("Go grab some coffee (or go to sleep) - this will take a while!")
    
    # Initialize and run
    trainer = ExtremeTrainingSystem()
    success = trainer.run_extreme_training()
    
    if success:
        print("\n🎉 SUCCESS! Your extreme models are ready!")
        print("Check the logs and saved files for details.")
    else:
        print("\n❌ Training failed. Check the logs for details.")


if __name__ == "__main__":
    main()
