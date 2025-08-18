#!/bin/bash

# AUTONOMOUS OVERNIGHT TRAINING SCRIPT
# ====================================
# This script will run completely autonomously while you sleep!
# It handles all setup, installation, and training without any user interaction.

echo "🔥🔥🔥 STARTING AUTONOMOUS OVERNIGHT TRAINING 🔥🔥🔥"
echo "=================================================="
echo "This will run for 6-12+ hours while you sleep!"
echo "No user interaction required - go to bed!"
echo "=================================================="

# Set up logging
LOGFILE="overnight_training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "$(date): Starting autonomous training setup..."

# Change to project directory
cd "$(dirname "$0")"

# Set Python version
echo "$(date): Setting Python 3.11.5..."
export PYENV_VERSION=3.11.5

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "$(date): Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "$(date): Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "$(date): Upgrading pip..."
pip install --upgrade pip --quiet

# Install packages one by one to handle any failures gracefully
echo "$(date): Installing core packages..."

# Core data science packages
pip install numpy==1.24.3 --quiet
pip install pandas==2.0.3 --quiet  # Using older compatible version
pip install scikit-learn==1.3.2 --quiet
pip install scipy==1.11.4 --quiet

# Financial data
pip install yfinance==0.2.28 --quiet

# Visualization
pip install matplotlib==3.8.2 --quiet
pip install seaborn==0.13.0 --quiet

# Technical analysis
pip install ta==0.10.2 --quiet

# Machine learning
pip install xgboost==2.0.2 --quiet
pip install lightgbm==4.1.0 --quiet

# Utilities
pip install requests==2.31.0 --quiet
pip install joblib==1.3.2 --quiet
pip install python-dotenv==1.0.0 --quiet
pip install psutil==5.9.6 --quiet

echo "$(date): Package installation complete!"

# Create a simplified training script that will definitely work
cat > autonomous_training.py << 'EOF'
"""
AUTONOMOUS EXTREME TRAINING SYSTEM
==================================
This version is designed to run completely autonomously overnight.
It will train the most accurate models possible without any user interaction.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
import lightgbm as lgb
import time
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
import joblib
import ta

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutonomousTrainer:
    def __init__(self):
        self.start_time = datetime.now()
        logger.info("🚀 AUTONOMOUS TRAINING SYSTEM INITIALIZED")
        
        # Top stocks for training
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL',
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK',
            'HD', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD',
            'BA', 'CAT', 'HON', 'MMM', 'GE',
            'XOM', 'CVX', 'COP',
            'SPY', 'QQQ', 'IWM'
        ]
        
    def download_data(self):
        logger.info("📡 DOWNLOADING MASSIVE DATASET...")
        all_data = {}
        
        for i, symbol in enumerate(self.symbols):
            try:
                logger.info(f"Downloading {symbol} ({i+1}/{len(self.symbols)})...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='5y', interval='1d')
                
                if not df.empty and len(df) >= 500:
                    all_data[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} days")
                else:
                    logger.warning(f"❌ {symbol}: Insufficient data")
                
                time.sleep(0.5)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        logger.info(f"📊 Downloaded data for {len(all_data)} symbols")
        return all_data
    
    def create_features(self, data_dict):
        logger.info("🧠 CREATING ADVANCED FEATURES...")
        
        all_features = []
        
        for symbol, df in data_dict.items():
            try:
                logger.info(f"Processing {symbol}...")
                
                # Ensure column names are lowercase
                df.columns = [col.lower() for col in df.columns]
                
                # Technical indicators using ta library
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                df['macd'] = ta.trend.macd(df['close'])
                df['macd_signal'] = ta.trend.macd_signal(df['close'])
                df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
                df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
                df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
                df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
                
                # Price-based features
                df['sma_10'] = df['close'].rolling(10).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                
                # Volatility features
                returns = df['close'].pct_change()
                df['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
                df['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
                
                # Volume features
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # Momentum features
                df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
                df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
                df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
                
                # Bollinger Band features
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                
                # Target variables
                df['price_up_1d'] = (df['close'].shift(-1) > df['close']).astype(int)
                df['return_1d'] = df['close'].pct_change(-1) * 100
                
                # Add symbol
                df['symbol'] = symbol
                
                # Clean data
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(method='ffill').fillna(method='bfill')
                df = df.dropna()
                
                if len(df) > 100:
                    all_features.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"📈 Created {len(combined_df)} samples with {len(combined_df.columns)} features")
            return combined_df
        else:
            raise Exception("No features created")
    
    def train_models(self, features_df):
        logger.info("🔥 TRAINING EXTREME MODELS...")
        
        # Feature columns (exclude targets and metadata)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['symbol', 'price_up_1d', 'return_1d']]
        
        X = features_df[feature_cols].fillna(0)
        
        # Train classification model (price direction)
        y_class = features_df['price_up_1d'].fillna(0)
        valid_mask = ~pd.isna(y_class)
        X_class = X[valid_mask]
        y_class = y_class[valid_mask]
        
        logger.info("Training classification models...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Model configurations with extensive hyperparameter search
        models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        best_models = {}
        
        for name, config in models.items():
            try:
                logger.info(f"Training {name}...")
                
                search = RandomizedSearchCV(
                    estimator=config['model'],
                    param_distributions=config['params'],
                    n_iter=50,  # 50 random combinations
                    cv=tscv,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42
                )
                
                search.fit(X_class, y_class)
                
                best_models[name] = {
                    'model': search.best_estimator_,
                    'score': search.best_score_,
                    'params': search.best_params_
                }
                
                logger.info(f"✅ {name} - Best Score: {search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Train regression model (return prediction)
        y_reg = features_df['return_1d'].fillna(0)
        valid_mask = ~pd.isna(y_reg) & (abs(y_reg) < 20)  # Remove extreme outliers
        X_reg = X[valid_mask]
        y_reg = y_reg[valid_mask]
        
        logger.info("Training regression models...")
        
        reg_models = {
            'random_forest_reg': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 0.5]
                }
            },
            'xgboost_reg': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        for name, config in reg_models.items():
            try:
                logger.info(f"Training {name}...")
                
                search = RandomizedSearchCV(
                    estimator=config['model'],
                    param_distributions=config['params'],
                    n_iter=30,
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42
                )
                
                search.fit(X_reg, y_reg)
                
                best_models[name] = {
                    'model': search.best_estimator_,
                    'score': search.best_score_,
                    'params': search.best_params_
                }
                
                logger.info(f"✅ {name} - Best R² Score: {search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return best_models, feature_cols
    
    def save_models(self, models, feature_cols):
        logger.info("💾 SAVING MODELS...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create models directory
        os.makedirs('trained_models', exist_ok=True)
        
        # Save models
        models_file = f'trained_models/extreme_models_{timestamp}.joblib'
        joblib.dump({
            'models': models,
            'feature_columns': feature_cols,
            'timestamp': timestamp
        }, models_file)
        
        logger.info(f"✅ Models saved to: {models_file}")
        
        # Save summary
        summary_file = f'trained_models/training_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("AUTONOMOUS EXTREME TRAINING - RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed: {datetime.now()}\n")
            f.write(f"Training duration: {datetime.now() - self.start_time}\n")
            f.write(f"Models trained: {len(models)}\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for name, info in models.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Score: {info['score']:.4f}\n")
                f.write(f"  Params: {info['params']}\n")
        
        logger.info(f"✅ Summary saved to: {summary_file}")
        
        return models_file, summary_file
    
    def run_autonomous_training(self):
        try:
            logger.info("🔥🔥🔥 STARTING AUTONOMOUS TRAINING 🔥🔥🔥")
            
            # Phase 1: Download data
            data_dict = self.download_data()
            
            # Phase 2: Create features
            features_df = self.create_features(data_dict)
            
            # Phase 3: Train models
            models, feature_cols = self.train_models(features_df)
            
            # Phase 4: Save everything
            model_file, summary_file = self.save_models(models, feature_cols)
            
            # Final summary
            duration = datetime.now() - self.start_time
            logger.info("🎉🎉🎉 AUTONOMOUS TRAINING COMPLETE! 🎉🎉🎉")
            logger.info(f"Total time: {duration}")
            logger.info(f"Models trained: {len(models)}")
            logger.info(f"Models saved to: {model_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TRAINING FAILED: {e}")
            return False

if __name__ == "__main__":
    trainer = AutonomousTrainer()
    success = trainer.run_autonomous_training()
    
    if success:
        print("\n🎉 SUCCESS! Your models are ready!")
    else:
        print("\n❌ Training failed. Check logs.")
EOF

echo "$(date): Starting autonomous training..."

# Run the training in the background with nohup
nohup python autonomous_training.py > autonomous_training_output.log 2>&1 &

TRAINING_PID=$!

echo "$(date): Training started with PID: $TRAINING_PID"
echo "Training is now running in the background!"
echo "You can safely close this terminal and go to sleep."
echo ""
echo "To monitor progress:"
echo "  tail -f autonomous_training_output.log"
echo ""
echo "To check if it's still running:"
echo "  ps aux | grep $TRAINING_PID"
echo ""
echo "Log files will be created in:"
echo "  - autonomous_training.log (detailed training log)"
echo "  - autonomous_training_output.log (script output)"
echo "  - $LOGFILE (setup log)"
echo ""
echo "Models will be saved in: trained_models/"
echo ""
echo "🌙 GOOD NIGHT! Your models will be ready in the morning! 🌙"

# Keep the script running for a few seconds to show the message
sleep 5

echo "$(date): Setup complete. Training is running autonomously."
EOF
