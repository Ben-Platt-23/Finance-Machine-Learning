"""
Machine Learning Models for Investment Predictions

This module contains various ML models designed to predict stock price movements
and generate investment signals. It uses ensemble methods and feature engineering
to create robust predictions for daily investment decisions.

Model Types:
1. Price Direction Classifier - Predicts if price will go up/down
2. Price Movement Regressor - Predicts magnitude of price change
3. Volatility Predictor - Forecasts future volatility
4. Ensemble Model - Combines multiple models for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import joblib
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestmentMLModels:
    """
    Comprehensive ML model suite for investment decision making.
    
    This class manages multiple machine learning models that work together
    to provide investment recommendations. It handles feature engineering,
    model training, prediction, and ensemble methods.
    """
    
    def __init__(self):
        """Initialize the ML models suite."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'direction_classifier': {
                'type': 'classification',
                'target': 'price_direction',
                'description': 'Predicts if price will go up (1) or down (0) tomorrow'
            },
            'movement_regressor': {
                'type': 'regression', 
                'target': 'price_change_pct',
                'description': 'Predicts percentage price change for tomorrow'
            },
            'volatility_predictor': {
                'type': 'regression',
                'target': 'future_volatility',
                'description': 'Predicts volatility for next 5 days'
            }
        }
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        This function takes a dataframe with technical indicators and creates
        additional features that are useful for ML models, including:
        - Lagged features (previous day values)
        - Rolling statistics (moving averages, std dev)
        - Interaction features (combinations of indicators)
        - Time-based features (day of week, month)
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with ML-ready features
        """
        logger.info("Preparing features for ML models...")
        
        # Make a copy to avoid modifying original data
        features_df = df.copy()
        
        # 1. CREATE TARGET VARIABLES
        # These are what we want to predict
        
        # Price direction (up=1, down=0) - for classification
        features_df['price_direction'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
        
        # Price change percentage - for regression
        features_df['price_change_pct'] = (features_df['close'].shift(-1) / features_df['close'] - 1) * 100
        
        # Future volatility (5-day rolling std of returns) - for volatility prediction
        returns = features_df['close'].pct_change()
        features_df['future_volatility'] = returns.shift(-1).rolling(window=5).std() * np.sqrt(252) * 100
        
        # 2. CREATE LAGGED FEATURES
        # Previous day values often predict next day movements
        lag_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'atr']
        
        for col in lag_columns:
            if col in features_df.columns:
                # 1-day, 2-day, and 5-day lags
                features_df[f'{col}_lag1'] = features_df[col].shift(1)
                features_df[f'{col}_lag2'] = features_df[col].shift(2)
                features_df[f'{col}_lag5'] = features_df[col].shift(5)
        
        # 3. CREATE ROLLING STATISTICS FEATURES
        # These capture recent trends and patterns
        
        # Price momentum features
        features_df['price_momentum_5d'] = features_df['close'] / features_df['close'].shift(5) - 1
        features_df['price_momentum_10d'] = features_df['close'] / features_df['close'].shift(10) - 1
        features_df['price_momentum_20d'] = features_df['close'] / features_df['close'].shift(20) - 1
        
        # Volume momentum features
        if 'volume' in features_df.columns:
            features_df['volume_momentum_5d'] = features_df['volume'] / features_df['volume'].rolling(5).mean()
            features_df['volume_momentum_10d'] = features_df['volume'] / features_df['volume'].rolling(10).mean()
        
        # Volatility features
        returns = features_df['close'].pct_change()
        features_df['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
        features_df['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
        features_df['volatility_ratio'] = features_df['volatility_5d'] / features_df['volatility_20d']
        
        # 4. CREATE INTERACTION FEATURES
        # Combinations of indicators can be more predictive
        
        if all(col in features_df.columns for col in ['rsi', 'bb_position']):
            # RSI and Bollinger Band position interaction
            features_df['rsi_bb_interaction'] = features_df['rsi'] * features_df['bb_position']
        
        if all(col in features_df.columns for col in ['macd', 'rsi']):
            # MACD and RSI divergence
            features_df['macd_rsi_divergence'] = features_df['macd'] - (features_df['rsi'] - 50)
        
        # 5. CREATE TIME-BASED FEATURES
        # Market patterns often depend on time
        
        if features_df.index.dtype == 'datetime64[ns]':
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            
            # Market calendar effects
            features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
            features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
            features_df['is_month_end'] = (features_df.index.day > 25).astype(int)
        
        # 6. CREATE REGIME FEATURES
        # Different market conditions require different strategies
        
        # Trend regime (based on multiple timeframe analysis)
        if all(col in features_df.columns for col in ['sma_10', 'sma_50', 'sma_200']):
            features_df['bullish_regime'] = (
                (features_df['sma_10'] > features_df['sma_50']) & 
                (features_df['sma_50'] > features_df['sma_200'])
            ).astype(int)
            
            features_df['bearish_regime'] = (
                (features_df['sma_10'] < features_df['sma_50']) & 
                (features_df['sma_50'] < features_df['sma_200'])
            ).astype(int)
        
        # Volatility regime
        if 'volatility_20d' in features_df.columns:
            vol_percentile = features_df['volatility_20d'].rolling(window=252).rank(pct=True)
            features_df['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            features_df['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
        
        logger.info(f"Created {len(features_df.columns) - len(df.columns)} new features")
        
        return features_df
    
    def select_features(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Select the most relevant features for a specific target.
        
        This function identifies which features are most predictive
        for the given target variable, helping to reduce overfitting
        and improve model performance.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            List of selected feature names
        """
        # Define base feature categories
        base_features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'bb_position', 'atr',
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'stoch_k', 'stoch_d', 'williams_r', 'adx', 'obv'
        ]
        
        # Add derived features
        derived_features = [
            col for col in df.columns 
            if any(suffix in col for suffix in ['_lag1', '_lag2', '_momentum_', '_ratio', '_interaction'])
        ]
        
        # Add time features
        time_features = [
            col for col in df.columns
            if col in ['day_of_week', 'month', 'is_monday', 'is_friday', 'is_month_end']
        ]
        
        # Add regime features
        regime_features = [
            col for col in df.columns
            if 'regime' in col or col in ['bullish_regime', 'bearish_regime']
        ]
        
        # Combine all feature categories
        all_features = base_features + derived_features + time_features + regime_features
        
        # Keep only features that exist in the dataframe and have no missing values in recent data
        available_features = []
        for feature in all_features:
            if (feature in df.columns and 
                feature != target_col and
                df[feature].notna().sum() > len(df) * 0.8):  # At least 80% non-null
                available_features.append(feature)
        
        logger.info(f"Selected {len(available_features)} features for {target_col}")
        
        return available_features
    
    def train_direction_classifier(self, df: pd.DataFrame) -> Dict:
        """
        Train a model to predict price direction (up/down).
        
        This classification model predicts whether the stock price will
        go up or down tomorrow. It's useful for binary trading decisions.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training price direction classifier...")
        
        target_col = 'price_direction'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Select features
        feature_cols = self.select_features(df, target_col)
        
        # Prepare data
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        y = df[target_col]
        
        # Remove rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            raise ValueError("Not enough data for training (need at least 100 samples)")
        
        # Time series split (preserves temporal order)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create ensemble of models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        # Train and evaluate each model
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    cv_scores.append(accuracy_score(y_val, y_pred))
                
                avg_score = np.mean(cv_scores)
                model_scores[name] = avg_score
                
                # Train on full dataset
                model.fit(X, y)
                trained_models[name] = model
                
                logger.info(f"{name} - CV Accuracy: {avg_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model
        if model_scores:
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = trained_models[best_model_name]
            
            # Store model and scaler
            self.models['direction_classifier'] = best_model
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance['direction_classifier'] = importance_df
            
            # Performance metrics
            y_pred = best_model.predict(X)
            performance = {
                'model_type': best_model_name,
                'accuracy': accuracy_score(y, y_pred),
                'cv_accuracy': model_scores[best_model_name],
                'n_features': len(feature_cols),
                'n_samples': len(X)
            }
            
            self.model_performance['direction_classifier'] = performance
            
            logger.info(f"Best direction classifier: {best_model_name} (Accuracy: {performance['accuracy']:.4f})")
            
            return performance
        
        else:
            raise RuntimeError("Failed to train any direction classifier models")
    
    def train_movement_regressor(self, df: pd.DataFrame) -> Dict:
        """
        Train a model to predict price movement magnitude.
        
        This regression model predicts the percentage change in price,
        which is useful for position sizing and risk management.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training price movement regressor...")
        
        target_col = 'price_change_pct'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Select features
        feature_cols = self.select_features(df, target_col)
        
        # Prepare data
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        y = df[target_col]
        
        # Remove rows with missing target or extreme outliers
        valid_idx = y.notna() & (abs(y) < 20)  # Remove >20% daily moves (likely errors)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            raise ValueError("Not enough data for training")
        
        # Scale features for regression
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.scalers['movement_regressor'] = scaler
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create ensemble of regression models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        }
        
        # Train and evaluate each model
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    cv_scores.append(r2_score(y_val, y_pred))
                
                avg_score = np.mean(cv_scores)
                model_scores[name] = avg_score
                
                # Train on full dataset
                model.fit(X_scaled, y)
                trained_models[name] = model
                
                logger.info(f"{name} - CV R² Score: {avg_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model
        if model_scores:
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = trained_models[best_model_name]
            
            # Store model
            self.models['movement_regressor'] = best_model
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance['movement_regressor'] = importance_df
            
            # Performance metrics
            y_pred = best_model.predict(X_scaled)
            performance = {
                'model_type': best_model_name,
                'r2_score': r2_score(y, y_pred),
                'cv_r2_score': model_scores[best_model_name],
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'n_features': len(feature_cols),
                'n_samples': len(X)
            }
            
            self.model_performance['movement_regressor'] = performance
            
            logger.info(f"Best movement regressor: {best_model_name} (R²: {performance['r2_score']:.4f})")
            
            return performance
        
        else:
            raise RuntimeError("Failed to train any movement regressor models")
    
    def make_predictions(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Make predictions using trained models.
        
        This function uses all trained models to make predictions
        on the latest data and combines them into actionable signals.
        
        Args:
            df: DataFrame with features (latest data)
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if df.empty:
            return {}
        
        predictions = {}
        latest_data = df.iloc[-1:].copy()  # Get latest row
        
        try:
            # 1. DIRECTION PREDICTION
            if 'direction_classifier' in self.models:
                model = self.models['direction_classifier']
                feature_cols = self.select_features(df, 'price_direction')
                
                # Prepare features
                X = latest_data[feature_cols].fillna(method='ffill').fillna(0)
                
                # Make prediction
                direction_pred = model.predict(X)[0]
                direction_proba = model.predict_proba(X)[0]
                
                predictions['direction'] = int(direction_pred)  # 0=down, 1=up
                predictions['direction_confidence'] = max(direction_proba)
                predictions['direction_probability'] = direction_proba[1] if len(direction_proba) > 1 else 0.5
            
            # 2. MOVEMENT MAGNITUDE PREDICTION
            if 'movement_regressor' in self.models:
                model = self.models['movement_regressor']
                scaler = self.scalers.get('movement_regressor')
                feature_cols = self.select_features(df, 'price_change_pct')
                
                # Prepare features
                X = latest_data[feature_cols].fillna(method='ffill').fillna(0)
                
                if scaler:
                    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
                else:
                    X_scaled = X
                
                # Make prediction
                movement_pred = model.predict(X_scaled)[0]
                predictions['expected_return'] = movement_pred
                predictions['expected_return_abs'] = abs(movement_pred)
            
            # 3. ENSEMBLE PREDICTION
            # Combine direction and magnitude predictions
            if 'direction' in predictions and 'expected_return' in predictions:
                # Adjust expected return based on direction confidence
                direction_factor = 1 if predictions['direction'] == 1 else -1
                confidence_factor = predictions['direction_confidence']
                
                predictions['ensemble_signal'] = (
                    direction_factor * 
                    predictions['expected_return_abs'] * 
                    confidence_factor
                )
                
                # Generate recommendation
                signal_strength = abs(predictions['ensemble_signal'])
                
                if predictions['ensemble_signal'] > 0.5:
                    predictions['recommendation'] = 'BUY'
                    predictions['confidence'] = min(signal_strength / 2.0, 1.0)
                elif predictions['ensemble_signal'] < -0.5:
                    predictions['recommendation'] = 'SELL'
                    predictions['confidence'] = min(signal_strength / 2.0, 1.0)
                else:
                    predictions['recommendation'] = 'HOLD'
                    predictions['confidence'] = 1.0 - signal_strength
            
            # 4. ADD METADATA
            predictions['prediction_timestamp'] = datetime.now().isoformat()
            predictions['models_used'] = list(self.models.keys())
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return default predictions
            predictions = {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'error': str(e)
            }
        
        return predictions
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.model_performance = model_data.get('model_performance', {})
            
            logger.info(f"Models loaded from {filepath}")
            logger.info(f"Available models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")


def main():
    """Example usage of InvestmentMLModels class."""
    print("Investment ML Models Module")
    print("=" * 40)
    print("This module provides machine learning models for:")
    print("- Price direction prediction (classification)")
    print("- Price movement magnitude (regression)")  
    print("- Ensemble predictions combining multiple models")
    print("- Feature engineering and selection")
    print("- Model performance tracking")
    print("\nModels use technical indicators and market data")
    print("to generate daily investment recommendations.")


if __name__ == "__main__":
    main()
