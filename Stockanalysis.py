"""
STOCK MARKET PREDICTION - Complete Preprocessing & ML Pipeline
For Google Colab / Jupyter Notebooks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Deep Learning (optional - uncomment if needed)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

print("="*60)
print("STOCK MARKET PREDICTION PIPELINE")
print("="*60)

# ======================
# 1. DOWNLOAD REAL STOCK DATA
# ======================
print("\n" + "-"*40)
print("STAGE 1: DOWNLOADING STOCK DATA")
print("-"*40)

def download_stock_data(ticker='AAPL', period='2y'):
    """
    Download stock data from Yahoo Finance
    ticker: Stock symbol (AAPL, GOOGL, TSLA, etc.)
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
    """
    print(f"Downloading {ticker} data for {period}...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Reset index to make Date a column
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    print(f"Downloaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df

# Download Apple stock data
df_stock = download_stock_data('AAPL', '2y')

print("\nFirst 5 rows:")
print(df_stock.head())

print("\nLast 5 rows:")
print(df_stock.tail())

print("\nDataset Info:")
print(df_stock.info())

print("\nBasic Statistics:")
print(df_stock.describe())

# ======================
# 2. VISUALIZE STOCK DATA
# ======================
print("\n" + "-"*40)
print("STAGE 2: VISUALIZING STOCK DATA")
print("-"*40)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price chart
axes[0,0].plot(df_stock.index, df_stock['Close'], label='Close Price', color='blue')
axes[0,0].set_title(f'AAPL Stock Price (Last {len(df_stock)} days)')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Price ($)')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# Volume chart
axes[0,1].bar(df_stock.index, df_stock['Volume'], color='green', alpha=0.6)
axes[0,1].set_title('Trading Volume')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Volume')
axes[0,1].grid(True, alpha=0.3)

# Returns distribution
df_stock['Returns'] = df_stock['Close'].pct_change() * 100
axes[1,0].hist(df_stock['Returns'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1,0].set_title('Daily Returns Distribution')
axes[1,0].set_xlabel('Returns (%)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

# Moving averages
axes[1,1].plot(df_stock.index, df_stock['Close'], label='Close', alpha=0.5)
axes[1,1].plot(df_stock.index, df_stock['Close'].rolling(20).mean(), label='20-day MA', color='orange')
axes[1,1].plot(df_stock.index, df_stock['Close'].rolling(50).mean(), label='50-day MA', color='red')
axes[1,1].set_title('Moving Averages')
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Price ($)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_data_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved visualization as 'stock_data_visualization.png'")

# ======================
# 3. FEATURE ENGINEERING FOR STOCK DATA
# ======================
print("\n" + "-"*40)
print("STAGE 3: FEATURE ENGINEERING")
print("-"*40)

class StockFeatureEngineer:
    """Create technical indicators and features for stock prediction"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        
    def add_lagged_features(self, columns=['Close', 'Volume'], lags=[1,2,3,5,10]):
        """Add lagged values of price and volume"""
        print("\nAdding lagged features...")
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                self.feature_names.append(f'{col}_lag_{lag}')
        return self.df
    
    def add_rolling_features(self, windows=[5,10,20,50]):
        """Add rolling statistics"""
        print("Adding rolling features...")
        
        # Rolling means
        for window in windows:
            self.df[f'Close_rolling_mean_{window}'] = self.df['Close'].rolling(window).mean()
            self.df[f'Volume_rolling_mean_{window}'] = self.df['Volume'].rolling(window).mean()
            self.feature_names.extend([f'Close_rolling_mean_{window}', f'Volume_rolling_mean_{window}'])
        
        # Rolling standard deviation (volatility)
        for window in [5,10,20]:
            self.df[f'Close_rolling_std_{window}'] = self.df['Close'].rolling(window).std()
            self.feature_names.append(f'Close_rolling_std_{window}')
        
        return self.df
    
    def add_technical_indicators(self):
        """Add common technical indicators"""
        print("Adding technical indicators...")
        
        # RSI (Relative Strength Index)
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        self.df['RSI'] = calculate_rsi(self.df['Close'], 14)
        self.feature_names.append('RSI')
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_histogram'] = self.df['MACD'] - self.df['MACD_signal']
        self.feature_names.extend(['MACD', 'MACD_signal', 'MACD_histogram'])
        
        # Bollinger Bands
        rolling_mean = self.df['Close'].rolling(window=20).mean()
        rolling_std = self.df['Close'].rolling(window=20).std()
        self.df['Bollinger_upper'] = rolling_mean + (rolling_std * 2)
        self.df['Bollinger_lower'] = rolling_mean - (rolling_std * 2)
        self.df['Bollinger_width'] = (self.df['Bollinger_upper'] - self.df['Bollinger_lower']) / rolling_mean
        self.feature_names.extend(['Bollinger_upper', 'Bollinger_lower', 'Bollinger_width'])
        
        # Price rate of change
        for period in [1,5,10]:
            self.df[f'Price_ROC_{period}'] = self.df['Close'].pct_change(period) * 100
            self.feature_names.append(f'Price_ROC_{period}')
        
        # Volume rate of change
        self.df['Volume_ROC'] = self.df['Volume'].pct_change() * 100
        self.feature_names.append('Volume_ROC')
        
        return self.df
    
    def add_temporal_features(self):
        """Add time-based features"""
        print("Adding temporal features...")
        
        self.df['Day_of_week'] = self.df.index.dayofweek
        self.df['Month'] = self.df.index.month
        self.df['Quarter'] = self.df.index.quarter
        self.df['Year'] = self.df.index.year
        self.df['Day_of_month'] = self.df.index.day
        
        self.feature_names.extend(['Day_of_week', 'Month', 'Quarter', 'Year', 'Day_of_month'])
        
        return self.df
    
    def add_price_features(self):
        """Add price-based features"""
        print("Adding price-based features...")
        
        # Price ratios
        self.df['High_Low_ratio'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['Open_Close_ratio'] = (self.df['Open'] - self.df['Close']) / self.df['Open']
        
        self.feature_names.extend(['High_Low_ratio', 'Open_Close_ratio'])
        
        return self.df
    
    def create_target(self, prediction_days=1):
        """Create target variable (future price)"""
        print(f"\nCreating target: Predict price {prediction_days} days ahead")
        
        # Target: Future price
        self.df['Target'] = self.df['Close'].shift(-prediction_days)
        
        # Alternative: Future return (%)
        self.df['Target_return'] = (self.df['Close'].shift(-prediction_days) - self.df['Close']) / self.df['Close'] * 100
        
        # Alternative: Binary direction (1 if price up, 0 if down)
        self.df['Target_direction'] = (self.df['Close'].shift(-prediction_days) > self.df['Close']).astype(int)
        
        return self.df
    
    def engineer_all_features(self):
        """Run all feature engineering steps"""
        print("\n" + "="*40)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*40)
        
        self.add_lagged_features()
        self.add_rolling_features()
        self.add_technical_indicators()
        self.add_temporal_features()
        self.add_price_features()
        self.create_target(prediction_days=5)  # Predict 5 days ahead
        
        print(f"\n✅ Created {len(self.feature_names)} features")
        print(f"Dataset shape: {self.df.shape}")
        
        return self.df

# Apply feature engineering
engineer = StockFeatureEngineer(df_stock)
df_features = engineer.engineer_all_features()

# Drop rows with NaN values (from lags and rolling calculations)
df_clean = df_features.dropna()

print(f"\nAfter dropping NaN values: {df_clean.shape}")

# ======================
# 4. DATA PREPROCESSING
# ======================
print("\n" + "-"*40)
print("STAGE 4: DATA PREPROCESSING")
print("-"*40)

# Separate features and targets
feature_cols = engineer.feature_names
target_col = 'Target'  # Predicting future price
target_direction = 'Target_direction'  # For classification

X = df_clean[feature_cols]
y_regression = df_clean[target_col]  # For regression (price prediction)
y_classification = df_clean[target_direction]  # For classification (up/down)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y_regression.shape}")

# ======================
# 5. TRAIN-TEST SPLIT (TIME SERIES APPROPRIATE)
# ======================
print("\n" + "-"*40)
print("STAGE 5: TIME SERIES SPLIT")
print("-"*40)

# For time series, we DON'T shuffle - we split chronologically
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]

y_train = y_regression.iloc[:train_size]
y_val = y_regression.iloc[train_size:train_size+val_size]
y_test = y_regression.iloc[train_size+val_size:]

# For classification
y_train_class = y_classification.iloc[:train_size]
y_val_class = y_classification.iloc[train_size:train_size+val_size]
y_test_class = y_classification.iloc[train_size+val_size:]

print(f"Training dates: {df_clean.index[train_size-1].date()} back to {df_clean.index[0].date()}")
print(f"Validation dates: {df_clean.index[train_size+val_size-1].date()} to {df_clean.index[train_size].date()}")
print(f"Test dates: {df_clean.index[-1].date()} to {df_clean.index[train_size+val_size].date()}")

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# ======================
# 6. FEATURE SCALING
# ======================
print("\n" + "-"*40)
print("STAGE 6: FEATURE SCALING")
print("-"*40)

# Create different scalers for comparison
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

scaled_data = {}

for name, scaler in scalers.items():
    print(f"\nApplying {name}...")
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    scaled_data[name] = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'scaler': scaler
    }

# ======================
# 7. MODEL TRAINING - REGRESSION (Price Prediction)
# ======================
print("\n" + "-"*40)
print("STAGE 7: MODEL TRAINING - REGRESSION")
print("-"*40)

def train_regression_models(X_train, y_train, X_val, y_val, X_test, y_test, scaler_name):
    """Train multiple regression models and compare"""
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} with {scaler_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'model': model,
            'predictions': y_pred_test
        }
        
        print(f"   Train RMSE: ${train_rmse:.2f}")
        print(f"   Val RMSE: ${val_rmse:.2f}")
        print(f"   Test RMSE: ${test_rmse:.2f}")
        print(f"   Test R²: {test_r2:.4f}")
    
    return results

# Train models with different scalers
all_results = {}

for scaler_name, data in scaled_data.items():
    print(f"\n" + "="*40)
    print(f"RESULTS WITH {scaler_name}")
    print("="*40)
    
    results = train_regression_models(
        data['X_train'], y_train,
        data['X_val'], y_val,
        data['X_test'], y_test,
        scaler_name
    )
    
    all_results[scaler_name] = results

# ======================
# 8. MODEL EVALUATION AND COMPARISON
# ======================
print("\n" + "-"*40)
print("STAGE 8: MODEL COMPARISON")
print("-"*40)

# Create comparison dataframe
comparison_data = []
for scaler_name, results in all_results.items():
    for model_name, metrics in results.items():
        comparison_data.append({
            'Scaler': scaler_name,
            'Model': model_name,
            'Test RMSE': metrics['test_rmse'],
            'Test MAE': metrics['test_mae'],
            'Test R²': metrics['test_r2']
        })

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Comparison:")
print(comparison_df.sort_values('Test RMSE'))

# Find best model
best_row = comparison_df.loc[comparison_df['Test RMSE'].idxmin()]
print(f"\n🏆 Best Model: {best_row['Model']} with {best_row['Scaler']}")
print(f"   Test RMSE: ${best_row['Test RMSE']:.2f}")
print(f"   Test R²: {best_row['Test R²']:.4f}")

# ======================
# 9. FEATURE IMPORTANCE
# ======================
print("\n" + "-"*40)
print("STAGE 9: FEATURE IMPORTANCE ANALYSIS")
print("-"*40)

# Get the best model
best_scaler = best_row['Scaler']
best_model_name = best_row['Model']
best_model = all_results[best_scaler][best_model_name]['model']

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 10 Most Important Features:")
    for i in range(min(10, len(feature_cols))):
        print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title(f'Feature Importance - {best_model_name}')
    plt.bar(range(min(20, len(feature_cols))), importances[indices[:20]])
    plt.xticks(range(min(20, len(feature_cols))), [feature_cols[i] for i in indices[:20]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved feature importance plot as 'feature_importance.png'")

# ======================
# 10. VISUALIZE PREDICTIONS
# ======================
print("\n" + "-"*40)
print("STAGE 10: VISUALIZING PREDICTIONS")
print("-"*40)

# Get predictions from best model
best_predictions = all_results[best_scaler][best_model_name]['predictions']

# Create prediction plot
plt.figure(figsize=(15, 6))

# Plot actual vs predicted
plt.plot(df_clean.index[train_size+val_size:], y_test.values, label='Actual Price', color='blue', linewidth=2)
plt.plot(df_clean.index[train_size+val_size:], best_predictions, label='Predicted Price', color='red', linewidth=2, linestyle='--')

plt.title(f'{best_model_name} - Stock Price Prediction (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved predictions plot as 'price_predictions.png'")

# ======================
# 11. CLASSIFICATION MODEL (Up/Down Prediction)
# ======================
print("\n" + "-"*40)
print("STAGE 11: DIRECTIONAL PREDICTION (Up/Down)")
print("-"*40)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use the same scaled data for classification
X_train_class = scaled_data[best_scaler]['X_train']
X_val_class = scaled_data[best_scaler]['X_val']
X_test_class = scaled_data[best_scaler]['X_test']

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_class, y_train_class)

# Predict
y_pred_class = clf.predict(X_test_class)

# Accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"\nDirectional Prediction Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class, target_names=['Down', 'Up']))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_class, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix - Direction Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved confusion matrix as 'confusion_matrix.png'")

# ======================
# 12. SIMPLE TRADING STRATEGY BACKTEST
# ======================
print("\n" + "-"*40)
print("STAGE 12: SIMPLE TRADING STRATEGY BACKTEST")
print("-"*40)

# Create signals based on predictions
signals = pd.DataFrame(index=df_clean.index[train_size+val_size:])
signals['Actual_Price'] = y_test.values
signals['Predicted_Price'] = best_predictions
signals['Actual_Return'] = signals['Actual_Price'].pct_change()
signals['Predicted_Return'] = signals['Predicted_Price'].pct_change()

# Trading strategy: Buy when predicted price > current price
signals['Signal'] = (signals['Predicted_Price'] > signals['Actual_Price'].shift(1)).astype(int)

# Calculate strategy returns
signals['Strategy_Return'] = signals['Signal'].shift(1) * signals['Actual_Return']
signals['Buy_Hold_Return'] = signals['Actual_Return']

# Cumulative returns
signals['Cumulative_Strategy'] = (1 + signals['Strategy_Return']).cumprod()
signals['Cumulative_Buy_Hold'] = (1 + signals['Buy_Hold_Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(signals.index, signals['Cumulative_Strategy'], label='Strategy Returns', color='green', linewidth=2)
plt.plot(signals.index, signals['Cumulative_Buy_Hold'], label='Buy & Hold Returns', color='blue', linewidth=2)
plt.title('Trading Strategy Backtest')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trading_strategy.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved trading strategy plot as 'trading_strategy.png'")

# Calculate metrics
total_strategy_return = (signals['Cumulative_Strategy'].iloc[-1] - 1) * 100
total_buy_hold_return = (signals['Cumulative_Buy_Hold'].iloc[-1] - 1) * 100

print(f"\nBacktest Results:")
print(f"Total Strategy Return: {total_strategy_return:.2f}%")
print(f"Total Buy & Hold Return: {total_buy_hold_return:.2f}%")
print(f"Strategy Outperformance: {total_strategy_return - total_buy_hold_return:.2f}%")

# ======================
# 13. SAVE MODELS AND RESULTS
# ======================
print("\n" + "-"*40)
print("STAGE 13: SAVING RESULTS")
print("-"*40)

import joblib

# Save the best model and scaler
joblib.dump(best_model, 'best_stock_model.pkl')
joblib.dump(scaled_data[best_scaler]['scaler'], 'best_scaler.pkl')
joblib.dump(clf, 'direction_classifier.pkl')

print("✅ Saved models:")
print("   - best_stock_model.pkl (price prediction)")
print("   - direction_classifier.pkl (up/down prediction)")
print("   - best_scaler.pkl (feature scaler)")

# Save feature list
with open('feature_list.txt', 'w') as f:
    for feature in feature_cols:
        f.write(f"{feature}\n")
print("   - feature_list.txt")

# Generate final report
report = f"""
STOCK MARKET PREDICTION - FINAL REPORT
========================================

Data Summary:
- Ticker: AAPL
- Date Range: {df_clean.index[0].date()} to {df_clean.index[-1].date()}
- Total days: {len(df_clean)}
- Features created: {len(feature_cols)}

Model Performance:
- Best Model: {best_model_name}
- Best Scaler: {best_scaler}
- Test RMSE: ${best_row['Test RMSE']:.2f}
- Test R²: {best_row['Test R²']:.4f}
- Direction Accuracy: {accuracy:.4f}

Trading Strategy Backtest:
- Strategy Return: {total_strategy_return:.2f}%
- Buy & Hold Return: {total_buy_hold_return:.2f}%
- Outperformance: {total_strategy_return - total_buy_hold_return:.2f}%

Top 5 Important Features:
"""

# Add top features
if hasattr(best_model, 'feature_importances_'):
    for i in range(min(5, len(feature_cols))):
        report += f"- {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}\n"

# Save report
with open('stock_prediction_report.txt', 'w') as f:
    f.write(report)

print("\n✅ Saved final report as 'stock_prediction_report.txt'")
print(report)

print("\n" + "="*60)
print("✅ STOCK MARKET PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nFiles saved:")
print("1. stock_data_visualization.png - Initial data plots")
print("2. feature_importance.png - Feature importance chart")
print("3. price_predictions.png - Actual vs predicted prices")
print("4. confusion_matrix.png - Direction prediction results")
print("5. trading_strategy.png - Backtest results")
print("6. best_stock_model.pkl - Trained price prediction model")
print("7. direction_classifier.pkl - Up/down classifier")
print("8. best_scaler.pkl - Feature scaler")
print("9. feature_list.txt - List of all features")
print("10. stock_prediction_report.txt - Complete summary report")
