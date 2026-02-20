# Stockmarketdetection
A detection and analysis based on stock changes respect to time and its environment.
📈 Stock Market Prediction – Complete ML Pipeline

A complete end-to-end Machine Learning pipeline for stock market prediction using real stock data from Yahoo Finance.

This project includes:

Data downloading

Feature engineering (technical indicators)

Data preprocessing

Multiple ML models (Regression + Classification)

Model comparison

Trading strategy backtesting

Feature importance analysis

Model saving & reporting

🚀 Project Overview

This project builds a 5-day ahead stock price prediction system using:

📊 Technical indicators (RSI, MACD, Bollinger Bands, Rolling stats)

🤖 Machine Learning models (Random Forest, Gradient Boosting)

📉 Time-series aware data splitting

📈 Strategy backtesting

📦 Model exporting for reuse

The current implementation uses Apple (AAPL) stock data for the last 2 years, but you can change the ticker easily.

🛠️ Technologies & Libraries Used

Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

yFinance

Joblib

(Optional – for future deep learning extension):

TensorFlow / Keras (LSTM)

📂 Project Structure
.
├── stock_prediction.py
├── README.md
├── stock_data_visualization.png
├── feature_importance.png
├── price_predictions.png
├── confusion_matrix.png
├── trading_strategy.png
├── best_stock_model.pkl
├── direction_classifier.pkl
├── best_scaler.pkl
├── feature_list.txt
├── stock_prediction_report.txt
⚙️ Features Implemented
1️⃣ Data Download

Fetches real-time stock data from Yahoo Finance.

Supports any ticker (AAPL, TSLA, GOOGL, etc.)

2️⃣ Feature Engineering

Creates advanced technical indicators:

Lag Features

Rolling Means & Volatility

RSI (Relative Strength Index)

MACD

Bollinger Bands

Price Rate of Change

Volume Rate of Change

Temporal Features (Day, Month, Quarter)

Price Ratios

Total: 40+ engineered features

3️⃣ Target Variables

The pipeline creates three prediction targets:

📌 Future price (Regression)

📌 Future return (%)

📌 Binary direction (Up/Down classification)

Default setup:

Predicts stock price 5 days ahead

4️⃣ Time-Series Split

Proper chronological splitting:

80% Training

10% Validation

10% Testing

No shuffling (Time-series safe)

5️⃣ Feature Scaling Comparison

The pipeline compares:

StandardScaler

MinMaxScaler

RobustScaler

Automatically selects the best performing scaler + model.

6️⃣ Regression Models

Random Forest Regressor

Gradient Boosting Regressor

Evaluation metrics:

RMSE

MAE

R² Score

7️⃣ Classification Model (Directional Prediction)

Random Forest Classifier

Accuracy

Classification Report

Confusion Matrix

Predicts whether the stock will go Up or Down.

8️⃣ Trading Strategy Backtest

Implements a simple strategy:

Buy when predicted price > current price.

Outputs:

Strategy returns

Buy & Hold returns

Outperformance

9️⃣ Model Export

The project saves:

best_stock_model.pkl

direction_classifier.pkl

best_scaler.pkl

stock_prediction_report.txt

feature_list.txt

Ready for deployment or further research.

📊 Output Files Generated
File	Description
stock_data_visualization.png	Initial stock analysis plots
feature_importance.png	Important features chart
price_predictions.png	Predicted vs actual prices
confusion_matrix.png	Direction prediction results
trading_strategy.png	Backtesting performance
best_stock_model.pkl	Trained regression model
direction_classifier.pkl	Up/Down classifier
stock_prediction_report.txt	Performance summary
▶️ How to Run
1️⃣ Clone the Repository
git clone https://github.com/yourusername/stock-market-prediction.git
cd stock-market-prediction
2️⃣ Install Required Libraries
pip install numpy pandas matplotlib seaborn scikit-learn yfinance joblib
3️⃣ Run the Script
python stock_prediction.py

Or run in:

Google Colab

Jupyter Notebook

VS Code

🔄 How to Change Stock

Inside the script:

df_stock = download_stock_data('AAPL', '2y')

Change 'AAPL' to:

'TSLA'

'GOOGL'

'MSFT'

'INFY'

Any Yahoo Finance ticker

📈 Sample Pipeline Flow
Download Data
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Model Training
    ↓
Evaluation
    ↓
Backtesting
    ↓
Save Best Model
