"""
Deep Learning Module - Stock Price Prediction using LSTM
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import os
import pickle
from datetime import datetime, timedelta
from typing import Tuple, Optional
from utils.data_fetcher import DataFetcher
from utils.logger import setup_logger, log_data_processing, log_model_action

class DeepLearningModule:
    """Deep Learning module for stock price prediction using LSTM."""
    
    def __init__(self):
        """Initialize the DL module."""
        self.logger = setup_logger("DLModule")
        self.data_fetcher = DataFetcher()
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 60  # Number of days to look back
        
    def run(self):
        """Main execution method for the DL module."""
        try:
            # Get user inputs
            symbol, start_date, end_date, prediction_days = self._get_user_inputs()
            
            # Fetch and preprocess data
            self.logger.info("Starting data fetching and preprocessing")
            data = self._fetch_and_preprocess_data(symbol, start_date, end_date)
            
            # Train or load model
            self.logger.info("Preparing model for prediction")
            self._prepare_model(data)
            
            # Make predictions
            self.logger.info("Making predictions")
            predictions = self._make_predictions(data, prediction_days)
            
            # Display results
            self._display_results(data, predictions, symbol)
            
        except Exception as e:
            self.logger.error(f"Error in DL module: {e}")
            print(f"Error: {e}")
    
    def _get_user_inputs(self) -> Tuple[str, str, str, int]:
        """Get user inputs for stock prediction."""
        print("\n" + "-"*40)
        print("STOCK PRICE PREDICTION INPUTS")
        print("-"*40)
        
        # Get stock symbol
        while True:
            symbol = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                break
            print("Please enter a valid ticker symbol.")
        
        # Get date range
        while True:
            try:
                start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 2 years ago: ").strip()
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
                else:
                    datetime.strptime(start_date, "%Y-%m-%d")
                break
            except ValueError:
                print("Please enter a valid date in YYYY-MM-DD format.")
        
        while True:
            try:
                end_date = input("Enter end date (YYYY-MM-DD) or press Enter for today: ").strip()
                if not end_date:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    datetime.strptime(end_date, "%Y-%m-%d")
                break
            except ValueError:
                print("Please enter a valid date in YYYY-MM-DD format.")
        
        # Get prediction horizon
        while True:
            try:
                prediction_days = int(input("Enter number of days to predict (1-30): "))
                if 1 <= prediction_days <= 30:
                    break
                print("Please enter a number between 1 and 30.")
            except ValueError:
                print("Please enter a valid number.")
        
        self.logger.info(f"User inputs - Symbol: {symbol}, Start: {start_date}, End: {end_date}, Prediction days: {prediction_days}")
        return symbol, start_date, end_date, prediction_days
    
    def _fetch_and_preprocess_data(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Fetch and preprocess stock data with extra features."""
        # Fetch data
        data = self.data_fetcher.fetch_stock_data(symbol, start_date, end_date)
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        log_data_processing(self.logger, "Data Fetch", f"Fetched {len(data)} records for {symbol}")
        
        if data.empty:
            raise Exception(f"No data available for {symbol}")
        
        # Check for enough data
        if len(data) < self.sequence_length + 10:
            raise Exception(f"Insufficient data. Need at least {self.sequence_length + 10} days, got {len(data)}")
        
        # Extra features
        data['ma20'] = data['close'].rolling(20).mean()
        data['ma50'] = data['close'].rolling(50).mean()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['rsi'] = 100 - (100 / (1 + (data['returns'].rolling(14).mean() / data['returns'].rolling(14).std())))
        
        # Drop NaNs from rolling calculations
        data = data.dropna()
        
        # Scale all features
        features = ['close','ma20','ma50','returns','volatility','rsi']
        scaled_data = self.scaler.fit_transform(data[features])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict close price
        
        X = np.array(X)
        y = np.array(y)
        
        log_data_processing(self.logger, "Sequence Creation", f"Created {len(X)} sequences of length {self.sequence_length}")
        
        return {
            'scaled_data': scaled_data,
            'original_data': data['close'].values,
            'X': X,
            'y': y,
            'symbol': symbol,
            'df': data
        }
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def _prepare_model(self, data: dict):
        """Prepare and train LSTM model with multiple features."""
        model_path = f"models/lstm_model_{data.get('symbol', 'default')}.h5"
        
        # Try loading existing model
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.logger.info(f"Loaded existing model from {model_path}")
                metrics_path = model_path.replace('.h5', '_metrics.pkl')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'rb') as f:
                        self.metrics = pickle.load(f)
                else:
                    self.metrics = {}
                return
            except Exception as e:
                self.logger.warning(f"Could not load existing model: {e}")
        
        # Build new LSTM model
        log_model_action(self.logger, "Model Creation", "Building LSTM model architecture")
        
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, data['X'].shape[2])),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Split data
        X = data['X']
        y = data['y']
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        ]
        
        # Train model
        log_model_action(self.logger, "Model Training", f"Training on {len(X_train)} samples")
        self.model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        

        # Classification metrics based on actual next price, not last input
        threshold = 0.001  # 0.1% threshold
        y_train_class = (y_train[1:] > y_train[:-1] + threshold).astype(int)
        y_test_class = (y_test[1:] > y_test[:-1]).astype(int)
        train_pred_class = (train_pred.flatten()[1:] > y_train[:-1]).astype(int)
        test_pred_class = (test_pred.flatten()[1:] > y_test[:-1]).astype(int)
        
        # Metrics
        train_accuracy = accuracy_score(y_train_class, train_pred_class)
        test_accuracy = accuracy_score(y_test_class, test_pred_class)
        train_f1 = f1_score(y_train_class, train_pred_class, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_class, test_pred_class, average='weighted', zero_division=0)
        train_precision = precision_score(y_train_class, train_pred_class, average='weighted', zero_division=0)
        test_precision = precision_score(y_test_class, test_pred_class, average='weighted', zero_division=0)
        train_recall = recall_score(y_train_class, train_pred_class, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_class, test_pred_class, average='weighted', zero_division=0)
        
        self.metrics = {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
            'train_f1': train_f1, 'test_f1': test_f1,
            'train_precision': train_precision, 'test_precision': test_precision,
            'train_recall': train_recall, 'test_recall': test_recall,
            'y_test_class': y_test_class, 'test_pred_class': test_pred_class
        }
        
        log_model_action(self.logger, "Model Evaluation", f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        
        # Save model and metrics
        os.makedirs("models", exist_ok=True)
        self.model.save(model_path)
        metrics_path = model_path.replace('.h5', '_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def _make_predictions(self, data: dict, prediction_days: int) -> np.ndarray:
        """Make future predictions using the trained LSTM model (multi-feature aware)."""
        log_model_action(self.logger, "Prediction", f"Making {prediction_days} day predictions")
        
        # Get last sequence (use all features)
        last_sequence = data['X'][-1].reshape(1, self.sequence_length, data['X'].shape[2])
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(prediction_days):
            # Predict next value (only close price is predicted)
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred_scaled[0, 0])
            
            # Create next input sequence: shift and append new predicted close
            # Keep other features same as last row (or you could update with naive approach)
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = next_pred_scaled[0, 0]  # update only 'close'
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        # Inverse transform only the 'close' column
        close_index = 0  # 'close' is first column in scaled_data
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        dummy = np.zeros((len(predictions_scaled), data['scaled_data'].shape[1]))
        dummy[:, close_index] = predictions_scaled[:, 0]
        predictions_inv = self.scaler.inverse_transform(dummy)[:, close_index]
        
        return predictions_inv

    
    def _display_results(self, data: dict, predictions: np.ndarray, symbol: str):
        """Display prediction results."""
        print("\n" + "="*60)
        print("                    PREDICTION RESULTS")
        print("="*60)
        
        # Get recent actual data for comparison based on actual trading dates
        recent_df = data['df'].tail(10)
        recent_dates = recent_df.index.to_pydatetime()
        recent_data = recent_df['close'].values
        
        # Create prediction dates using business days starting after the last trading date
        last_date = data['df'].index[-1]
        pred_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=len(predictions)).to_pydatetime()
        
        # Display recent actual vs predicted
        print(f"\nRecent Actual Prices for {symbol}:")
        print("-" * 50)
        print("Date       | Close Price")
        print("-" * 50)
        
        for i, (date, price) in enumerate(zip(recent_dates, recent_data)):
            print(f"{date.strftime('%Y-%m-%d')} | ${price:.2f}")
        
        print(f"\nPredicted Prices for {symbol}:")
        print("-" * 50)
        print("Date       | Predicted Price | Confidence")
        print("-" * 50)
        
        for i, (date, pred) in enumerate(zip(pred_dates, predictions)):
            # Simple confidence calculation based on recent volatility
            recent_volatility = np.std(recent_data[-5:]) if len(recent_data) >= 5 else 0
            confidence = max(0.5, 1 - (recent_volatility / pred))
            print(f"{date.strftime('%Y-%m-%d')} | ${pred:.2f}        | {confidence:.1%}")
        
        # Calculate and display summary statistics
        if len(recent_data) > 0:
            last_actual = recent_data[-1]
            first_pred = predictions[0]
            change = ((first_pred - last_actual) / last_actual) * 100
            
        print(f"\nSummary Statistics:")
        print("-" * 30)
        print(f"Last Actual Price: ${last_actual:.2f}")
        print(f"First Prediction:  ${first_pred:.2f}")
        print(f"Predicted Change:  {change:+.2f}%")
        print(f"Prediction Range:  ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        # Display model performance metrics
        if hasattr(self, 'metrics') and self.metrics:
            print(f"\nModel Performance Metrics:")
            print("-" * 40)
            print(f"Regression Metrics:")
            print(f"  Train RMSE: {self.metrics['train_rmse']:.4f}")
            print(f"  Test RMSE:  {self.metrics['test_rmse']:.4f}")
            print(f"\nClassification Metrics (Price Direction):")
            print(f"  Test Accuracy:  {self.metrics['test_accuracy']:.4f}")
            print(f"  Test F1 Score:  {self.metrics['test_f1']:.4f}")
            print(f"  Test Precision: {self.metrics['test_precision']:.4f}")
            print(f"  Test Recall:    {self.metrics['test_recall']:.4f}")
            
            # Display confusion matrix
            if len(self.metrics['y_test_class']) > 0 and len(self.metrics['test_pred_class']) > 0:
                # Specify labels to ensure 2x2 matrix even if one class is absent
                cm = confusion_matrix(self.metrics['y_test_class'], self.metrics['test_pred_class'], labels=[0, 1])
                print(f"\nConfusion Matrix:")
                print("-" * 20)
                print("                 Predicted")
                print("               Down    Up")
                print(f"Actual Down    {cm[0,0]:4d}   {cm[0,1]:4d}")
                print(f"        Up     {cm[1,0]:4d}   {cm[1,1]:4d}")
                print(f"\nLegend: 0=Down, 1=Up")
        
        print("\n" + "="*60)
        print("Note: Predictions are for educational purposes only.")
        print("Past performance does not guarantee future results.")
        print("="*60)

