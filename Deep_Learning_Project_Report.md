## Deep Learning Project Report: Stock Price Prediction Using RNN/LSTM

### Project Prompt
Develop a robust, modular, and scalable deep learning system to predict target stock prices based on historical stock data. The framework utilizes Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models applied to time-series financial data fetched via APIs such as Yahoo Finance and Alpha Vantage.

### Core Features
- **Input**: User-specified stock ticker and prediction horizon
- **Data Preprocessing**: Normalization and handling missing data
- **Sequence Generation**: Sliding windows for LSTM input
- **Model Training**: Sequential deep learning model with stacked LSTM layers
- **Prediction**: Next-day or custom prediction horizons
- **Output**: Interactive visualization of historical vs predicted prices; exportable results
- **Feedback**: Loading states, error handling, prediction notifications
- **Confusion Matrix & Metrics**: Compare actual vs predicted; accuracy, precision, recall, F1; also MSE, RMSE, MAE

### Architecture
1. **Data Layer**
   - API integration for fetching OHLCV (Open, High, Low, Close, Volume) data (e.g., Yahoo Finance, Alpha Vantage).

2. **Preprocessing Module**
   - Data cleaning and imputation for missing values
   - Normalization using Min-Max scaler
   - Sliding window sequence generation for supervised learning (X sequences, y targets)

3. **Model Layer**
   - **Input**: Sequential historical data tensors shaped as (window_length, features)
   - **LSTM Layers**:
     - First LSTM: 64 units, return_sequences=True
     - Second LSTM: 32 units
   - **Dropout Layer**: 20% for regularization
   - **Dense Output Layer**: Linear activation to output predicted price
   - **Training Setup**:
     - Optimizer: Adam
     - Loss: Mean Squared Error (MSE)
     - Configurable epochs and batch size
   - **Evaluation Metrics**:
     - MSE, RMSE, MAE
     - Optional classification-style metrics (accuracy, precision, recall, F1) when binarizing direction (up/down)
   - **Visualization**: Predicted vs actual results using interactive plots

### Deep Learning Capabilities
- **Temporal Dependencies**: Captures short- and long-term patterns in volatile stock prices
- **LSTM Advantages**: Mitigates vanishing gradients common in vanilla RNNs
- **Multi-step Forecasting**: Extend output layer to support multi-horizon predictions
- **Regularization**: Dropout reduces overfitting and improves generalization
- **Performance Analysis**: Multiple metrics and plots to assess fit and generalization
- **Extensibility**: Can incorporate technical indicators and sentiment features

### Additional Features
- **Input Validation**: Ticker symbol checks and horizon constraints
- **Real-time Feedback**: Loading/progress indicators during data fetching and prediction
- **Export**: Save predictions and plots (CSV, PNG) for academic submission
- **Modularity**: Clear separation of preprocessing, modeling, training, and evaluation
- **Documentation**: RNN/LSTM methodology described for academic understanding

### Development Steps Summary
1. Connect and fetch historical stock data (Yahoo Finance or Alpha Vantage)
2. Clean and preprocess data (handle missing values, Min-Max scaling)
3. Convert price sequences into sliding windows for sequential input
4. Build LSTM network with specified architecture
5. Compile using Adam optimizer and MSE loss
6. Train on training split; validate for hyperparameter tuning
7. Evaluate on test data using RMSE and MAE (and direction-based metrics if applicable)
8. Visualize actual vs predicted stock prices with interactive plots
9. Document methodology, model choices, and performance

### Academic Fit
Delivering this project as a modular, scalable, and user-friendly system demonstrates deep learning proficiency applied to financial time series. The approach aligns with third-year engineering academic standards through end-to-end methodology, clear modularization, reproducible experiments, and interpretable evaluation.
