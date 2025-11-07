# Financial Project Application

A comprehensive terminal-based financial application with three modules: Deep Learning (DL), Natural Language Processing (NLP), and Intelligent Systems II (IS-II).

## Features

### 1. Deep Learning Module (DL)
- **Stock Price Prediction** using LSTM neural networks
- Fetches historical data from Alpha Vantage API (with Yahoo Finance fallback)
- Preprocesses and normalizes data for optimal model performance
- Trains LSTM model with dropout layers for regularization
- Provides future price predictions with confidence intervals
- Displays comprehensive results including RMSE and MAE metrics

### 2. NLP Module
- **Financial News Sentiment Analysis** using BiLSTM+Attention
- Fetches latest financial news from Alpha Vantage API
- Preprocesses text data (cleaning, tokenization, lemmatization)
- Classifies sentiment as Bullish, Bearish, or Neutral
- Provides aggregate sentiment trends and confidence analysis
- Real-time logging of all processing steps

### 3. IS-II Module
- **Portfolio Optimization** using Particle Swarm Optimization (PSO)
- Supports multiple stock tickers for portfolio construction
- Configurable PSO parameters (particles, iterations, risk tolerance)
- Calculates optimal portfolio weights to maximize Sharpe ratio
- Provides comprehensive risk analysis and diversification metrics
- Real-time optimization progress logging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Combined-DL,NLP,IS2-V1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

## Usage

### Interactive Menu Mode
```bash
python main.py
```

### Command Line Mode
```bash
# Deep Learning Module
python main.py --module dl

# NLP Module
python main.py --module nlp

# IS-II Module
python main.py --module is2
```

## Project Structure

```
Combined-DL,NLP,IS2-V1/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── modules/               # Application modules
│   ├── __init__.py
│   ├── dl_module.py       # Deep Learning module
│   ├── nlp_module.py      # NLP module
│   └── is2_module.py      # IS-II module
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── logger.py          # Logging utilities
│   └── data_fetcher.py    # Data fetching with API fallbacks
├── models/                # Saved models (created at runtime)
├── data/                  # Data storage (created at runtime)
└── logs/                  # Log files (created at runtime)
```

## API Configuration

### Alpha Vantage API
1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Set the environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```
3. Or the application will use Yahoo Finance as fallback

### API Fallback Strategy
1. **Primary**: Alpha Vantage API (requires API key)
2. **Fallback**: Yahoo Finance (no API key required)
3. **News**: Alpha Vantage News API → Simulated news data

## Module Details

### Deep Learning Module
- **Input**: Stock ticker, date range, prediction horizon
- **Process**: Data fetching, preprocessing, LSTM training/prediction
- **Output**: Future price predictions with confidence intervals
- **Features**: Model persistence, comprehensive error handling

### NLP Module
- **Input**: Stock ticker symbol
- **Process**: News fetching, text preprocessing, sentiment classification
- **Output**: Sentiment analysis results with aggregate trends
- **Features**: BiLSTM+Attention architecture, confidence scoring

### IS-II Module
- **Input**: Multiple stock tickers, PSO parameters, risk tolerance
- **Process**: Data fetching, returns calculation, PSO optimization
- **Output**: Optimal portfolio allocation with risk metrics
- **Features**: Configurable PSO parameters, comprehensive risk analysis

## Logging

The application provides comprehensive logging:
- **Console Output**: Real-time progress and results
- **File Logging**: Detailed logs saved to `logs/` directory
- **API Calls**: All data fetching attempts and results
- **Model Actions**: Training, prediction, and optimization steps
- **Error Handling**: Detailed error messages and recovery attempts

## Error Handling

- **API Failures**: Automatic fallback to alternative data sources
- **Data Validation**: Input validation with user-friendly error messages
- **Model Errors**: Graceful handling of model training and prediction errors
- **Network Issues**: Retry mechanisms and timeout handling

## Performance Considerations

- **Model Persistence**: Trained models are saved and reused when possible
- **Data Caching**: Historical data is cached to reduce API calls
- **Memory Management**: Efficient data structures and memory usage
- **Parallel Processing**: PSO optimization uses vectorized operations

## Educational Purpose

This application is designed for educational purposes and demonstrates:
- Modern deep learning techniques for financial prediction
- Natural language processing for sentiment analysis
- Metaheuristic optimization algorithms for portfolio management
- Software engineering best practices and modular design

## Disclaimer

**Important**: This application is for educational purposes only. The predictions and recommendations provided should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or support, please open an issue in the repository or contact the development team.



