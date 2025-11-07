"""
NLP Module - Financial News Sentiment Analysis using BiLSTM + Attention
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Embedding, Bidirectional, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from utils.data_fetcher import DataFetcher
from utils.logger import setup_logger, log_data_processing, log_model_action


class NLPModule:
    """NLP module for financial news sentiment analysis with BiLSTM architecture."""

    def __init__(self):
        self.logger = setup_logger("NLPModule")
        self.data_fetcher = DataFetcher()
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.max_length = 100
        self.vocab_size = 10000
        self.model_path = "models/bilstm_sentiment_model.h5"

    def run(self):
        """Main execution workflow."""
        try:
            symbol = self._get_user_inputs()
            news_data = self._fetch_news_data(symbol)
            processed_data = self._preprocess_news_data(news_data)

            self._prepare_model(processed_data)
            sentiment_results = self._analyze_sentiment(processed_data)

            self._display_results(sentiment_results, symbol)

        except Exception as e:
            self.logger.error(f"Error in NLP module: {e}")
            print(f"Error: {e}")

    def _get_user_inputs(self) -> str:
        """Prompt user for stock ticker symbol."""
        print("\n" + "-" * 40)
        print("SENTIMENT ANALYSIS INPUTS")
        print("-" * 40)

        while True:
            symbol = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                break
            print("Please enter a valid ticker symbol.")

        self.logger.info(f"User input - Symbol: {symbol}")
        return symbol

    def _fetch_news_data(self, symbol: str) -> List[Dict]:
        """Fetch financial news data for a given stock symbol."""
        news_data = self.data_fetcher.fetch_news_data(symbol, limit=100)
        log_data_processing(self.logger, "News Fetch", f"Fetched {len(news_data)} articles for {symbol}")

        if not news_data:
            raise Exception(f"No news data available for {symbol}")
        return news_data

    def _preprocess_news_data(self, news_data: List[Dict]) -> pd.DataFrame:
        """Clean and structure raw news data."""
        log_data_processing(self.logger, "Text Preprocessing", "Cleaning and tokenizing news headlines")

        processed_articles = []
        for article in news_data:
            title, summary = article.get("title", ""), article.get("summary", "")
            text = f"{title} {summary}".strip()
            if not text:
                continue

            cleaned_text = self._clean_text(text)
            processed_articles.append({
                "text": cleaned_text,
                "original_title": title,
                "sentiment_label": article.get("sentiment_label", "Neutral"),
                "sentiment_score": article.get("sentiment_score", 0.0),
                "time_published": article.get("time_published", "")
            })

        df = pd.DataFrame(processed_articles)
        log_data_processing(self.logger, "Text Preprocessing", f"Processed {len(df)} articles")
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        """Apply regex-based cleaning to raw text."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_model(self, data: pd.DataFrame):
        """Load or train a BiLSTM model for sentiment classification."""
        if self._try_load_existing_model():
            return

        log_model_action(self.logger, "Model Creation", "Building BiLSTM model")

        # Tokenize and encode data
        texts, labels = data["text"].tolist(), data["sentiment_label"].tolist()
        self.tokenizer.fit_on_texts(texts)
        X = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_length)
        y = self.label_encoder.fit_transform(labels)

        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # Class weights for imbalance
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Build model
        self.model = Sequential([
            Embedding(self.vocab_size, 128),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.4),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(len(self.label_encoder.classes_), activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        log_model_action(self.logger, "Model Training", f"Training on {len(X)} samples")
        self.model.fit(
            X_train, y_train,
            batch_size=16, epochs=30, verbose=0,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict
        )

        self._save_model()
        self._evaluate_model(X_test, y_test)

    def _try_load_existing_model(self) -> bool:
        """Attempt to load an existing trained model."""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                with open("models/tokenizer.pkl", "rb") as f:
                    self.tokenizer = pickle.load(f)
                with open("models/label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.logger.info(f"Loaded existing model from {self.model_path}")
                return True
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}")
        return False

    def _save_model(self):
        """Persist model and preprocessing objects."""
        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)
        with open("models/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        self.logger.info(f"Model and preprocessors saved at {self.model_path}")

    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the trained model."""
        preds = np.argmax(self.model.predict(X_test), axis=1)
        accuracy = np.mean(preds == y_test)
        log_model_action(self.logger, "Model Evaluation", f"Test Accuracy: {accuracy:.4f}")

    def _analyze_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict sentiment for given news articles."""
        X = pad_sequences(self.tokenizer.texts_to_sequences(data["text"].tolist()), maxlen=self.max_length)
        predictions = self.model.predict(X)

        predicted_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        sentiments = self.label_encoder.inverse_transform(predicted_labels)

        enhanced = [self._enhance_sentiment_prediction(t, s, c)
                    for t, s, c in zip(data["text"], sentiments, confidences)]

        results = data.copy()
        results["predicted_sentiment"], results["confidence_score"] = zip(*enhanced)
        return results

    def _enhance_sentiment_prediction(self, text: str, sentiment: str, confidence: float) -> Tuple[str, float]:
        """Refine sentiment prediction using keyword rules."""
        text_lower = text.lower()

        bullish_kw = ["strong", "growth", "increase", "rise", "gain", "positive", "exceed",
                      "beat", "surge", "robust", "record", "expansion", "upgrade", "profit"]
        bearish_kw = ["decline", "drop", "fall", "negative", "concern", "pressure",
                      "regulatory", "slow", "weak", "miss", "risk", "uncertainty", "struggle"]
        neutral_kw = ["maintain", "steady", "stable", "update", "announce", "meeting", "mixed"]

        bullish, bearish, neutral = (
            sum(kw in text_lower for kw in bullish_kw),
            sum(kw in text_lower for kw in bearish_kw),
            sum(kw in text_lower for kw in neutral_kw),
        )

        if bullish > bearish and bullish > neutral:
            return "Bullish", min(0.9, confidence + 0.2)
        if bearish > bullish and bearish > neutral:
            return "Bearish", min(0.9, confidence + 0.2)
        if neutral > 1 or bullish == bearish:
            return "Neutral", min(0.9, confidence + 0.1)

        return sentiment, max(confidence, 0.6)

    def _display_results(self, results: pd.DataFrame, symbol: str):
        """Display sentiment analysis results in a structured format."""
        print("\n" + "=" * 80)
        print("                    SENTIMENT ANALYSIS RESULTS")
        print("=" * 80)
        print(f"\nSentiment Analysis for {symbol} News Articles:")
        print("-" * 80)
        print(f"{'Headline':<50} | {'Sentiment':<10} | {'Confidence':<10}")
        print("-" * 80)

        for _, row in results.iterrows():
            headline = row["original_title"][:47] + "..." if len(row["original_title"]) > 50 else row["original_title"]
            print(f"{headline:<50} | {row['predicted_sentiment']:<10} | {row['confidence_score']:.1%}")

        sentiment_counts = results["predicted_sentiment"].value_counts()
        bullish, bearish, neutral = [sentiment_counts.get(s, 0) for s in ["Bullish", "Bearish", "Neutral"]]
        total = len(results)

        print(f"\nAggregate Sentiment Analysis for {symbol}:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment:<15}: {count:>3} articles ({(count / total) * 100:>5.1f}%)")

        print("\nOverall Market Sentiment:")
        if bullish > max(bearish, neutral):
            print(f"Trend: BULLISH 📈")
        elif bearish > max(bullish, neutral):
            print(f"Trend: BEARISH 📉")
        else:
            print(f"Trend: NEUTRAL ➡️")

        print(f"Bullish: {bullish} ({bullish/total:.1%})")
        print(f"Bearish: {bearish} ({bearish/total:.1%})")
        print(f"Neutral: {neutral} ({neutral/total:.1%})")

        print("\nConfidence Analysis:")
        print(f"Average Confidence: {results['confidence_score'].mean():.1%}")
        print(f"High Confidence (>80%): {(results['confidence_score'] > 0.8).sum()} articles")
        print("\n" + "=" * 80)
        print("Note: Sentiment analysis is based on news headlines and summaries.")
        print("Results are for educational purposes only.")
        print("=" * 80)