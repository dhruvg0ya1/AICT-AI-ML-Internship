import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol, period='5y'):
        """
        Initialize the Stock Predictor
        
        Parameters:
        symbol (str): Stock symbol (e.g., 'AAPL', 'TCS.NS', 'INFY.NS')
        period (str): Time period for historical data
        """
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {self.symbol}...")
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"Data fetched successfully! Shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def preprocess_data(self, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume']):
        """
        Clean and preprocess the stock data
        
        Parameters:
        feature_columns (list): List of columns to use as features
        """
        if self.data is None:
            print("No data available. Please fetch data first.")
            return None
        
        # Check for missing values
        print(f"Missing values before cleaning: {self.data.isnull().sum().sum()}")
        
        # Handle missing values
        self.data = self.data.dropna()
        
        # Select features
        self.features = self.data[feature_columns].values
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(self.features)
        
        print(f"Data preprocessed successfully! Shape: {self.scaled_data.shape}")
        print(f"Missing values after cleaning: {pd.DataFrame(self.scaled_data).isnull().sum().sum()}")
        
        return self.scaled_data
    
    def create_sequences(self, sequence_length=60, target_column=3):
        """
        Create sequences for LSTM training
        
        Parameters:
        sequence_length (int): Number of time steps to look back
        target_column (int): Index of the target column (Close price)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-sequence_length:i])
            y.append(self.scaled_data[i, target_column])  # Close price
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_size=0.8):
        """Split data into training and testing sets"""
        split_index = int(len(X) * train_size)
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, lstm_units=[50, 50], dropout_rate=0.2):
        """
        Build LSTM model
        
        Parameters:
        input_shape (tuple): Shape of input data
        lstm_units (list): List of LSTM units for each layer
        dropout_rate (float): Dropout rate for regularization
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units=lstm_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        self.model = model
        print("Model built successfully!")
        print(model.summary())
        
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            print("Model not built. Please build the model first.")
            return None
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Model training completed!")
        return history
    
    def make_predictions(self, X_test):
        """Make predictions using the trained model"""
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        predictions = self.model.predict(X_test)
        return predictions
    
    def inverse_transform_predictions(self, predictions, y_test):
        """
        Inverse transform predictions and actual values to original scale
        """
        # Create arrays with the same shape as original scaled data
        pred_extended = np.zeros((len(predictions), self.scaled_data.shape[1]))
        test_extended = np.zeros((len(y_test), self.scaled_data.shape[1]))
        
        # Place predictions and actual values in the Close price column (index 3)
        pred_extended[:, 3] = predictions.flatten()
        test_extended[:, 3] = y_test.flatten()
        
        # Inverse transform
        pred_actual = self.scaler.inverse_transform(pred_extended)[:, 3]
        test_actual = self.scaler.inverse_transform(test_extended)[:, 3]
        
        return pred_actual, test_actual
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nModel Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    def plot_results(self, y_true, y_pred, history=None):
        """Plot training history and prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training history
        if history is not None:
            axes[0, 0].plot(history.history['loss'], label='Training Loss')
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Plot predictions vs actual
        axes[0, 1].plot(y_true, label='Actual Prices', alpha=0.7)
        axes[0, 1].plot(y_pred, label='Predicted Prices', alpha=0.7)
        axes[0, 1].set_title(f'{self.symbol} - Actual vs Predicted Prices')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        
        # Plot scatter plot
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Prices')
        axes[1, 0].set_ylabel('Predicted Prices')
        axes[1, 0].set_title('Actual vs Predicted Scatter Plot')
        
        # Plot residuals
        residuals = y_true - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Prices')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def plot_price_trend(self):
        """Plot the historical price trend"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.title(f'{self.symbol} - Historical Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def run_complete_pipeline(self, sequence_length=60, epochs=100, batch_size=32):
        """Run the complete stock prediction pipeline"""
        print(f"Starting stock price prediction for {self.symbol}")
        print("=" * 50)
        
        # Step 1: Fetch data
        if self.fetch_data() is None:
            return None
        
        # Step 2: Preprocess data
        if self.preprocess_data() is None:
            return None
        
        # Step 3: Create sequences
        X, y = self.create_sequences(sequence_length)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 5: Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_model(input_shape)
        
        # Step 6: Train model
        history = self.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)
        
        # Step 7: Make predictions
        predictions = self.make_predictions(X_test)
        
        # Step 8: Inverse transform predictions
        pred_actual, test_actual = self.inverse_transform_predictions(predictions, y_test)
        
        # Step 9: Evaluate model
        metrics = self.evaluate_model(test_actual, pred_actual)
        
        # Step 10: Plot results
        self.plot_results(test_actual, pred_actual, history)
        
        return {
            'predictions': pred_actual,
            'actual': test_actual,
            'metrics': metrics,
            'history': history
        }

def main():
    """Main function to run stock prediction"""
    # Example usage with different stocks
    stocks = ['AAPL', 'TCS.NS', 'INFY.NS']  # Apple, TCS, Infosys
    
    for stock in stocks:
        print(f"\n{'='*60}")
        print(f"ANALYZING {stock}")
        print(f"{'='*60}")
        
        try:
            # Create predictor instance
            predictor = StockPredictor(stock, period='2y')
            
            # Show historical trend
            predictor.fetch_data()
            predictor.plot_price_trend()
            
            # Run complete pipeline
            results = predictor.run_complete_pipeline(
                sequence_length=60,
                epochs=50,  # Reduced for faster execution
                batch_size=32
            )
            
            if results:
                print(f"\nSuccessfully completed prediction for {stock}")
                print(f"Final R² Score: {results['metrics']['R2']:.4f}")
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue

if __name__ == "__main__":
    # For individual stock analysis
    print("Stock Price Prediction using LSTM")
    print("Choose a stock to analyze:")
    print("1. AAPL (Apple)")
    print("2. TCS.NS (Tata Consultancy Services)")
    print("3. INFY.NS (Infosys)")
    print("4. Custom stock symbol")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        symbol = 'AAPL'
    elif choice == '2':
        symbol = 'TCS.NS'
    elif choice == '3':
        symbol = 'INFY.NS'
    elif choice == '4':
        symbol = input("Enter stock symbol: ").upper()
    else:
        symbol = 'AAPL'  # Default
    
    # Create and run predictor
    predictor = StockPredictor(symbol, period='2y')
    results = predictor.run_complete_pipeline()
    
    if results:
        print(f"\nPrediction completed for {symbol}")
        print("Check the plots above for visual analysis!")
