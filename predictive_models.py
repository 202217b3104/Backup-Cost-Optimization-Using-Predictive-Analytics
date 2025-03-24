# Treat this file as a module
from __future__ import annotations

from .data import CohesityDataProcessor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta




class BackupPredictionModels:
    def __init__(self, csv_file=None):
        """
        Initialize prediction models using a single CSV file.
        If csv_file is provided, load data; otherwise, data can be set manually.
        """
        if csv_file:
            self.processor = CohesityDataProcessor(csv_file)
            self.processor.load_data()
            self.df = self.processor.df
        else:
            self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.prophet_model = None
        self.linear_model = None
        self.rf_model = None

    def set_data(self, df):
        """Set data directly from a DataFrame."""
        self.df = df

    def get_daily_data(self):
        """
        Aggregate daily backup data.
        Returns a DataFrame with daily aggregated metrics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please load data first.")
        processor = CohesityDataProcessor(None)
        processor.df = self.df
        daily_data = processor.aggregate_daily_backup_data()
        # Rename 'date' to 'backup_day' to match the expected naming
        daily_data.rename(columns={'date': 'backup_day'}, inplace=True)
        return daily_data

    def train_linear_regression(self, daily_data):
        """
        Train a linear regression model.
        
        Args:
            daily_data (DataFrame): Daily aggregated backup data.
            
        Returns:
            model: Trained linear regression model.
        """
        df = daily_data.copy()
        df['days_since_start'] = (df['backup_day'] - df['backup_day'].min()).dt.days
        
        # Use storage_used_GB as the target variable for physical storage
        X = df[['days_since_start']]
        y = df['storage_used_GB']
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.linear_model = model
        return model

    def train_random_forest(self, daily_data, features=None):
        """
        Train a random forest model for backup growth prediction.
        
        Args:
            daily_data (DataFrame): Daily aggregated backup data.
            features (list): List of features to use (default: days_since_start, day_of_week, day_of_month).
            
        Returns:
            model: Trained random forest model.
        """
        df = daily_data.copy()
        df['days_since_start'] = (df['backup_day'] - df['backup_day'].min()).dt.days
        df['day_of_week'] = pd.to_datetime(df['backup_day']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['backup_day']).dt.day
        
        if features is None:
            features = ['days_since_start', 'day_of_week', 'day_of_month']
        
        X = df[features]
        y = df['storage_used_GB']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        self.rf_model = model
        return model

    def train_lstm_model(self, X, y, epochs=50, batch_size=32):
        """
        Train an LSTM model for time series prediction.
        
        Args:
            X (ndarray): Input features (sequence of historical values).
            y (ndarray): Target values.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            model: Trained LSTM model.
        """
        # Reshape input for LSTM [samples, time steps, features]
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Scale data for each sample
        X_scaled = np.array([self.scaler.fit_transform(x.reshape(-1, 1)).reshape(-1) for x in X_reshaped])
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_scaled.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
        
        self.lstm_model = model
        return model

    def train_prophet_model(self, daily_data):
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            daily_data (DataFrame): Daily aggregated backup data.
            
        Returns:
            model: Trained Prophet model.
        """
        df = daily_data.copy()
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['backup_day']),
            'y': df['storage_used_GB']  # Use storage_used_GB as the target variable
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        self.prophet_model = model
        return model

    def predict_future_storage(self, daily_data, prediction_days=90, model_type='prophet'):
        """
        Predict future storage usage.
        
        Args:
            daily_data (DataFrame): Daily aggregated backup data.
            prediction_days (int): Number of days to predict into the future.
            model_type (str): Type of model to use (prophet, linear, rf).
            
        Returns:
            DataFrame: Predictions for future dates.
        """
        last_date = pd.to_datetime(daily_data['backup_day']).max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        if model_type == 'prophet':
            if self.prophet_model is None:
                self.train_prophet_model(daily_data)
            future = pd.DataFrame({'ds': future_dates})
            forecast = self.prophet_model.predict(future)
            predictions = pd.DataFrame({
                'date': forecast['ds'],
                'predicted_storage': forecast['yhat'],
                'lower_bound': forecast['yhat_lower'],
                'upper_bound': forecast['yhat_upper']
            })
        elif model_type == 'linear':
            if self.linear_model is None:
                self.train_linear_regression(daily_data)
            start_date = pd.to_datetime(daily_data['backup_day']).min()
            days_since_start = [(date - start_date).days for date in future_dates]
            future_storage = self.linear_model.predict(np.array(days_since_start).reshape(-1, 1))
            predictions = pd.DataFrame({
                'date': future_dates,
                'predicted_storage': future_storage
            })
        elif model_type == 'rf':
            if self.rf_model is None:
                self.train_random_forest(daily_data)
            start_date = pd.to_datetime(daily_data['backup_day']).min()
            future_features = pd.DataFrame({
                'days_since_start': [(date - start_date).days for date in future_dates],
                'day_of_week': [date.dayofweek for date in future_dates],
                'day_of_month': [date.day for date in future_dates]
            })
            future_storage = self.rf_model.predict(future_features)
            predictions = pd.DataFrame({
                'date': future_dates,
                'predicted_storage': future_storage
            })
        else:
            raise ValueError("Invalid model_type. Choose from 'prophet', 'linear', or 'rf'.")
        
        # Calculate cumulative storage prediction
        last_storage_value = daily_data['storage_used_GB'].iloc[-1]
        predictions['cumulative_storage'] = last_storage_value + predictions['predicted_storage'].cumsum()
        return predictions

    def plot_predictions(self, historical_data, predictions, title="Backup Storage Growth Prediction"):
        """
        Plot historical data and predictions.
        
        Args:
            historical_data (DataFrame): Historical backup data.
            predictions (DataFrame): Predicted future data.
            title (str): Plot title.
            
        Returns:
            Matplotlib plot.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(historical_data['backup_day']), historical_data['storage_used_GB'], 'b-', label='Historical Storage Usage')
        plt.plot(predictions['date'], predictions['predicted_storage'], 'r--', label='Predicted Storage Usage')
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            plt.fill_between(predictions['date'], predictions['lower_bound'], predictions['upper_bound'], color='r', alpha=0.2, label='95% Confidence Interval')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Storage Used (GB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt

# Example usage:
if __name__ == "__main__":
    predictor = BackupPredictionModels("backup_data.csv")
    daily_data = predictor.get_daily_data()
    # Train models (example using Prophet)
    predictor.train_linear_regression(daily_data)
    predictor.train_random_forest(daily_data)
    predictor.train_prophet_model(daily_data)
    predictions = predictor.predict_future_storage(daily_data, prediction_days=90, model_type='prophet')
    print(predictions.head())
