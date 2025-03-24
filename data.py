# Treat this file as a module
from __future__ import annotations

import pandas as pd
import numpy as np




class CohesityDataProcessor:
    def __init__(self, csv_file):

        self.csv_file = csv_file
        self.df = None

    def load_data(self):
        """
        Load the backup data from a CSV file.
        """
        try:
            self.df = pd.read_csv(self.csv_file)

            # Convert date column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'], format='%d-%m-%Y')

            # Sort data chronologically
            self.df.sort_values(by=['date'], inplace=True)

            print("✅ Data loaded successfully!")
            print(self.df.head())

        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")

    def aggregate_daily_backup_data(self):
        """
        Aggregate backup data on a daily basis.
        """
        if self.df is None:
            print("❌ Data not loaded. Run `load_data()` first.")
            return None

        daily_summary = self.df.groupby('date').agg({
            'backup_size_GB': 'sum',
            'storage_used_GB': 'sum',
            'cost_per_GB': 'mean'
        }).reset_index()

        # Compute deduplication efficiency per day
        daily_summary['daily_dedup_ratio'] = daily_summary['backup_size_GB'] / daily_summary['storage_used_GB']
        
        print("✅ Daily backup summary generated!")
        return daily_summary

    def prepare_prediction_dataset(self, days_window=30):
        """
        Prepare data for machine learning predictions using a sliding window.
        """
        if self.df is None:
            print("❌ Data not loaded. Run `load_data()` first.")
            return None, None

        daily_summary = self.aggregate_daily_backup_data()
        storage_data = daily_summary['storage_used_GB'].values

        X, y = [], []
        for i in range(len(storage_data) - days_window):
            X.append(storage_data[i:i + days_window])
            y.append(storage_data[i + days_window])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Prediction dataset prepared! Training Data Shape: {X.shape}, Target Data Shape: {y.shape}")
        return X, y

# Example Usage
processor = CohesityDataProcessor("backup_data1.csv")

# Load and process the data
processor.load_data()

# Get daily backup summary
daily_backup = processor.aggregate_daily_backup_data()
print(daily_backup.head())

# Prepare the dataset for predictions
X, y = processor.prepare_prediction_dataset()
