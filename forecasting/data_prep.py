import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
SALES_CSV_PATH = os.path.join(WORKSPACE_DIR, "sales_history.csv")

def prepare_data(csv_path=SALES_CSV_PATH, train_end_date='2026-05-11'):
    """
    Loads sales_history.csv, formats columns, extracts datetime features,
    generates lag and rolling features, and performs imputation using
    training-set statistics to prevent data leakage.
    
    Lags are shifted by at least 4 weeks to enable 4-week-ahead forecasting.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sales history file not found at {csv_path}")

    # 1. Load and sort data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['sku', 'date']).reset_index(drop=True)
    
    # 2. Extract calendar attributes
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    df['promo_flag'] = df['promo_flag'].astype(int)
    
    # 3. Create lag features (shift by at least 4 weeks to prevent leakage in 4-week forecasting)
    # Lags: 4, 5, 6, 7, 8, 12, 52
    lag_cols = [4, 5, 6, 7, 8, 12, 52]
    for lag in lag_cols:
        df[f'lag_{lag}'] = df.groupby('sku')['units_sold'].shift(lag)
        
    # 4. Create rolling window features based on the 4-week shifted series
    # (Since 4 is the minimum lag, shifting by 4 and taking rolling stats prevents leakage of t-1, t-2, t-3)
    shifted_series = df.groupby('sku')['units_sold'].shift(4)
    df['rolling_mean_4_w4'] = shifted_series.groupby(df['sku']).transform(lambda x: x.rolling(4).mean())
    df['rolling_mean_8_w4'] = shifted_series.groupby(df['sku']).transform(lambda x: x.rolling(8).mean())
    
    # 5. Prevent Leakage in Imputation: Calculate SKU medians ONLY using training data
    # (data on or before train_end_date)
    train_mask = df['date'] <= pd.to_datetime(train_end_date)
    train_df = df[train_mask]
    
    # Calculate median units_sold per SKU in training set
    sku_medians = train_df.groupby('sku')['units_sold'].median().to_dict()
    
    # Map the medians back to the rows for filling NaNs
    sku_medians_series = df['sku'].map(sku_medians)
    
    # Fill NaN values for all generated lag/rolling features
    feature_cols = [f'lag_{lag}' for lag in lag_cols] + ['rolling_mean_4_w4', 'rolling_mean_8_w4']
    for col in feature_cols:
        df[col] = df[col].fillna(sku_medians_series)
        
    # Fill any remaining NaNs (e.g. if a SKU didn't have median, which shouldn't happen here)
    # with the overall training median
    overall_median = train_df['units_sold'].median()
    df = df.fillna(overall_median)
    
    return df

if __name__ == "__main__":
    df = prepare_data()
    print("Data preparation complete. Shape:", df.shape)
    print("Columns in prepared dataset:", list(df.columns))
    print("Checking for nulls:\n", df.isnull().sum())
    print("\nSample processed rows for SKU BRK-1015:")
    print(df[df['sku'] == 'BRK-1015'].head(3))
