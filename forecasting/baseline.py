import numpy as np
import pandas as pd

def calculate_mape(actuals, forecasts):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles zero values in actuals by replacing them with 1.0 in the denominator.
    """
    actuals_denom = np.where(actuals == 0, 1.0, actuals)
    return np.mean(np.abs(actuals - forecasts) / actuals_denom) * 100

def calculate_mae(actuals, forecasts):
    """Calculates Mean Absolute Error (MAE)."""
    return np.mean(np.abs(actuals - forecasts))

def run_naive_baseline(df, test_dates):
    """
    Computes Naive Last-Value baseline:
    For each SKU, predicts the demand of the last training week for all test weeks.
    """
    train_df = df[~df['date'].isin(test_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    results = {}
    overall_actuals = []
    overall_forecasts = []
    
    for sku in df['sku'].unique():
        sku_train = train_df[train_df['sku'] == sku].sort_values('date')
        sku_test = test_df[test_df['sku'] == sku].sort_values('date')
        
        # Last value from train
        last_val = sku_train.iloc[-1]['units_sold']
        
        actual = sku_test['units_sold'].values
        forecast = np.full_like(actual, last_val)
        
        overall_actuals.extend(actual)
        overall_forecasts.extend(forecast)
        
        results[sku] = {
            "mae": calculate_mae(actual, forecast),
            "mape": calculate_mape(actual, forecast),
            "forecast": list(forecast)
        }
        
    overall_actuals = np.array(overall_actuals)
    overall_forecasts = np.array(overall_forecasts)
    
    overall_metrics = {
        "mae": calculate_mae(overall_actuals, overall_forecasts),
        "mape": calculate_mape(overall_actuals, overall_forecasts)
    }
    
    return results, overall_metrics

def run_moving_average_baseline(df, test_dates, window_size=4):
    """
    Computes Moving Average baseline:
    For each SKU, predicts the average of the last `window_size` training weeks.
    """
    train_df = df[~df['date'].isin(test_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    results = {}
    overall_actuals = []
    overall_forecasts = []
    
    for sku in df['sku'].unique():
        sku_train = train_df[train_df['sku'] == sku].sort_values('date')
        sku_test = test_df[test_df['sku'] == sku].sort_values('date')
        
        # Moving average of last window_size weeks
        ma_val = sku_train.iloc[-window_size:]['units_sold'].mean()
        
        actual = sku_test['units_sold'].values
        forecast = np.full_like(actual, ma_val)
        
        overall_actuals.extend(actual)
        overall_forecasts.extend(forecast)
        
        results[sku] = {
            "mae": calculate_mae(actual, forecast),
            "mape": calculate_mape(actual, forecast),
            "forecast": list(forecast)
        }
        
    overall_actuals = np.array(overall_actuals)
    overall_forecasts = np.array(overall_forecasts)
    
    overall_metrics = {
        "mae": calculate_mae(overall_actuals, overall_forecasts),
        "mape": calculate_mape(overall_actuals, overall_forecasts)
    }
    
    return results, overall_metrics

def run_seasonal_naive_baseline(df, test_dates):
    """
    Computes Seasonal Naive baseline:
    For each test week t, predicts the demand from 52 weeks ago (t-52).
    """
    results = {}
    overall_actuals = []
    overall_forecasts = []
    
    for sku in df['sku'].unique():
        sku_full = df[df['sku'] == sku].sort_values('date').reset_index(drop=True)
        
        # Get test set rows and indices
        test_rows = sku_full[sku_full['date'].isin(test_dates)]
        test_indices = test_rows.index
        
        actual = test_rows['units_sold'].values
        
        # Retrieve values from 52 weeks prior
        forecast = []
        for idx in test_indices:
            prev_idx = idx - 52
            if prev_idx >= 0:
                forecast.append(sku_full.iloc[prev_idx]['units_sold'])
            else:
                # Fallback to training median if 52 weeks ago is out of range
                forecast.append(sku_full.iloc[:idx]['units_sold'].median())
                
        forecast = np.array(forecast)
        
        overall_actuals.extend(actual)
        overall_forecasts.extend(forecast)
        
        results[sku] = {
            "mae": calculate_mae(actual, forecast),
            "mape": calculate_mape(actual, forecast),
            "forecast": list(forecast)
        }
        
    overall_actuals = np.array(overall_actuals)
    overall_forecasts = np.array(overall_forecasts)
    
    overall_metrics = {
        "mae": calculate_mae(overall_actuals, overall_forecasts),
        "mape": calculate_mape(overall_actuals, overall_forecasts)
    }
    
    return results, overall_metrics
