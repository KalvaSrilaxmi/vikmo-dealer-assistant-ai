import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from forecasting.baseline import calculate_mae, calculate_mape

def train_and_predict_rf(df, test_dates, n_estimators=100, random_state=42):
    """
    Prepares training and test data, performs one-hot encoding on SKU,
    trains a Random Forest Regressor, and returns forecasts on the test set.
    """
    # 1. Split into train and test sets
    train_df = df[~df['date'].isin(test_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    
    # 2. Define feature set
    base_features = [
        'promo_flag', 'week_of_year', 'month',
        'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_12', 'lag_52',
        'rolling_mean_4_w4', 'rolling_mean_8_w4'
    ]
    
    # One-hot encode the SKU column to capture SKU-specific demand biases
    # Ensure one-hot encoding columns are identical between train and test
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=['sku'], drop_first=True)
    
    # Identify the newly created one-hot columns
    sku_encoded_cols = [col for col in combined_encoded.columns if col.startswith('sku_')]
    
    features = base_features + sku_encoded_cols
    target = 'units_sold'
    
    # Split back to encoded train/test
    train_encoded = combined_encoded[~combined_encoded['date'].isin(test_dates)]
    test_encoded = combined_encoded[combined_encoded['date'].isin(test_dates)]
    
    X_train = train_encoded[features]
    y_train = train_encoded[target]
    X_test = test_encoded[features]
    y_test = test_encoded[target]
    
    # 3. Train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # 4. Predict
    preds = rf.predict(X_test)
    
    # Map predictions back to the test_df for per-SKU scoring
    test_df = test_df.copy()
    test_df['forecast'] = preds
    
    results = {}
    overall_actuals = []
    overall_forecasts = []
    
    for sku in df['sku'].unique():
        sku_test = test_df[test_df['sku'] == sku].sort_values('date')
        actual = sku_test['units_sold'].values
        forecast = sku_test['forecast'].values
        
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
    
    return results, overall_metrics, rf
