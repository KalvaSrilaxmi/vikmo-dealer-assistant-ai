import os
import pandas as pd
import numpy as np
from forecasting.data_prep import prepare_data
from forecasting.baseline import (
    run_naive_baseline,
    run_moving_average_baseline,
    run_seasonal_naive_baseline
)
from forecasting.model import train_and_predict_rf

def main():
    print("=" * 60)
    print("             VIKMO Demand Forecasting Runner")
    print("=" * 60)
    
    # 1. Prepare data
    print("Preparing sales history data and generating features...")
    df = prepare_data()
    
    # 2. Identify holdout test dates (last 4 weeks of data)
    unique_dates = sorted(df['date'].unique())
    test_dates = unique_dates[-4:]
    print(f"Test window: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} (4 weeks)")
    print(f"Training window: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-5].strftime('%Y-%m-%d')} (74 weeks)")
    print("-" * 60)
    
    # 3. Run Baselines
    print("Running Baseline 1: Naive (Last Value)...")
    naive_res, naive_overall = run_naive_baseline(df, test_dates)
    
    print("Running Baseline 2: Moving Average (4-week)...")
    ma_res, ma_overall = run_moving_average_baseline(df, test_dates, window_size=4)
    
    print("Running Baseline 3: Seasonal Naive (52-week)...")
    snaive_res, snaive_overall = run_seasonal_naive_baseline(df, test_dates)
    
    # 4. Run Machine Learning Model (Random Forest)
    print("Running Forecasting Model: Random Forest Regressor...")
    rf_res, rf_overall, rf_model = train_and_predict_rf(df, test_dates)
    print("-" * 60)
    
    # 5. Output Summary Results
    print("                     OVERALL PERFORMANCE COMPARISON")
    print("-" * 60)
    print(f"{'Model / Baseline':<35} | {'Overall MAE':<11} | {'Overall MAPE':<12}")
    print("-" * 60)
    print(f"{'Naive (Last Value)':<35} | {naive_overall['mae']:<11.4f} | {naive_overall['mape']:<11.2f}%")
    print(f"{'Moving Average (4-week)':<35} | {ma_overall['mae']:<11.4f} | {ma_overall['mape']:<11.2f}%")
    print(f"{'Seasonal Naive (52-week)':<35} | {snaive_overall['mae']:<11.4f} | {snaive_overall['mape']:<11.2f}%")
    print(f"{'Random Forest Regressor (Ours)':<35} | {rf_overall['mae']:<11.4f} | {rf_overall['mape']:<11.2f}%")
    print("-" * 60)
    
    # 6. SKU-wise breakdown for verification
    print("\nSKU-wise MAE Breakdown (Top 10 SKUs):")
    print("-" * 75)
    print(f"{'SKU':<10} | {'Naive MAE':<10} | {'MA4 MAE':<10} | {'SNaive MAE':<10} | {'RF MAE (Ours)':<12} | {'Improvement':<12}")
    print("-" * 75)
    
    skus = sorted(list(df['sku'].unique()))
    for sku in skus[:10]:
        n_mae = naive_res[sku]['mae']
        m_mae = ma_res[sku]['mae']
        s_mae = snaive_res[sku]['mae']
        r_mae = rf_res[sku]['mae']
        
        # Best baseline mae
        best_baseline = min(n_mae, m_mae, s_mae)
        imp = ((best_baseline - r_mae) / (best_baseline if best_baseline > 0 else 1.0)) * 100
        
        print(f"{sku:<10} | {n_mae:<10.2f} | {m_mae:<10.2f} | {s_mae:<10.2f} | {r_mae:<12.2f} | {imp:<11.1f}%")
    print("-" * 75)
    
    # Check if RF beats baselines
    beaten_naive = rf_overall['mae'] < naive_overall['mae']
    beaten_ma = rf_overall['mae'] < ma_overall['mae']
    beaten_snaive = rf_overall['mae'] < snaive_overall['mae']
    
    print("\nValidation Summary:")
    print(f"- Beats Naive (Last Value)? {'YES' if beaten_naive else 'NO'}")
    print(f"- Beats Moving Average (4-week)? {'YES' if beaten_ma else 'NO'}")
    print(f"- Beats Seasonal Naive (52-week)? {'YES' if beaten_snaive else 'NO'}")
    
    if beaten_naive and beaten_ma and beaten_snaive:
        print("\nSuccess! The Random Forest model beats all established baselines on the validation set.")
    else:
        print("\nWarning: The Random Forest model did not outperform all baselines. Tune parameters or features.")

if __name__ == "__main__":
    main()
