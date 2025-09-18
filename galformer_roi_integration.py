"""
Galformer ROI Integration Script

This script integrates the ROI calculator directly with your Galformer model
to analyze trading performance on predictions.
"""

import numpy as np
import pandas as pd
import torch
import yaml
from roi_calculator import ROICalculator, quick_roi_analysis, compare_strategies

def load_galformer_predictions_and_prices(config_path='config.yaml', model_path=None):
    """
    Load Galformer predictions and corresponding actual prices
    
    This function demonstrates how to integrate with your existing Galformer setup
    """
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load actual price data
    filename = config['data']['filename']
    df = pd.read_csv(filename, delimiter=',', usecols=['Date','Open','High','Low','Close', 'Adj Close','Volume'])
    df = df.sort_values('Date')
    
    # Get the test portion of actual prices
    division_rate2 = config['data']['division_rate2']
    seq_len = config['sequence']['src_len']
    
    # Extract actual prices for the test period
    all_prices = df['Adj Close'].values
    test_start_idx = int(division_rate2 * len(all_prices)) + seq_len
    actual_test_prices = all_prices[test_start_idx:]
    
    print(f"Loaded {len(actual_test_prices)} test price points")
    print(f"Test price range: ${actual_test_prices.min():.2f} - ${actual_test_prices.max():.2f}")
    
    # For demonstration, create synthetic predictions
    # In practice, you would load actual model predictions here
    predictions = simulate_galformer_predictions(actual_test_prices)
    
    return predictions, actual_test_prices, config

def simulate_galformer_predictions(actual_prices, skill_level=0.3):
    """
    Simulate Galformer predictions for demonstration
    
    Replace this with actual model predictions in practice
    """
    predictions = []
    
    for i in range(len(actual_prices)):
        if i == 0:
            predictions.append(actual_prices[i])
        else:
            # Simulate prediction with some skill
            actual_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            predicted_change = actual_change * skill_level + np.random.normal(0, 0.015)
            pred_price = actual_prices[i-1] * (1 + predicted_change)
            predictions.append(pred_price)
    
    return np.array(predictions)

def analyze_galformer_roi(config_path='config.yaml', 
                         initial_capital=10000,
                         strategies=['long_only', 'threshold'],
                         save_results=True):
    """
    Complete ROI analysis for Galformer predictions
    """
    
    print("🚀 GALFORMER ROI ANALYSIS")
    print("="*50)
    
    # Load predictions and actual prices
    predictions, actual_prices, config = load_galformer_predictions_and_prices(config_path)
    
    # Create ROI calculator
    calculator = ROICalculator(initial_capital=initial_capital)
    
    # Analyze each strategy
    all_results = {}
    
    for strategy in strategies:
        print(f"\n📊 Analyzing {strategy.upper()} strategy...")
        
        results = calculator.calculate_roi(
            predictions=predictions,
            actual_prices=actual_prices,
            strategy=strategy,
            transaction_cost=0.001  # 0.1% transaction cost
        )
        
        all_results[strategy] = results
        
        # Print summary
        buy_hold = calculator.calculate_buy_hold_roi(actual_prices)
        print(f"  Strategy Return: {results['total_return_pct']:.2f}%")
        print(f"  Buy & Hold: {buy_hold:.2f}%")
        print(f"  Excess Return: {results['total_return_pct'] - buy_hold:.2f}%")
        print(f"  Number of Trades: {results['num_trades']}")
        
        # Create detailed analysis plot
        if save_results:
            plot_path = f"galformer_roi_{strategy}.png"
            calculator.plot_comprehensive_analysis(results, save_path=plot_path)
            
            # Save detailed results
            csv_path = f"galformer_roi_{strategy}_results.csv"
            calculator.save_results_to_csv(results, csv_path)
    
    # Print detailed results for best strategy
    best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['total_return_pct'])
    print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
    calculator.print_detailed_results(all_results[best_strategy])
    
    return all_results

def quick_galformer_analysis(config_path='config.yaml'):
    """
    Quick ROI analysis with default settings
    """
    predictions, actual_prices, config = load_galformer_predictions_and_prices(config_path)
    
    return quick_roi_analysis(
        predictions=predictions,
        actual_prices=actual_prices,
        initial_capital=10000,
        strategy='long_only',
        plot=True,
        save_path='quick_galformer_roi.png'
    )

def roi_sensitivity_analysis(config_path='config.yaml'):
    """
    Analyze ROI sensitivity to different parameters
    """
    print("\n🔬 ROI SENSITIVITY ANALYSIS")
    print("="*50)
    
    predictions, actual_prices, config = load_galformer_predictions_and_prices(config_path)
    
    # Test different initial capitals
    capitals = [5000, 10000, 25000, 50000]
    transaction_costs = [0.0005, 0.001, 0.002, 0.005]  # 0.05% to 0.5%
    
    results_summary = []
    
    calculator = ROICalculator()
    
    print("\n💰 Testing different initial capitals:")
    for capital in capitals:
        calculator.initial_capital = capital
        results = calculator.calculate_roi(predictions, actual_prices, strategy='long_only')
        print(f"  ${capital:,}: {results['total_return_pct']:.2f}% return")
        results_summary.append({
            'Capital': capital,
            'Return_Pct': results['total_return_pct'],
            'Final_Value': results['final_value'],
            'Num_Trades': results['num_trades']
        })
    
    print("\n💸 Testing different transaction costs:")
    calculator.initial_capital = 10000
    for cost in transaction_costs:
        results = calculator.calculate_roi(predictions, actual_prices, 
                                         strategy='long_only', transaction_cost=cost)
        print(f"  {cost*100:.2f}%: {results['total_return_pct']:.2f}% return, {results['num_trades']} trades")
    
    return results_summary

if __name__ == "__main__":
    print("🎯 GALFORMER ROI INTEGRATION")
    print("This script analyzes ROI performance of Galformer predictions")
    
    try:
        # Quick analysis
        print("\n1️⃣ Running quick analysis...")
        quick_results = quick_galformer_analysis()
        
        # Comprehensive analysis
        print("\n2️⃣ Running comprehensive analysis...")
        comprehensive_results = analyze_galformer_roi(
            strategies=['long_only', 'long_short', 'threshold'],
            save_results=True
        )
        
        # Sensitivity analysis
        print("\n3️⃣ Running sensitivity analysis...")
        sensitivity_results = roi_sensitivity_analysis()
        
        print("\n✅ All analyses completed!")
        print("Check the generated PNG files and Excel spreadsheets for detailed results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your config.yaml and data files are properly set up.")
