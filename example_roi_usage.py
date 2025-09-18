"""
Example Usage of ROI Calculator with Galformer Predictions

This script demonstrates how to use the roi_calculator module with 
predictions from the Galformer model.
"""

import numpy as np
import pandas as pd
from roi_calculator import ROICalculator, quick_roi_analysis, compare_strategies

def load_sample_data(csv_file='Datasets/BTC-USD.csv', n_samples=200):
    """Load sample data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        prices = df['Adj Close'].values[-n_samples:]  # Last n_samples days
        return prices
    except FileNotFoundError:
        print(f"File {csv_file} not found. Generating synthetic data...")
        return generate_synthetic_data(n_samples)

def generate_synthetic_data(n_days=200):
    """Generate synthetic price data for demonstration"""
    np.random.seed(42)
    base_price = 45000  # Starting BTC price
    prices = [base_price]
    
    for i in range(n_days - 1):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.03)  # 0.1% daily trend, 3% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Prevent unrealistic low prices
    
    return np.array(prices)

def simulate_galformer_predictions(actual_prices, skill_level=0.2):
    """
    Simulate Galformer predictions with varying skill levels
    
    Args:
        actual_prices: Array of actual prices
        skill_level: Prediction skill (0=random, 1=perfect)
    """
    predictions = []
    
    for i in range(len(actual_prices)):
        if i == 0:
            predictions.append(actual_prices[i])
        else:
            # Calculate actual price change
            actual_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Create prediction with some skill + noise
            predicted_change = actual_change * skill_level + np.random.normal(0, 0.02)
            pred_price = actual_prices[i-1] * (1 + predicted_change)
            predictions.append(pred_price)
    
    return np.array(predictions)

def demo_basic_roi_analysis():
    """Demonstrate basic ROI analysis"""
    print("\n" + "="*60)
    print("🚀 BASIC ROI ANALYSIS DEMO")
    print("="*60)
    
    # Load data
    actual_prices = load_sample_data()
    predictions = simulate_galformer_predictions(actual_prices, skill_level=0.3)
    
    print(f"Loaded {len(actual_prices)} days of price data")
    print(f"Price range: ${actual_prices.min():,.2f} - ${actual_prices.max():,.2f}")
    
    # Run quick analysis
    results = quick_roi_analysis(
        predictions=predictions,
        actual_prices=actual_prices,
        initial_capital=10000,
        strategy='long_only',
        plot=True,
        save_path='roi_analysis_demo.png'
    )
    
    return results

def demo_strategy_comparison():
    """Demonstrate strategy comparison"""
    print("\n" + "="*60)
    print("📊 STRATEGY COMPARISON DEMO")
    print("="*60)
    
    # Load data
    actual_prices = load_sample_data()
    predictions = simulate_galformer_predictions(actual_prices, skill_level=0.25)
    
    # Compare different strategies
    strategy_results = compare_strategies(
        predictions=predictions,
        actual_prices=actual_prices,
        strategies=['long_only', 'long_short', 'threshold'],
        initial_capital=10000
    )
    
    return strategy_results

def demo_custom_analysis():
    """Demonstrate custom ROI analysis with detailed settings"""
    print("\n" + "="*60)
    print("🔧 CUSTOM ROI ANALYSIS DEMO")
    print("="*60)
    
    # Load data
    actual_prices = load_sample_data()
    predictions = simulate_galformer_predictions(actual_prices, skill_level=0.4)
    
    # Create calculator with custom settings
    calculator = ROICalculator(initial_capital=50000, risk_free_rate=0.03)
    
    # Calculate ROI with custom parameters
    results = calculator.calculate_roi(
        predictions=predictions,
        actual_prices=actual_prices,
        strategy='long_short',
        transaction_cost=0.002  # 0.2% transaction cost
    )
    
    # Print detailed results
    calculator.print_detailed_results(results)
    
    # Create comprehensive plots
    calculator.plot_comprehensive_analysis(
        results, 
        save_path='custom_roi_analysis.png',
        figsize=(18, 14)
    )
    
    # Save results to Excel
    calculator.save_results_to_csv(results, 'custom_roi_results.csv')
    
    return results

def demo_with_real_galformer_predictions():
    """
    Template for using with actual Galformer predictions
    
    This function shows how you would integrate the ROI calculator
    with actual predictions from your trained Galformer model.
    """
    print("\n" + "="*60)
    print("🤖 GALFORMER INTEGRATION TEMPLATE")
    print("="*60)
    
    print("To use with actual Galformer predictions:")
    print("1. Load your trained model")
    print("2. Generate predictions on test data")
    print("3. Load corresponding actual prices")
    print("4. Use roi_calculator as shown below")
    
    example_code = '''
# Example integration code:
from roi_calculator import ROICalculator

# Assuming you have:
# - galformer_predictions: numpy array of predicted prices
# - actual_test_prices: numpy array of actual prices

calculator = ROICalculator(initial_capital=10000)

# Calculate ROI
results = calculator.calculate_roi(
    predictions=galformer_predictions,
    actual_prices=actual_test_prices,
    strategy='long_only'
)

# Print results
calculator.print_detailed_results(results)

# Create plots
calculator.plot_comprehensive_analysis(results, save_path='galformer_roi.png')

# Save to Excel
calculator.save_results_to_csv(results, 'galformer_roi_results.csv')
'''
    
    print(example_code)

if __name__ == "__main__":
    print("🎯 ROI Calculator Demo Suite")
    print("This script demonstrates various ways to use the ROI calculator")
    
    # Run demonstrations
    try:
        # Basic analysis
        basic_results = demo_basic_roi_analysis()
        
        # Strategy comparison
        strategy_results = demo_strategy_comparison()
        
        # Custom analysis
        custom_results = demo_custom_analysis()
        
        # Show integration template
        demo_with_real_galformer_predictions()
        
        print("\n✅ All demos completed successfully!")
        print("Check the generated plots and Excel files for detailed results.")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install numpy pandas matplotlib seaborn openpyxl")
