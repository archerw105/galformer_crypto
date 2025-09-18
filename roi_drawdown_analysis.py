import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import sys

# Import the model classes and data functions from the main Galformer file
from Galformer_pytorch import Transformer, G, positional_encoding, create_look_ahead_mask, get_stock_data, load_data

def load_saved_model(model_path, config_path='config.yaml'):
    """Load a saved Galformer model and its configuration"""
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = Transformer().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load the saved scaler
    scaler = checkpoint.get('scaler', None)
    
    return model, config, device, scaler

def prepare_test_data(config, scaler=None):
    """Prepare test data for evaluation using the same logic as Galformer_pytorch.py"""
    
    # Use the exact same data preprocessing as get_stock_data()
    filename = config['data']['filename']
    df = pd.read_csv(filename, delimiter=',', usecols=['Date','Open','High','Low','Close', 'Adj Close','Volume'])
    df = df.sort_values('Date')
    df.drop(['Date', 'Close'], axis=1, inplace=True)  # Keep 5 dimensions like in training
    
    list_prices = df['Adj Close']
    list1 = list_prices.diff(1).dropna()  # First-order difference
    list_prices = list_prices.tolist()
    list1 = list1.tolist()
    list1 = np.array(list1)
    
    df = df.drop(0, axis=0)
    df['Adj Close'] = list1
    df = df.reset_index(drop=True)
    
    # Use the exact same data loading logic as load_data()
    seq_len = config['sequence']['src_len']
    tgt_len = config['sequence']['tgt_len']
    mulpre = config['sequence']['mulpr_len']
    division_rate1 = config['data']['division_rate1']
    division_rate2 = config['data']['division_rate2']
    
    amount_of_features = 1
    data = df.values
    row1 = round(division_rate1 * data.shape[0])
    row2 = round(division_rate2 * data.shape[0])
    
    # Split data exactly like in load_data()
    train = data[:int(row1), :]
    test = data[int(row2):, :]
    
    # Create a scaler that works only on the Adj Close column (index -2)
    # This ensures denormalization will work correctly with single-column predictions
    adj_close_scaler = preprocessing.StandardScaler()
    train_adj_close = train[:, -2:(-2+1)]  # Extract just the Adj Close column
    test_adj_close = test[:, -2:(-2+1)]
    
    # Fit scaler on training Adj Close data only
    adj_close_scaler.fit(train_adj_close)
    
    # Normalize the full test data using the same logic as training
    if scaler is not None:
        # Use the saved full scaler for consistency with training normalization
        try:
            test_norm = scaler.transform(test)
            print("Using saved scaler for test data normalization")
        except ValueError:
            print("Saved scaler dimension mismatch, using new scaler")
            full_scaler = preprocessing.StandardScaler()
            train_norm = full_scaler.fit_transform(train)
            test_norm = full_scaler.transform(test)
    else:
        print("No scaler found, recreating scaler from training data")
        full_scaler = preprocessing.StandardScaler()
        train_norm = full_scaler.fit_transform(train)
        test_norm = full_scaler.transform(test)
    
    # Use the Adj Close-only scaler for denormalization
    standard_scaler = adj_close_scaler
    
    # Create sequences exactly like in load_data()
    X_test = []
    y_test = []
    test_samples = test_norm.shape[0] - seq_len - mulpre + 1
    
    for i in range(0, test_samples, mulpre):
        X_test.append(test_norm[i:i + seq_len, -2])  # -2 is Adj Close column (same as training)
        y_test.append(test_norm[i + seq_len:i + seq_len + tgt_len, -2])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print("X_test:", X_test[0] if len(X_test) > 0 else "No test samples")
    print("y_test:", y_test[0] if len(y_test) > 0 else "No test samples")
    
    # Reshape for model input exactly like in load_data()
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
    
    return X_test, y_test, standard_scaler, test, row2, seq_len, mulpre, list_prices

def denormalize_predictions(predictions, targets, standard_scaler, test_data, row2, seq_len, mulpre, original_prices, config):
    """Denormalize predictions and targets from standardized differences to actual price differences"""
    
    # Denormalize predictions and targets back to original scale
    # Reshape for inverse_transform (expects 2D array)
    pred_reshaped = predictions.reshape(-1, 1)
    target_reshaped = targets.reshape(-1, 1)
    
    # Denormalize using the scaler
    pred_diffs = standard_scaler.inverse_transform(pred_reshaped).flatten()
    target_diffs = standard_scaler.inverse_transform(target_reshaped).flatten()
    
    print("Raw predictions (first 5):", predictions.flatten()[:5])
    print("Raw targets (first 5):", targets.flatten()[:5])
    print("Denormalized pred_diffs (first 5):", pred_diffs[:5])
    print("Denormalized target_diffs (first 5):", target_diffs[:5])
    print("Pred_diffs mean:", np.mean(pred_diffs), "std:", np.std(pred_diffs))
    print("Target_diffs mean:", np.mean(target_diffs), "std:", np.std(target_diffs))
    
    # Get the starting prices from the original data
    # Start from the beginning of the test segment plus seq_len
    # This aligns reconstructed prices with test predictions
    start_idx = row2 + seq_len
    
    pred_prices = []
    target_prices = []
    
    for i in range(len(pred_diffs)):
        price_idx = start_idx + i
        if price_idx < len(original_prices):
            # Use the actual previous day's price as base
            base_price = original_prices[price_idx - 1] # index: start_idx + i - 1
            
            # Add the predicted/actual difference to get the price
            pred_price = base_price + pred_diffs[i]
            target_price = base_price + target_diffs[i]
            # print("pred diffs, target diffs:", pred_diffs[i], target_diffs[i])
            
            # Debug: print first few values
            if i < 5:
                print(f"i={i}: base_price={base_price:.2f}, pred_diff={pred_diffs[i]:.2f}, pred_price={pred_price:.2f}")
            
            pred_prices.append(pred_price)
            target_prices.append(target_price)
    
    return np.array(pred_prices), np.array(target_prices)

def calculate_roi(predictions, csv_file, seq_len, row2, initial_capital=10000):
    """
    Calculate Return on Investment based on trading strategy using actual CSV prices
    
    Strategy: Buy when prediction > current price, Sell when prediction < current price
    """
    
    # Read actual prices directly from CSV
    df = pd.read_csv(csv_file)
    all_prices = df['Adj Close'].values
    
    # Align with model predictions - skip the first seq_len prices since model needs that history
    # Model predictions start from index seq_len in the original data
    # For test-only evaluation, offset by row2 to match the test segment
    targets = all_prices[row2 + seq_len:]
    
    if len(predictions) > len(targets):
        predictions = predictions[:len(targets)]
    elif len(targets) > len(predictions):
        targets = targets[:len(predictions)]
    
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position
    shares = 0
    trades = []
    portfolio_values = [initial_capital]
    
    print("First 10 actual prices (aligned with predictions):", targets[:10])
    print("First 10 predictions:", predictions[:10])
    
    for i in range(1, len(predictions)):
        current_price = targets[i-1]  # Use actual price as "current" price. start_idx + i - 1
        predicted_price = predictions[i]
        next_actual_price = targets[i]
        # print("current_price:", current_price)
        # print("predicted_price:", predicted_price)
        # print("next_actual_price:", next_actual_price)
        
        # Trading decision based on prediction
        if predicted_price > current_price and position == 0:
            # Buy signal - go long
            shares = capital / current_price
            capital = 0
            position = 1
            trades.append(('BUY', current_price, shares))
            daily_roi = (next_actual_price - current_price) / current_price * 100
            if i < 20:  # Debug first 10 trades
                print(f"BUY (day {i}) | daily ROI: {daily_roi:.2f}% | pred={predicted_price:.2f} > curr={current_price:.2f}, next_actual={next_actual_price:.2f}")
            
        elif predicted_price < current_price and position == 1:
            # Sell signal - close long position
            capital = shares * current_price
            position = 0
            trades.append(('SELL', current_price, shares))
            shares = 0
            if i < 20:  # Debug first 10 trades
                print(f"SELL (day {i}) | pred={predicted_price:.2f} < curr={current_price:.2f}, next_actual={next_actual_price:.2f}")
        
        # Calculate portfolio value
        if position == 1:
            portfolio_value = shares * next_actual_price
        else:
            portfolio_value = capital
            
        portfolio_values.append(portfolio_value)
    
    # Close any remaining position
    if position == 1:
        capital = shares * targets[-1]
        trades.append(('SELL', targets[-1], shares))
    
    final_value = capital if position == 0 else shares * targets[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values,
        'actual_prices': targets
    }

def calculate_buy_hold_roi(prices, initial_capital=10000):
    """Calculate buy and hold return for comparison"""
    if len(prices) < 2:
        return 0
    
    shares = initial_capital / prices[0]
    final_value = shares * prices[-1]
    return (final_value - initial_capital) / initial_capital * 100

def calculate_maximum_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from portfolio values
    
    Maximum drawdown is the largest peak-to-trough decline
    """
    portfolio_values = np.array(portfolio_values)
    
    # Calculate running maximum (peak values)
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdown = (portfolio_values - running_max) / running_max * 100
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Find the indices of the peak and trough
    max_dd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(running_max[:max_dd_idx + 1])
    
    return {
        'max_drawdown_pct': max_drawdown,
        'peak_value': running_max[peak_idx],
        'trough_value': portfolio_values[max_dd_idx],
        'peak_date_idx': peak_idx,
        'trough_date_idx': max_dd_idx,
        'drawdown_series': drawdown
    }

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def add_noise_to_predictions(pred_prices, target_prices, noise_level):
    """Add noise to predictions based on price volatility"""
    price_volatility = np.std(target_prices)
    noise_std = noise_level * price_volatility / 100
    noise = np.random.normal(0, noise_std, len(pred_prices))
    return pred_prices + noise

def plot_results(portfolio_values, predictions, targets, roi_results, drawdown_results, save_path=None):
    """Plot comprehensive results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate buy-and-hold portfolio values
    initial_capital = roi_results['initial_capital']
    shares = initial_capital / targets[0]
    buy_hold_portfolio = [shares * price for price in targets]
    
    # Plot 1: Portfolio Value Over Time
    ax1.plot(portfolio_values, label='Strategy Portfolio', linewidth=2, color='blue')
    ax1.plot(buy_hold_portfolio, label='Buy & Hold BTC', linewidth=2, color='orange', alpha=0.8)
    ax1.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital', alpha=0.7)
    ax1.set_title('Portfolio Value Over Time: Strategy vs Buy & Hold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual Prices
    time_idx = range(len(predictions))
    ax2.plot(time_idx, predictions, label='Predictions', alpha=0.7)
    ax2.plot(time_idx, targets, label='Actual Prices', alpha=0.7)
    ax2.set_title('Price Predictions vs Actual')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Over Time
    ax3.plot(drawdown_results['drawdown_series'], color='red', linewidth=2)
    ax3.fill_between(range(len(drawdown_results['drawdown_series'])), 
                     drawdown_results['drawdown_series'], 0, alpha=0.3, color='red')
    ax3.set_title(f'Drawdown Over Time (Max: {drawdown_results["max_drawdown_pct"]:.2f}%)')
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Returns Distribution
    returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
    ax4.hist(returns, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.3f}%')
    ax4.set_title('Daily Returns Distribution')
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run ROI and drawdown analysis"""
    
    # Configuration
    config_path = 'config_roi.yaml'
    model_path = 'runs/galformer_btc_1day_prediction/galformer_model.pth'
    
    print("Loading saved model and configuration...")
    model, config, device, scaler = load_saved_model(model_path, config_path)
    
    print("Preparing test data...")
    X_test, y_test, data_scaler, test_data, row2, seq_len, mulpre, original_prices = prepare_test_data(config, scaler)
    
    print("Making predictions on test data...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor, training=False) # shape (batch_size, seq_len, d_model)
        predictions = predictions.cpu().numpy()
        print("Raw model predictions shape:", predictions.shape)
        print("Raw model predictions (first 5):", predictions.flatten()[:5])
    
    print("Denormalizing predictions...")
    # Squeeze predictions to remove the last dimension (batch_size, tgt_len, 1) -> (batch_size, tgt_len)
    predictions_squeezed = predictions.squeeze(-1)
    print("Predictions after squeeze:", predictions_squeezed.shape)
    print("y_test shape:", y_test.shape)
    
    pred_prices, target_prices = denormalize_predictions(
        predictions_squeezed, y_test, data_scaler, test_data, row2, seq_len, mulpre, original_prices, config
    )
    
    print("Calculating ROI and trading performance...")
    csv_file = config['data']['filename']
    roi_results = calculate_roi(pred_prices, csv_file, seq_len, row2)
    
    print("Calculating buy-and-hold benchmark...")
    buy_hold_return = calculate_buy_hold_roi(roi_results['actual_prices'])
    
    print("Calculating maximum drawdown...")
    drawdown_results = calculate_maximum_drawdown(roi_results['portfolio_values'])
    
    print("Calculating additional metrics...")
    sharpe_ratio = calculate_sharpe_ratio(roi_results['portfolio_values'])
    
    # Check if noise testing is enabled
    if config.get('noise_test', {}).get('enabled', False):
        noise_level = config['noise_test']['noise_level']
        print(f"Adding {noise_level}% noise to predictions...")
        pred_prices = add_noise_to_predictions(pred_prices, target_prices, noise_level)
        
        # Recalculate ROI with noisy predictions
        print("Recalculating ROI with noisy predictions...")
        roi_results = calculate_roi(pred_prices, csv_file, seq_len, row2)
        drawdown_results = calculate_maximum_drawdown(roi_results['portfolio_values'])
        sharpe_ratio = calculate_sharpe_ratio(roi_results['portfolio_values'])
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE TRADING PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"\n📊 RETURN ON INVESTMENT (ROI)")
    print(f"Initial Capital: ${roi_results['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ${roi_results['final_value']:,.2f}")
    print(f"Total Return: {roi_results['total_return_pct']:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy vs Buy & Hold: {roi_results['total_return_pct'] - buy_hold_return:.2f}% {'✅' if roi_results['total_return_pct'] > buy_hold_return else '❌'}")
    
    print(f"\n📉 MAXIMUM DRAWDOWN ANALYSIS")
    print(f"Maximum Drawdown: {drawdown_results['max_drawdown_pct']:.2f}%")
    print(f"Peak Portfolio Value: ${drawdown_results['peak_value']:,.2f}")
    print(f"Trough Portfolio Value: ${drawdown_results['trough_value']:,.2f}")
    print(f"Peak to Trough Loss: ${drawdown_results['peak_value'] - drawdown_results['trough_value']:,.2f}")
    
    print(f"\n📈 ADDITIONAL METRICS")
    print(f"Number of Trades: {roi_results['num_trades']}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    if len(roi_results['portfolio_values']) > 1:
        returns = np.diff(roi_results['portfolio_values']) / roi_results['portfolio_values'][:-1]
        win_rate = np.sum(returns > 0) / len(returns) * 100
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Daily Return: {np.mean(returns) * 100:.3f}%")
        print(f"Volatility (Daily): {np.std(returns) * 100:.3f}%")
    
    print(f"\n🔄 RECENT TRADES (Last 5)")
    for trade in roi_results['trades'][-5:]:
        action, price, shares = trade
        print(f"  {action}: {shares:.2f} shares @ ${price:.2f}")
    
    # Show noise test status
    if config.get('noise_test', {}).get('enabled', False):
        print(f"\n🧪 NOISE TESTING")
        print(f"Noise level applied: {config['noise_test']['noise_level']}% of price volatility")
        print(f"Results above reflect performance with noisy predictions")
    
    # Create visualization
    print(f"\nGenerating performance visualization...")
    if config.get('noise_test', {}).get('enabled', False):
        noise_level = config['noise_test']['noise_level']
        plot_save_path = f"runs/{config['experiment']['name']}/roi_drawdown_analysis_noise_{noise_level}pct.png"
    else:
        plot_save_path = f"runs/{config['experiment']['name']}/roi_drawdown_analysis.png"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    plot_results(
        roi_results['portfolio_values'], 
        pred_prices, 
        roi_results['actual_prices'], 
        roi_results, 
        drawdown_results, 
        save_path=plot_save_path
    )
    
    # Save detailed results to CSV
    if config.get('noise_test', {}).get('enabled', False):
        noise_level = config['noise_test']['noise_level']
        results_save_path = f"runs/{config['experiment']['name']}/roi_drawdown_results_noise_{noise_level}pct.csv"
    else:
        results_save_path = f"runs/{config['experiment']['name']}/roi_drawdown_results.csv"
    
    results_df = pd.DataFrame({
        'Portfolio_Value': roi_results['portfolio_values'],
        'Drawdown_Pct': list(drawdown_results['drawdown_series']) + [drawdown_results['drawdown_series'][-1]] * (len(roi_results['portfolio_values']) - len(drawdown_results['drawdown_series']))
    })
    
    # Add summary statistics
    noise_status = f"Enabled ({config['noise_test']['noise_level']}%)" if config.get('noise_test', {}).get('enabled', False) else "Disabled"
    summary_stats = {
        'Metric': ['Total_Return_Pct', 'Buy_Hold_Return_Pct', 'Max_Drawdown_Pct', 'Sharpe_Ratio', 'Num_Trades', 'Noise_Testing'],
        'Value': [roi_results['total_return_pct'], buy_hold_return, drawdown_results['max_drawdown_pct'], sharpe_ratio, roi_results['num_trades'], noise_status]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    with pd.ExcelWriter(results_save_path.replace('.csv', '.xlsx'), engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Portfolio_Performance', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"Detailed results saved to {results_save_path.replace('.csv', '.xlsx')}")
    print("\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
