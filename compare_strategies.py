#!/usr/bin/env python3
"""
Minimal script to compare Long-Only vs Long-Short trading strategies
using the enhanced calculate_roi function with shorting capability.
"""

import numpy as np
import pandas as pd

def calculate_roi(predictions, actual_prices, initial_capital=10000, allow_shorting=True):
    """
    Calculate ROI based on trading strategy using predictions
    
    Strategy: 
    - Buy (long) when prediction > current price
    - Sell (short) when prediction < current price (if shorting allowed)
    - Close positions when prediction reverses
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position, -1 = short position
    shares = 0
    trades = []
    portfolio_values = [initial_capital]
    
    for i in range(1, len(predictions)):
        current_price = actual_prices[i-1]
        predicted_price = predictions[i]
        next_actual_price = actual_prices[i]
        
        # Trading decision based on prediction
        if predicted_price > current_price:
            if position == 0:
                # Buy signal - go long
                shares = capital / current_price
                capital = 0
                position = 1
                trades.append(('BUY', current_price, shares))
            elif position == -1 and allow_shorting:
                # Close short position and go long
                capital = capital - (shares * current_price)
                trades.append(('COVER', current_price, shares))
                if capital > 0:
                    shares = capital / current_price
                    capital = 0
                    position = 1
                    trades.append(('BUY', current_price, shares))
                else:
                    shares = 0
                    position = 0
                    
        elif predicted_price < current_price:
            if position == 0 and allow_shorting:
                # Short signal - go short
                shares = initial_capital / current_price
                capital = initial_capital + (shares * current_price)
                position = -1
                trades.append(('SHORT', current_price, shares))
            elif position == 1:
                # Close long position
                capital = shares * current_price
                position = 0
                trades.append(('SELL', current_price, shares))
                shares = 0
                # If shorting allowed, go short after closing long
                if allow_shorting:
                    shares = capital / current_price
                    capital = capital + (shares * current_price)
                    position = -1
                    trades.append(('SHORT', current_price, shares))
        
        # Calculate portfolio value
        if position == 1:  # Long position
            portfolio_value = shares * next_actual_price
        elif position == -1:  # Short position
            portfolio_value = capital - (shares * next_actual_price)
        else:  # No position
            portfolio_value = capital
            
        portfolio_values.append(portfolio_value)
    
    # Close any remaining position
    if position == 1:
        capital = shares * actual_prices[-1]
        trades.append(('SELL', actual_prices[-1], shares))
    elif position == -1:
        capital = capital - (shares * actual_prices[-1])
        trades.append(('COVER', actual_prices[-1], shares))
    
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Calculate buy and hold return for comparison
    buy_hold_shares = initial_capital / actual_prices[0]
    buy_hold_final = buy_hold_shares * actual_prices[-1]
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def generate_sample_data(n_days=100):
    """Generate sample price data and predictions for demonstration"""
    np.random.seed(42)
    
    # Generate realistic price movement
    initial_price = 50000  # Starting price (like BTC)
    prices = [initial_price]
    
    for i in range(n_days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% daily trend, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    prices = np.array(prices)
    
    # Generate predictions with some accuracy but noise
    predictions = []
    for i in range(len(prices)-1):
        # Prediction has some skill but isn't perfect
        true_future = prices[i+1]
        noise = np.random.normal(0, 0.01)  # 1% prediction noise
        pred = true_future * (1 + noise)
        predictions.append(pred)
    
    return predictions, prices[:-1]  # Remove last price to match prediction length

def main():
    print("🚀 Comparing Long-Only vs Long-Short Trading Strategies")
    print("=" * 60)
    
    # Generate sample data
    predictions, actual_prices = generate_sample_data(200)
    
    print(f"📊 Dataset: {len(predictions)} trading days")
    print(f"💰 Initial Capital: $10,000")
    print(f"📈 Price Range: ${actual_prices.min():,.0f} - ${actual_prices.max():,.0f}")
    print()
    
    # Run Long-Only Strategy
    print("📈 LONG-ONLY STRATEGY (No Shorting)")
    print("-" * 40)
    roi_long_only = calculate_roi(predictions, actual_prices, 
                                 initial_capital=10000, allow_shorting=False)
    
    print(f"Final Value: ${roi_long_only['final_value']:,.2f}")
    print(f"Total Return: {roi_long_only['total_return_pct']:+.2f}%")
    print(f"Buy & Hold: {roi_long_only['buy_hold_return_pct']:+.2f}%")
    print(f"Number of Trades: {roi_long_only['num_trades']}")
    print()
    
    # Run Long-Short Strategy
    print("📊 LONG-SHORT STRATEGY (With Shorting)")
    print("-" * 40)
    roi_long_short = calculate_roi(predictions, actual_prices, 
                                  initial_capital=10000, allow_shorting=True)
    
    print(f"Final Value: ${roi_long_short['final_value']:,.2f}")
    print(f"Total Return: {roi_long_short['total_return_pct']:+.2f}%")
    print(f"Buy & Hold: {roi_long_short['buy_hold_return_pct']:+.2f}%")
    print(f"Number of Trades: {roi_long_short['num_trades']}")
    print()
    
    # Compare Strategies
    print("🎯 STRATEGY COMPARISON")
    print("-" * 40)
    improvement = roi_long_short['total_return_pct'] - roi_long_only['total_return_pct']
    print(f"Long-Short vs Long-Only: {improvement:+.2f}% difference")
    
    if improvement > 0:
        print("✅ Long-Short strategy outperformed!")
    elif improvement < 0:
        print("❌ Long-Only strategy was better")
    else:
        print("🤝 Both strategies performed equally")
    
    # Show recent trades for each strategy
    print(f"\n📋 Recent Long-Only Trades (last 5):")
    for trade in roi_long_only['trades'][-5:]:
        action, price, shares = trade
        print(f"  {action}: {shares:.4f} shares @ ${price:,.2f}")
    
    print(f"\n📋 Recent Long-Short Trades (last 5):")
    for trade in roi_long_short['trades'][-5:]:
        action, price, shares = trade
        print(f"  {action}: {shares:.4f} shares @ ${price:,.2f}")

if __name__ == "__main__":
    main()
