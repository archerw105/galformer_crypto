"""
ROI Calculator Module for Financial Time Series Predictions

This module provides comprehensive ROI calculation and visualization functionality
for trading strategies based on model predictions. It includes:
- ROI calculation with customizable trading strategies
- Maximum drawdown analysis
- Sharpe ratio calculation
- Comprehensive plotting and visualization
- Buy-and-hold benchmark comparison

Usage:
    from roi_calculator import ROICalculator
    
    calculator = ROICalculator(initial_capital=10000)
    results = calculator.calculate_roi(predictions, actual_prices)
    calculator.plot_comprehensive_analysis(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ROICalculator:
    """
    A comprehensive ROI calculator for financial time series predictions
    """
    
    def __init__(self, initial_capital: float = 10000, risk_free_rate: float = 0.02):
        """
        Initialize the ROI calculator
        
        Args:
            initial_capital: Starting capital for trading simulation
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
    def calculate_roi(self, 
                     predictions: np.ndarray, 
                     actual_prices: np.ndarray,
                     strategy: str = 'long_only',
                     transaction_cost: float = 0.001) -> Dict:
        """
        Calculate ROI based on predictions and actual prices
        
        Args:
            predictions: Array of predicted prices
            actual_prices: Array of actual prices
            strategy: Trading strategy ('long_only', 'long_short', 'threshold')
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            
        Returns:
            Dictionary containing ROI results and trading statistics
        """
        
        # Ensure arrays are the same length
        min_len = min(len(predictions), len(actual_prices))
        predictions = predictions[:min_len]
        actual_prices = actual_prices[:min_len]
        
        if len(predictions) < 2:
            raise ValueError("Need at least 2 data points for ROI calculation")
        
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        shares = 0
        trades = []
        portfolio_values = [self.initial_capital]
        positions = [0]  # Track position over time
        
        for i in range(1, len(predictions)):
            current_price = actual_prices[i-1]
            predicted_price = predictions[i]
            next_actual_price = actual_prices[i]
            
            # Generate trading signals based on strategy
            signal = self._generate_signal(predicted_price, current_price, strategy)
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Close short position first
                    capital = capital - shares * current_price * (1 + transaction_cost)
                    trades.append(('COVER', current_price, abs(shares), capital))
                
                # Open long position
                shares = capital / (current_price * (1 + transaction_cost))
                capital = 0
                position = 1
                trades.append(('BUY', current_price, shares, capital))
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Close long position first
                    capital = shares * current_price * (1 - transaction_cost)
                    trades.append(('SELL', current_price, shares, capital))
                
                if strategy == 'long_short':
                    # Open short position
                    shares = capital / (current_price * (1 + transaction_cost))
                    capital = capital - shares * current_price * (1 + transaction_cost)
                    position = -1
                    trades.append(('SHORT', current_price, shares, capital))
                else:
                    shares = 0
                    position = 0
            
            # Calculate portfolio value
            if position == 1:  # Long position
                portfolio_value = shares * next_actual_price
            elif position == -1:  # Short position
                portfolio_value = capital + shares * (2 * trades[-1][1] - next_actual_price)
            else:  # No position
                portfolio_value = capital
                
            portfolio_values.append(portfolio_value)
            positions.append(position)
        
        # Close any remaining position
        if position != 0:
            final_price = actual_prices[-1]
            if position == 1:
                capital = shares * final_price * (1 - transaction_cost)
                trades.append(('SELL', final_price, shares, capital))
            elif position == -1:
                capital = capital - shares * final_price * (1 + transaction_cost)
                trades.append(('COVER', final_price, shares, capital))
        
        final_value = capital if position == 0 else portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values,
            'positions': positions,
            'actual_prices': actual_prices,
            'predictions': predictions,
            'strategy': strategy,
            'transaction_cost': transaction_cost
        }
    
    def _generate_signal(self, predicted_price: float, current_price: float, strategy: str) -> int:
        """
        Generate trading signal based on prediction and strategy
        
        Returns:
            1: Buy signal, -1: Sell signal, 0: Hold signal
        """
        price_diff_pct = (predicted_price - current_price) / current_price * 100
        
        if strategy == 'long_only':
            if price_diff_pct > 0.5:  # Buy if prediction > 0.5% higher
                return 1
            elif price_diff_pct < -0.5:  # Sell if prediction > 0.5% lower
                return -1
                
        elif strategy == 'long_short':
            if price_diff_pct > 0.5:
                return 1
            elif price_diff_pct < -0.5:
                return -1
                
        elif strategy == 'threshold':
            if price_diff_pct > 2.0:  # Higher threshold for trades
                return 1
            elif price_diff_pct < -2.0:
                return -1
        
        return 0
    
    def calculate_buy_hold_roi(self, prices: np.ndarray) -> float:
        """Calculate buy and hold return for comparison"""
        if len(prices) < 2:
            return 0
        
        shares = self.initial_capital / prices[0]
        final_value = shares * prices[-1]
        return (final_value - self.initial_capital) / self.initial_capital * 100
    
    def calculate_maximum_drawdown(self, portfolio_values: List[float]) -> Dict:
        """
        Calculate maximum drawdown from portfolio values
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
    
    def calculate_sharpe_ratio(self, portfolio_values: List[float]) -> float:
        """Calculate Sharpe ratio"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_additional_metrics(self, roi_results: Dict) -> Dict:
        """Calculate additional performance metrics"""
        portfolio_values = roi_results['portfolio_values']
        
        if len(portfolio_values) < 2:
            return {}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {
            'win_rate': np.sum(returns > 0) / len(returns) * 100,
            'avg_daily_return': np.mean(returns) * 100,
            'volatility': np.std(returns) * 100,
            'max_single_day_gain': np.max(returns) * 100,
            'max_single_day_loss': np.min(returns) * 100,
            'total_days': len(portfolio_values) - 1,
            'profitable_days': np.sum(returns > 0),
            'losing_days': np.sum(returns < 0)
        }
        
        return metrics
    
    def plot_comprehensive_analysis(self, 
                                  roi_results: Dict, 
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True,
                                  figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create comprehensive ROI analysis plots
        """
        
        # Calculate additional metrics
        buy_hold_return = self.calculate_buy_hold_roi(roi_results['actual_prices'])
        drawdown_results = self.calculate_maximum_drawdown(roi_results['portfolio_values'])
        sharpe_ratio = self.calculate_sharpe_ratio(roi_results['portfolio_values'])
        additional_metrics = self.calculate_additional_metrics(roi_results)
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Portfolio Value Over Time (Large plot)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Calculate buy-and-hold portfolio values
        initial_capital = roi_results['initial_capital']
        shares = initial_capital / roi_results['actual_prices'][0]
        buy_hold_portfolio = [shares * price for price in roi_results['actual_prices']]
        
        time_idx = range(len(roi_results['portfolio_values']))
        ax1.plot(time_idx, roi_results['portfolio_values'], 
                label=f'Strategy ({roi_results["strategy"]})', linewidth=2, color='#2E86AB')
        ax1.plot(time_idx, buy_hold_portfolio, 
                label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.8)
        ax1.axhline(y=initial_capital, color='#F18F01', linestyle='--', 
                   label='Initial Capital', alpha=0.7)
        
        ax1.set_title('Portfolio Performance: Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add performance annotations
        final_return = roi_results['total_return_pct']
        ax1.text(0.02, 0.98, f'Strategy Return: {final_return:.2f}%\nBuy & Hold: {buy_hold_return:.2f}%', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Predictions vs Actual Prices
        ax2 = fig.add_subplot(gs[1, 0])
        time_idx = range(len(roi_results['predictions']))
        ax2.plot(time_idx, roi_results['predictions'], label='Predictions', alpha=0.7, color='#C73E1D')
        ax2.plot(time_idx, roi_results['actual_prices'], label='Actual', alpha=0.7, color='#2E86AB')
        ax2.set_title('Predictions vs Actual Prices')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown Over Time
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(drawdown_results['drawdown_series'], color='#A23B72', linewidth=2)
        ax3.fill_between(range(len(drawdown_results['drawdown_series'])), 
                        drawdown_results['drawdown_series'], 0, alpha=0.3, color='#A23B72')
        ax3.set_title(f'Drawdown (Max: {drawdown_results["max_drawdown_pct"]:.2f}%)')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        if len(roi_results['portfolio_values']) > 1:
            returns = np.diff(roi_results['portfolio_values']) / roi_results['portfolio_values'][:-1] * 100
            ax4.hist(returns, bins=30, alpha=0.7, edgecolor='black', color='#F18F01')
            ax4.axvline(x=np.mean(returns), color='#C73E1D', linestyle='--', 
                       label=f'Mean: {np.mean(returns):.3f}%')
            ax4.set_title('Daily Returns Distribution')
            ax4.set_xlabel('Daily Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Position Over Time
        ax5 = fig.add_subplot(gs[2, 0])
        positions = roi_results.get('positions', [0] * len(roi_results['portfolio_values']))
        ax5.plot(positions, linewidth=2, color='#2E86AB')
        ax5.fill_between(range(len(positions)), positions, 0, alpha=0.3, color='#2E86AB')
        ax5.set_title('Position Over Time')
        ax5.set_xlabel('Time Period')
        ax5.set_ylabel('Position (1=Long, -1=Short, 0=Cash)')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-1.5, 1.5)
        
        # Plot 6: Cumulative Returns Comparison
        ax6 = fig.add_subplot(gs[2, 1])
        strategy_returns = [(v - initial_capital) / initial_capital * 100 for v in roi_results['portfolio_values']]
        buy_hold_returns = [(v - initial_capital) / initial_capital * 100 for v in buy_hold_portfolio]
        
        ax6.plot(strategy_returns, label='Strategy', linewidth=2, color='#2E86AB')
        ax6.plot(buy_hold_returns, label='Buy & Hold', linewidth=2, color='#A23B72')
        ax6.set_title('Cumulative Returns (%)')
        ax6.set_xlabel('Time Period')
        ax6.set_ylabel('Return (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Performance Metrics Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        metrics_text = f"""
Performance Metrics:
━━━━━━━━━━━━━━━━━━━━
Total Return: {roi_results['total_return_pct']:.2f}%
Buy & Hold: {buy_hold_return:.2f}%
Max Drawdown: {drawdown_results['max_drawdown_pct']:.2f}%
Sharpe Ratio: {sharpe_ratio:.3f}
Number of Trades: {roi_results['num_trades']}
"""
        
        if additional_metrics:
            metrics_text += f"""
Win Rate: {additional_metrics['win_rate']:.1f}%
Avg Daily Return: {additional_metrics['avg_daily_return']:.3f}%
Volatility: {additional_metrics['volatility']:.3f}%
Max Daily Gain: {additional_metrics['max_single_day_gain']:.2f}%
Max Daily Loss: {additional_metrics['max_single_day_loss']:.2f}%
"""
        
        ax7.text(0.05, 0.95, metrics_text, transform=ax7.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Comprehensive ROI Analysis Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def print_detailed_results(self, roi_results: Dict) -> None:
        """Print detailed ROI analysis results"""
        
        buy_hold_return = self.calculate_buy_hold_roi(roi_results['actual_prices'])
        drawdown_results = self.calculate_maximum_drawdown(roi_results['portfolio_values'])
        sharpe_ratio = self.calculate_sharpe_ratio(roi_results['portfolio_values'])
        additional_metrics = self.calculate_additional_metrics(roi_results)
        
        print("\n" + "="*70)
        print("🚀 COMPREHENSIVE ROI ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\n📊 STRATEGY PERFORMANCE")
        print(f"Strategy Type: {roi_results['strategy'].upper()}")
        print(f"Initial Capital: ${roi_results['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${roi_results['final_value']:,.2f}")
        print(f"Total Return: {roi_results['total_return_pct']:.2f}%")
        print(f"Transaction Cost: {roi_results['transaction_cost']*100:.2f}%")
        
        print(f"\n📈 BENCHMARK COMPARISON")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        excess_return = roi_results['total_return_pct'] - buy_hold_return
        print(f"Excess Return: {excess_return:.2f}% {'✅' if excess_return > 0 else '❌'}")
        
        print(f"\n📉 RISK METRICS")
        print(f"Maximum Drawdown: {drawdown_results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Peak Portfolio Value: ${drawdown_results['peak_value']:,.2f}")
        print(f"Trough Portfolio Value: ${drawdown_results['trough_value']:,.2f}")
        
        print(f"\n🔄 TRADING ACTIVITY")
        print(f"Total Trades: {roi_results['num_trades']}")
        
        if additional_metrics:
            print(f"Win Rate: {additional_metrics['win_rate']:.1f}%")
            print(f"Profitable Days: {additional_metrics['profitable_days']}")
            print(f"Losing Days: {additional_metrics['losing_days']}")
            print(f"Average Daily Return: {additional_metrics['avg_daily_return']:.3f}%")
            print(f"Daily Volatility: {additional_metrics['volatility']:.3f}%")
            print(f"Best Single Day: +{additional_metrics['max_single_day_gain']:.2f}%")
            print(f"Worst Single Day: {additional_metrics['max_single_day_loss']:.2f}%")
        
        print(f"\n💼 RECENT TRADES (Last 5)")
        recent_trades = roi_results['trades'][-5:] if len(roi_results['trades']) >= 5 else roi_results['trades']
        for i, trade in enumerate(recent_trades, 1):
            action, price, shares = trade[:3]
            print(f"  {i}. {action}: {shares:.4f} shares @ ${price:.2f}")
        
        print("\n✅ Analysis completed successfully!")
    
    def save_results_to_csv(self, roi_results: Dict, filename: str) -> None:
        """Save detailed results to CSV file"""
        
        # Create main results DataFrame
        results_df = pd.DataFrame({
            'Portfolio_Value': roi_results['portfolio_values'],
            'Actual_Price': roi_results['actual_prices'],
            'Prediction': roi_results['predictions'][:len(roi_results['portfolio_values'])],
            'Position': roi_results.get('positions', [0] * len(roi_results['portfolio_values']))
        })
        
        # Calculate additional columns
        if len(roi_results['portfolio_values']) > 1:
            returns = [0] + list(np.diff(roi_results['portfolio_values']) / roi_results['portfolio_values'][:-1] * 100)
            results_df['Daily_Return_Pct'] = returns
        
        # Calculate drawdown
        drawdown_results = self.calculate_maximum_drawdown(roi_results['portfolio_values'])
        drawdown_series = list(drawdown_results['drawdown_series'])
        if len(drawdown_series) < len(roi_results['portfolio_values']):
            drawdown_series.extend([drawdown_series[-1]] * (len(roi_results['portfolio_values']) - len(drawdown_series)))
        results_df['Drawdown_Pct'] = drawdown_series
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename.replace('.csv', '.xlsx'), engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Portfolio_Performance', index=False)
            
            # Create summary statistics sheet
            buy_hold_return = self.calculate_buy_hold_roi(roi_results['actual_prices'])
            sharpe_ratio = self.calculate_sharpe_ratio(roi_results['portfolio_values'])
            additional_metrics = self.calculate_additional_metrics(roi_results)
            
            summary_data = {
                'Metric': ['Initial_Capital', 'Final_Value', 'Total_Return_Pct', 'Buy_Hold_Return_Pct', 
                          'Excess_Return_Pct', 'Max_Drawdown_Pct', 'Sharpe_Ratio', 'Num_Trades',
                          'Strategy', 'Transaction_Cost_Pct'],
                'Value': [roi_results['initial_capital'], roi_results['final_value'], 
                         roi_results['total_return_pct'], buy_hold_return,
                         roi_results['total_return_pct'] - buy_hold_return,
                         drawdown_results['max_drawdown_pct'], sharpe_ratio, roi_results['num_trades'],
                         roi_results['strategy'], roi_results['transaction_cost'] * 100]
            }
            
            if additional_metrics:
                for key, value in additional_metrics.items():
                    summary_data['Metric'].append(key)
                    summary_data['Value'].append(value)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Create trades sheet
            if roi_results['trades']:
                trades_df = pd.DataFrame(roi_results['trades'], 
                                       columns=['Action', 'Price', 'Shares', 'Capital'])
                trades_df.to_excel(writer, sheet_name='Trading_History', index=False)
        
        print(f"Detailed results saved to {filename.replace('.csv', '.xlsx')}")


# Convenience functions for quick usage
def quick_roi_analysis(predictions: np.ndarray, 
                      actual_prices: np.ndarray,
                      initial_capital: float = 10000,
                      strategy: str = 'long_only',
                      plot: bool = True,
                      save_path: Optional[str] = None) -> Dict:
    """
    Quick ROI analysis with default settings
    
    Args:
        predictions: Array of predicted prices
        actual_prices: Array of actual prices
        initial_capital: Starting capital
        strategy: Trading strategy
        plot: Whether to show plots
        save_path: Path to save results
    
    Returns:
        ROI results dictionary
    """
    calculator = ROICalculator(initial_capital=initial_capital)
    results = calculator.calculate_roi(predictions, actual_prices, strategy=strategy)
    
    if plot:
        calculator.plot_comprehensive_analysis(results, save_path=save_path)
    
    calculator.print_detailed_results(results)
    
    return results


def compare_strategies(predictions: np.ndarray,
                      actual_prices: np.ndarray,
                      strategies: List[str] = ['long_only', 'long_short', 'threshold'],
                      initial_capital: float = 10000) -> Dict:
    """
    Compare multiple trading strategies
    
    Returns:
        Dictionary with results for each strategy
    """
    calculator = ROICalculator(initial_capital=initial_capital)
    results = {}
    
    print("\n" + "="*70)
    print("📊 STRATEGY COMPARISON ANALYSIS")
    print("="*70)
    
    for strategy in strategies:
        print(f"\n🔄 Analyzing {strategy.upper()} strategy...")
        strategy_results = calculator.calculate_roi(predictions, actual_prices, strategy=strategy)
        results[strategy] = strategy_results
        
        buy_hold = calculator.calculate_buy_hold_roi(actual_prices)
        print(f"  Return: {strategy_results['total_return_pct']:.2f}% | "
              f"Trades: {strategy_results['num_trades']} | "
              f"vs Buy&Hold: {strategy_results['total_return_pct'] - buy_hold:.2f}%")
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['total_return_pct'])
    print(f"\n🏆 Best Strategy: {best_strategy.upper()} "
          f"({results[best_strategy]['total_return_pct']:.2f}% return)")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("ROI Calculator Module - Example Usage")
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    base_price = 100
    actual_prices = [base_price]
    
    for i in range(n_days - 1):
        change = np.random.normal(0, 2)  # 2% daily volatility
        new_price = actual_prices[-1] * (1 + change/100)
        actual_prices.append(max(new_price, 1))  # Prevent negative prices
    
    # Generate predictions with some skill
    predictions = []
    for i in range(len(actual_prices)):
        if i == 0:
            predictions.append(actual_prices[i])
        else:
            # Add some predictive skill with noise
            true_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            predicted_change = true_change * 0.3 + np.random.normal(0, 0.01)  # 30% skill
            pred_price = actual_prices[i-1] * (1 + predicted_change)
            predictions.append(pred_price)
    
    actual_prices = np.array(actual_prices)
    predictions = np.array(predictions)
    
    print(f"\nGenerated {len(actual_prices)} days of sample data")
    print(f"Price range: ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")
    
    # Run quick analysis
    results = quick_roi_analysis(predictions, actual_prices, 
                               initial_capital=10000, 
                               strategy='long_only',
                               plot=True)
