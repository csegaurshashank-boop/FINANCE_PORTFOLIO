"""
Backtesting module for rolling window evaluation and performance comparison
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass

from engine import PortfolioEngine
from data_loader import DataLoader
from config import PortfolioConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestMetrics:
    """Container for backtest metrics"""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

class BacktestEngine:
    """
    Rolling window backtesting engine with performance comparison
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.results = {}
        
    async def run_rolling_backtest(self, window_size: int = 252, 
                                   step_size: int = 63) -> Dict:
        """
        Run rolling window backtest
        
        Args:
            window_size: Training window size in days
            step_size: Step size between windows in days
        """
        logger.info(f"Starting rolling backtest with window={window_size}, step={step_size}")
        
        # Load full dataset
        prices, volumes = await self.data_loader.download_data(
            self.config.assets,
            self.config.start_date,
            self.config.end_date,
            use_sample_if_fails=True
        )
        
        if prices.empty:
            raise ValueError("Failed to load data")
        
        # Adjust window size based on available data
        total_days = len(prices)
        logger.info(f"Total days of data: {total_days}")
        
        # If we have less than 2 years of data, adjust window sizes
        if total_days < 252:
            window_size = max(20, total_days // 3)  # Use 1/3 of data for training
            step_size = max(5, window_size // 4)    # Step size as 1/4 of window
            logger.info(f"Adjusted window_size to {window_size}, step_size to {step_size} for small dataset")
        
        # Generate windows
        windows = self.generate_windows(prices.index, window_size, step_size)
        logger.info(f"Generated {len(windows)} windows for backtesting")
        
        if len(windows) == 0:
            # If no windows, do a simple train/test split
            logger.warning("No rolling windows generated, using simple train/test split")
            return await self.run_simple_backtest(prices, volumes)
        
        results_with_risk = []
        results_without_risk = []
        dates = []
        
        for i, (train_end, test_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Use data up to test_end (expanding window approach for training)
            # This ensures the engine has access to sufficient history for indicators
            window_prices = prices.loc[:test_end]
            window_volumes = volumes.loc[:test_end]
            
            test_start = train_end + timedelta(days=1)
            # If test_start not in index (weekend/holiday), find next available day
            # Actually, calculate test performance from train_end date
            
            # Check if we have enough test data
            test_data_slice = prices.loc[train_end:test_end]
            if len(test_data_slice) < 5:  # Skip very short test periods
                logger.warning(f"Test period too short ({len(test_data_slice)} days), skipping")
                continue
            
            # Run with risk management
            with_risk_result = await self.run_window_backtest_with_engine(
                window_prices, window_volumes, train_end, test_end, enable_risk=True
            )
            results_with_risk.append(with_risk_result)
            
            # Run without risk management
            without_risk_result = await self.run_window_backtest_with_engine(
                window_prices, window_volumes, train_end, test_end, enable_risk=False
            )
            results_without_risk.append(without_risk_result)
            
            dates.append(test_end)
            
            logger.info(f"Completed window ending {test_end.strftime('%Y-%m-%d')}")
        
        # Calculate metrics
        metrics_with_risk = self.calculate_window_metrics(results_with_risk) if results_with_risk else {}
        metrics_without_risk = self.calculate_window_metrics(results_without_risk) if results_without_risk else {}
        
        # Compare performance
        comparison = self.compare_performance(
            metrics_with_risk, metrics_without_risk
        )
        
        return {
            "metrics_with_risk": metrics_with_risk,
            "metrics_without_risk": metrics_without_risk,
            "comparison": comparison,
            "rolling_values": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "with_risk": results_with_risk,
                "without_risk": results_without_risk
            }
        }
    
    async def run_simple_backtest(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Dict:
        """Run a simple train/test split backtest when rolling windows aren't possible"""
        logger.info("Running simple train/test split backtest")
        
        # Use 70% for training, 30% for testing
        split_idx = int(len(prices) * 0.7)
        train_end = prices.index[split_idx]
        test_end = prices.index[-1]
        
        logger.info(f"Train end: {train_end}, Test end: {test_end}")
        
        # Run with risk management
        with_risk_result = await self.run_window_backtest_with_engine(
            prices, volumes, train_end, test_end, enable_risk=True
        )
        
        # Run without risk management
        without_risk_result = await self.run_window_backtest_with_engine(
            prices, volumes, train_end, test_end, enable_risk=False
        )
        
        # Calculate metrics
        metrics_with_risk = self.calculate_single_metrics(with_risk_result)
        metrics_without_risk = self.calculate_single_metrics(without_risk_result)
        
        # Compare performance
        comparison = self.compare_performance(metrics_with_risk, metrics_without_risk)
        
        # Convert all numpy types to Python native types
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime("%Y-%m-%d")
            else:
                return obj
        
        return {
            "metrics_with_risk": convert_to_python(metrics_with_risk),
            "metrics_without_risk": convert_to_python(metrics_without_risk),
            "comparison": convert_to_python(comparison),
            "rolling_values": {
                "dates": [test_end.strftime("%Y-%m-%d")],
                "with_risk": [convert_to_python(with_risk_result)],
                "without_risk": [convert_to_python(without_risk_result)]
            }
        }
    
    def generate_windows(self, dates: pd.DatetimeIndex, 
                        window_size: int, 
                        step_size: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate rolling windows"""
        windows = []
        
        if len(dates) < window_size + step_size:
            logger.warning(f"Not enough data for rolling windows: {len(dates)} < {window_size + step_size}")
            return windows
        
        for i in range(window_size, len(dates) - 1, step_size):
            train_end = dates[i]
            test_end = dates[min(i + step_size, len(dates) - 1)]
            windows.append((train_end, test_end))
        
        return windows
    
    async def run_window_backtest_with_engine(self, prices: pd.DataFrame, 
                                            volumes: pd.DataFrame,
                                            train_end_date: pd.Timestamp,
                                            test_end_date: pd.Timestamp,
                                            enable_risk: bool) -> Dict:
        """
        Run backtest using the actual PortfolioEngine logic.
        This runs the engine on the full data provided (up to test_end),
        but only calculates performance for the period [train_end_date, test_end_date].
        """
        
        # Create config for this run
        window_config = PortfolioConfig(
            assets=self.config.assets,
            start_date=prices.index[0].strftime("%Y-%m-%d"),
            end_date=test_end_date.strftime("%Y-%m-%d"),
            initial_capital=self.config.initial_capital,
            risk_profile=self.config.risk_profile,
            rebalance_frequency=self.config.rebalance_frequency,
            enable_risk_management=enable_risk
        )
        
        # Initialize engine
        engine = PortfolioEngine(window_config)
        
        # Run engine with provided data
        # Note: We give it data up to test_end_date so it can simulate the test period
        results = await engine.run_with_data(prices, volumes)
        
        # Extract results for the test period
        portfolio_values = results["portfolio_values"]
        dates = prices.index
        
        # Find indices for test period
        # We start tracking return from train_end_date
        try:
            start_idx = dates.get_loc(train_end_date)
            end_idx = dates.get_loc(test_end_date)
        except KeyError:
            # Fallback for approximate indices
            start_idx = dates.searchsorted(train_end_date)
            end_idx = dates.searchsorted(test_end_date)
            
        # Ensure indices are within bounds
        start_idx = min(max(0, start_idx), len(portfolio_values)-1)
        end_idx = min(max(0, end_idx), len(portfolio_values)-1)
        
        # Calculate performance over this specific period
        start_value = portfolio_values[start_idx]
        end_value = portfolio_values[end_idx]
        
        # Get daily values for this period for metrics
        test_period_values = portfolio_values[start_idx:end_idx+1]
        
        # Calculate returns
        if len(test_period_values) > 1:
            returns_series = pd.Series(test_period_values).pct_change().dropna()
            returns_list = [float(x) for x in returns_series.tolist()]
        else:
            returns_list = []
            
        # Calculate period return
        period_return = (end_value / start_value) - 1 if start_value > 0 else 0
        
        # Use initial capital as base for "final_value" to allow aggregation
        # We simulate what $InitialCapital would become over this period
        simulated_final_value = self.config.initial_capital * (1 + period_return)
        
        return {
            "final_value": float(simulated_final_value), # Normalized to initial capital
            "returns": returns_list,
            "total_return": float(period_return)
        }
    
    def calculate_single_metrics(self, result: Dict) -> Dict:
        """Calculate metrics for a single backtest result"""
        returns = pd.Series(result.get("returns", []))
        
        if len(returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = result.get("total_return", 0)
        
        # Annualized metrics
        years = len(returns) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Risk-free rate (2% annual)
        rf_rate = 0.02
        excess_returns = returns.mean() * 252 - rf_rate
        sharpe = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = excess_returns / downside_dev if downside_dev > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade metrics
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        avg_win = returns[returns > 0].mean() if any(returns > 0) else 0
        avg_loss = returns[returns < 0].mean() if any(returns < 0) else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Convert all to native Python types
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.999
        }
    
    def calculate_window_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across windows"""
        
        if not results:
            return {}
        
        all_returns = []
        final_values = []
        
        for result in results:
            final_values.append(result["final_value"])
            all_returns.extend(result.get("returns", []))
        
        if not all_returns:
            return {}
        
        returns_series = pd.Series(all_returns)
        
        # Calculate metrics
        avg_final_value = np.mean(final_values)
        total_return = (avg_final_value / self.config.initial_capital) - 1
        
        # Annualized metrics
        years = len(returns_series) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        volatility = returns_series.std() * np.sqrt(252)
        
        # Risk-free rate (2% annual)
        rf_rate = 0.02
        excess_returns = returns_series.mean() * 252 - rf_rate
        sharpe = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino
        downside_returns = returns_series[returns_series < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = excess_returns / downside_dev if downside_dev > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade metrics
        win_rate = len(returns_series[returns_series > 0]) / len(returns_series) if len(returns_series) > 0 else 0
        avg_win = returns_series[returns_series > 0].mean() if any(returns_series > 0) else 0
        avg_loss = returns_series[returns_series < 0].mean() if any(returns_series < 0) else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Convert all to native Python types
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.999
        }
    
    def compare_performance(self, metrics_with: Dict, 
                           metrics_without: Dict) -> Dict:
        """Compare performance with and without risk management"""
        
        comparison = {}
        
        # If no metrics to compare, return empty comparison
        if not metrics_with or not metrics_without:
            return {
                "summary": {
                    "risk_management_benefit": False,
                    "avg_improvement": 0.0,
                    "avg_improvement_pct": "0%",
                    "recommendation": "Insufficient data for comparison"
                }
            }
        
        # Compare key metrics
        for metric in ["total_return", "sharpe_ratio", "sortino_ratio", 
                      "max_drawdown", "calmar_ratio"]:
            if metric in metrics_with and metric in metrics_without:
                with_val = metrics_with[metric]
                without_val = metrics_without[metric]
                
                if metric == "max_drawdown":
                    # For drawdown, less negative is better
                    improvement = (without_val - with_val) / abs(without_val) if without_val != 0 else 0
                else:
                    improvement = (with_val - without_val) / abs(without_val) if without_val != 0 else 0
                
                # Handle NaN or infinite values
                if np.isnan(improvement) or np.isinf(improvement):
                    improvement = 0.0
                
                comparison[metric] = {
                    "with_risk": float(with_val),
                    "without_risk": float(without_val),
                    "improvement": float(improvement),
                    "improvement_pct": f"{improvement*100:.1f}%"
                }
        
        # Overall assessment
        if comparison:
            improvements = []
            for v in comparison.values():
                if isinstance(v, dict) and "improvement" in v:
                    imp = v["improvement"]
                    if not np.isnan(imp) and not np.isinf(imp):
                        improvements.append(imp)
            
            avg_improvement = float(np.mean(improvements)) if improvements else 0.0
            comparison["summary"] = {
                "risk_management_benefit": bool(avg_improvement > 0),
                "avg_improvement": float(avg_improvement),
                "avg_improvement_pct": f"{avg_improvement*100:.1f}%",
                "recommendation": "Risk management improves performance" if avg_improvement > 0 else "Risk management may not be beneficial for this portfolio"
            }
        
        return comparison