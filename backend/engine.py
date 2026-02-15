"""
Portfolio Engine - Core logic for regime detection, allocation, and risk management
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from scipy.optimize import minimize
from dataclasses import dataclass

from data_loader import DataLoader
from config import PortfolioConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeInfo:
    """Information about current market regime"""
    name: str
    volatility_level: float
    drawdown_level: float
    description: str
    risk_multiplier: float

class PortfolioEngine:
    """
    Main portfolio engine handling:
    - Data loading and preprocessing
    - Regime detection
    - Portfolio allocation
    - Risk management
    - Performance tracking
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.prices = None
        self.returns = None
        self.volatility = None
        self.drawdown = None
        self.correlation_matrix = None
        
        # Portfolio state
        self.portfolio_value = config.initial_capital
        self.positions = {asset: 0 for asset in config.assets}
        self.allocation_history = []
        self.value_history = [config.initial_capital]
        self.drawdown_history = [0]
        self.regime_history = []
        self.explanations = []
        
        # Current regime
        self.current_regime = "Normal"
        self.regime_info = None
        
        # Risk parameters based on profile
        self.set_risk_parameters()
        
    def set_risk_parameters(self):
        """Set risk parameters based on user profile"""
        profiles = {
            "conservative": {
                "max_volatility": 0.10,
                "max_drawdown": 0.15,
                "min_cash": 0.30,
                "stop_loss": 0.08,
                "target_volatility": 0.08
            },
            "moderate": {
                "max_volatility": 0.15,
                "max_drawdown": 0.25,
                "min_cash": 0.15,
                "stop_loss": 0.12,
                "target_volatility": 0.12
            },
            "aggressive": {
                "max_volatility": 0.25,
                "max_drawdown": 0.35,
                "min_cash": 0.05,
                "stop_loss": 0.18,
                "target_volatility": 0.18
            }
        }
        
        self.risk_params = profiles.get(self.config.risk_profile, profiles["moderate"])
        
    async def load_data(self) -> bool:
        """Load and prepare market data"""
        try:
            # Download data
            self.prices = await self.data_loader.download_data(
                self.config.assets,
                self.config.start_date,
                self.config.end_date
            )
            
            if self.prices.empty:
                logger.error("No data loaded")
                return False
            
            # Calculate returns
            self.returns = self.prices.pct_change().dropna()
            
            if self.returns.empty:
                logger.error("No valid returns calculated")
                return False
            
            # Calculate rolling volatility (21-day)
            self.volatility = self.returns.rolling(window=21).std() * np.sqrt(252)
            
            # Calculate rolling correlation (only if multiple assets)
            if len(self.config.assets) > 1:
                self.correlation_matrix = self.returns.rolling(window=60).corr()
            
            logger.info(f"Data loaded successfully: {len(self.prices)} days, {len(self.returns)} returns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def detect_regime(self, date: pd.Timestamp) -> RegimeInfo:
        """
        Detect market regime based on volatility and drawdown
        Returns: RegimeInfo object with regime details
        """
        if date not in self.prices.index:
            return RegimeInfo("Normal", 0.1, 0, "Normal market conditions", 1.0)
        
        # Get current price data
        current_prices = self.prices.loc[date]
        
        # Calculate current drawdown from peak
        historical_peak = self.prices.loc[:date].max()
        current_drawdown = (current_prices / historical_peak - 1).min()
        
        # Get current volatility
        if date in self.volatility.index:
            current_vol = self.volatility.loc[date].mean()
        else:
            # Find the nearest date with volatility data
            nearest_idx = self.volatility.index.get_indexer([date], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(self.volatility.index):
                nearest_date = self.volatility.index[nearest_idx]
                current_vol = self.volatility.loc[nearest_date].mean()
            else:
                current_vol = 0.15  # Default
        
        # Handle NaN values
        if pd.isna(current_vol):
            current_vol = 0.15
        if pd.isna(current_drawdown):
            current_drawdown = 0
        
        # Determine regime
        if current_drawdown < -0.20 and current_vol > 0.30:
            regime = "Crash"
            description = "Market crash detected: Severe drawdown with extreme volatility"
            risk_mult = 0.3
        elif current_drawdown < -0.10 and current_vol > 0.25:
            regime = "Bear"
            description = "Bear market: Significant drawdown with high volatility"
            risk_mult = 0.5
        elif current_vol > 0.25:
            regime = "High Volatility"
            description = "High volatility regime: Increased market uncertainty"
            risk_mult = 0.6
        elif current_vol > 0.15:
            regime = "Elevated Volatility"
            description = "Moderately elevated volatility"
            risk_mult = 0.8
        else:
            regime = "Normal"
            description = "Normal market conditions with stable volatility"
            risk_mult = 1.0
        
        # Add drawdown consideration to regime
        if current_drawdown < -0.05:
            description += f" Current drawdown: {current_drawdown:.1%}"
        
        self.current_regime = regime
        
        return RegimeInfo(
            name=regime,
            volatility_level=float(current_vol),
            drawdown_level=float(current_drawdown),
            description=description,
            risk_multiplier=risk_mult
        )
    
    def calculate_risk_parity_weights(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate risk parity weights based on asset volatilities and correlations
        """
        if date not in self.returns.index:
            return {asset: 1.0/len(self.config.assets) for asset in self.config.assets}
        
        # Get recent returns for covariance estimation
        date_idx = self.returns.index.get_loc(date)
        lookback = min(60, date_idx)
        if lookback < 30:
            lookback = 30
        
        recent_returns = self.returns.iloc[date_idx - lookback:date_idx + 1]
        
        if len(recent_returns) < 2:
            return {asset: 1.0/len(self.config.assets) for asset in self.config.assets}
        
        # Calculate covariance matrix
        cov_matrix = recent_returns.cov() * 252
        
        # Handle potential issues with covariance matrix
        if cov_matrix.isnull().any().any():
            return {asset: 1.0/len(self.config.assets) for asset in self.config.assets}
        
        # Objective function for risk parity
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            
            # Calculate risk contribution of each asset
            marginal_risk = cov_matrix @ weights
            risk_contrib = weights * marginal_risk / portfolio_risk
            
            # Target equal risk contribution
            target_risk = 1.0 / len(weights)
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.05, 0.4) for _ in self.config.assets]  # Min/max weights
        
        # Initial guess
        n_assets = len(self.config.assets)
        x0 = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                weights = result.x
            else:
                weights = x0
        except:
            weights = x0
        
        return {asset: float(weights[i]) for i, asset in enumerate(self.config.assets)}
    
    def apply_momentum_filter(self, weights: Dict[str, float], date: pd.Timestamp) -> Dict[str, float]:
        """
        Apply momentum filter - reduce weights of assets with negative momentum
        """
        if date not in self.prices.index:
            return weights
        
        # Calculate momentum (6-month performance)
        date_idx = self.prices.index.get_loc(date)
        lookback = min(126, date_idx)  # ~6 months, ensure min data
        if lookback < 63:  # Need at least 3 months
            return weights
        
        start_idx = max(0, date_idx - lookback)
        start_date = self.prices.index[start_idx]
        
        # Calculate momentum safely
        momentum = {}
        for asset in self.config.assets:
            if asset in self.prices.columns:
                current_price = self.prices.loc[date, asset]
                start_price = self.prices.loc[start_date, asset]
                if start_price > 0 and current_price > 0:
                    momentum[asset] = (current_price / start_price - 1)
                else:
                    momentum[asset] = 0
            else:
                momentum[asset] = 0
        
        # Apply momentum filter
        filtered_weights = {}
        total_positive_momentum = 0
        
        for asset, weight in weights.items():
            if asset in momentum and momentum[asset] > 0:
                # Keep full weight for positive momentum
                filtered_weights[asset] = weight
                total_positive_momentum += weight
            else:
                # Reduce weight for negative momentum
                filtered_weights[asset] = weight * 0.5
                total_positive_momentum += weight * 0.5
        
        # Renormalize
        if total_positive_momentum > 0:
            for asset in filtered_weights:
                filtered_weights[asset] /= total_positive_momentum
        else:
            # If all momentum negative, keep original weights
            filtered_weights = weights.copy()
        
        return filtered_weights
    
    def apply_risk_management(self, weights: Dict[str, float], regime: RegimeInfo, 
                            date: pd.Timestamp) -> Tuple[Dict[str, float], List[str]]:
        """
        Apply comprehensive risk management rules
        Returns: (adjusted_weights, explanations)
        """
        explanations = []
        adjusted_weights = weights.copy()
        
        # 1. Volatility targeting
        if regime.volatility_level > self.risk_params["target_volatility"]:
            vol_ratio = self.risk_params["target_volatility"] / regime.volatility_level
            vol_ratio = min(vol_ratio, 1.0)  # Don't increase risk
            
            for asset in adjusted_weights:
                adjusted_weights[asset] *= vol_ratio
            
            explanations.append(
                f"Volatility targeting: Current volatility {regime.volatility_level:.1%} exceeds target "
                f"{self.risk_params['target_volatility']:.1%}. Reducing exposure by {(1-vol_ratio)*100:.0f}%."
            )
        
        # 2. Drawdown protection
        if regime.drawdown_level < -self.risk_params["max_drawdown"]:
            # Reduce exposure proportionally to drawdown severity
            drawdown_ratio = 1 - (abs(regime.drawdown_level) / self.risk_params["max_drawdown"])
            drawdown_ratio = max(0.3, min(1.0, drawdown_ratio))
            
            for asset in adjusted_weights:
                adjusted_weights[asset] *= drawdown_ratio
            
            explanations.append(
                f"Drawdown protection: Current drawdown {regime.drawdown_level:.1%} exceeds "
                f"maximum {self.risk_params['max_drawdown']:.1%}. Reducing exposure by {(1-drawdown_ratio)*100:.0f}%."
            )
        
        # 3. Stop-loss check (for individual positions)
        if date in self.prices.index:
            for asset in list(adjusted_weights.keys()):
                # Check if position has hit stop-loss
                if asset in self.positions and self.positions.get(asset, 0) > 0:
                    entry_price = self.get_entry_price(asset, date)
                    current_price = self.prices.loc[date, asset]
                    
                    if entry_price > 0:
                        loss = (current_price - entry_price) / entry_price
                        if loss < -self.risk_params["stop_loss"]:
                            adjusted_weights[asset] = 0
                            explanations.append(
                                f"Stop-loss triggered for {asset}: Loss of {loss:.1%} exceeds "
                                f"{self.risk_params['stop_loss']:.1%}. Position closed."
                            )
        
        # 4. Regime-based adjustment
        for asset in adjusted_weights:
            adjusted_weights[asset] *= regime.risk_multiplier
        
        if regime.risk_multiplier < 1.0:
            explanations.append(
                f"Regime adjustment: {regime.description}. Reducing overall exposure by "
                f"{(1-regime.risk_multiplier)*100:.0f}%."
            )
        
        # 5. Ensure minimum cash position
        total_exposure = sum(adjusted_weights.values())
        if total_exposure > 1 - self.risk_params["min_cash"]:
            scale = (1 - self.risk_params["min_cash"]) / total_exposure
            for asset in adjusted_weights:
                adjusted_weights[asset] *= scale
            explanations.append(
                f"Minimum cash position: Maintaining {self.risk_params['min_cash']:.0%} cash reserve."
            )
        
        # 6. Re-normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for asset in adjusted_weights:
                adjusted_weights[asset] /= total_weight
        
        return adjusted_weights, explanations
    
    def get_entry_price(self, asset: str, current_date: pd.Timestamp) -> float:
        """Get entry price for position tracking (simplified)"""
        # In production, would track actual purchase prices
        # For backtesting, use average price over last month
        date_idx = self.prices.index.get_loc(current_date)
        lookback = min(21, date_idx)
        if lookback < 5:
            return self.prices.loc[current_date, asset]
        
        return self.prices.iloc[date_idx - lookback:date_idx + 1][asset].mean()
    
    async def run(self) -> Dict:
        """
        Main execution loop - runs portfolio optimization and tracking
        """
        logger.info("Starting portfolio engine...")
        
        # Load data
        if not await self.load_data():
            logger.warning("Using default/sample data for demonstration")
            # Create sample data if loading failed
            if self.prices is None or self.prices.empty:
                self.prices = self.data_loader._get_sample_data(
                    self.config.assets,
                    self.config.start_date,
                    self.config.end_date
                )
                self.returns = self.prices.pct_change().dropna()
                self.volatility = self.returns.rolling(window=21).std() * np.sqrt(252)
        
        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates()
        
        # Initialize tracking
        portfolio_values = []
        drawdowns = []
        allocations = []
        regimes = []
        all_explanations = []
        peak_value = self.config.initial_capital
        
        # Current portfolio state
        current_weights = {asset: 1.0/len(self.config.assets) for asset in self.config.assets}
        
        # Iterate through each date
        for i, date in enumerate(self.prices.index):
            daily_explanations = []
            
            # Check if rebalance needed
            if date in rebalance_dates or i == 0:
                # Detect regime
                regime_info = self.detect_regime(date)
                regimes.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "regime": regime_info.name,
                    "volatility": float(regime_info.volatility_level),
                    "drawdown": float(regime_info.drawdown_level)
                })
                
                # Calculate base allocation
                if self.config.enable_risk_management:
                    # Risk parity + momentum
                    rp_weights = self.calculate_risk_parity_weights(date)
                    momentum_weights = self.apply_momentum_filter(rp_weights, date)
                    
                    # Apply risk management
                    adjusted_weights, risk_explanations = self.apply_risk_management(
                        momentum_weights, regime_info, date
                    )
                    daily_explanations.extend(risk_explanations)
                    
                    current_weights = adjusted_weights
                else:
                    # Equal weight benchmark
                    current_weights = {asset: 1.0/len(self.config.assets) for asset in self.config.assets}
                    daily_explanations.append("Equal weight allocation (risk management disabled)")
                
                # Record allocation
                allocations.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "weights": current_weights.copy()
                })
            
            # Calculate portfolio value for the day
            if i > 0:
                prev_date = self.prices.index[i-1]
                daily_returns = (self.prices.loc[date] / self.prices.loc[prev_date] - 1)
                
                # Update portfolio value
                portfolio_return = 0
                for asset in self.config.assets:
                    if asset in daily_returns.index:
                        portfolio_return += current_weights.get(asset, 0) * daily_returns[asset]
                
                self.portfolio_value *= (1 + portfolio_return)
                
                # Track peak for drawdown
                peak_value = max(peak_value, self.portfolio_value)
                current_drawdown = (self.portfolio_value - peak_value) / peak_value
                
                # Update positions (simplified)
                for asset in self.config.assets:
                    if asset in daily_returns.index:
                        self.positions[asset] = (self.positions.get(asset, 0) * 
                                               (1 + daily_returns[asset]))
            else:
                current_drawdown = 0
            
            portfolio_values.append(float(self.portfolio_value))
            drawdowns.append(float(current_drawdown))
            
            # Add daily explanation
            if i > 0 and 'portfolio_return' in locals():
                daily_explanations.append(
                    f"Daily return: {portfolio_return:.4f}. Portfolio value: ${self.portfolio_value:,.2f}"
                )
            
            all_explanations.extend([{
                "date": date.strftime("%Y-%m-%d"),
                "explanation": exp
            } for exp in daily_explanations])
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio_values)
        
        logger.info(f"Portfolio run completed. Final value: ${metrics['final_value']:,.2f}")
        
        return {
            "portfolio_values": portfolio_values,
            "drawdowns": drawdowns,
            "allocations": allocations,
            "regimes": regimes,
            "explanations": all_explanations,
            "metrics": metrics,
            "current_allocation": current_weights
        }
    
    async def run_with_data(self, prices: pd.DataFrame) -> Dict:
        """Run engine with pre-loaded data (for stress testing)"""
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.volatility = self.returns.rolling(window=21).std() * np.sqrt(252)
        
        # Reset state
        self.portfolio_value = self.config.initial_capital
        self.positions = {asset: 0 for asset in self.config.assets}
        
        return await self.run()
    
    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get rebalance dates based on frequency"""
        if self.config.rebalance_frequency == "daily":
            return self.prices.index.tolist()
        elif self.config.rebalance_frequency == "weekly":
            return [date for date in self.prices.index if date.weekday() == 0]
        elif self.config.rebalance_frequency == "monthly":
            return pd.date_range(start=self.prices.index[0], 
                               end=self.prices.index[-1], 
                               freq='MS').intersection(self.prices.index)
        else:  # quarterly
            return pd.date_range(start=self.prices.index[0], 
                               end=self.prices.index[-1], 
                               freq='QS').intersection(self.prices.index)
    
    def calculate_metrics(self, portfolio_values: List[float]) -> Dict:
        """Calculate performance metrics"""
        if len(portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "final_value": float(portfolio_values[-1]) if portfolio_values else 0.0
            }
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if len(returns) == 0:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "final_value": float(portfolio_values[-1])
            }
        
        # Risk-free rate (assume 2% annual)
        rf_rate = 0.02
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        years = len(portfolio_values) / 252
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1/years) - 1 if years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = returns.mean() * 252 - rf_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = excess_return / downside_dev if downside_dev > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "final_value": float(portfolio_values[-1])
        }