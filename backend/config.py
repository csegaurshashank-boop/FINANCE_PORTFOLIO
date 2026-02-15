"""
Configuration module for portfolio parameters
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class PortfolioConfig:
    """
    Portfolio configuration parameters
    """
    # Required parameters
    assets: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    risk_profile: str  # conservative, moderate, aggressive
    
    # Optional parameters with defaults
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    enable_risk_management: bool = True
    target_volatility: Optional[float] = None
    max_position_size: float = 0.4
    min_position_size: float = 0.05
    cash_buffer: float = 0.05
    
    # Risk thresholds
    stop_loss: Optional[float] = None
    max_drawdown_limit: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
        self.set_risk_thresholds()
    
    def validate(self):
        """Validate configuration parameters"""
        if not self.assets:
            raise ValueError("At least one asset must be specified")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.risk_profile not in ["conservative", "moderate", "aggressive"]:
            raise ValueError("Risk profile must be 'conservative', 'moderate', or 'aggressive'")
        
        if self.rebalance_frequency not in ["daily", "weekly", "monthly", "quarterly"]:
            raise ValueError("Invalid rebalance frequency")
        
        # Validate dates
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def set_risk_thresholds(self):
        """Set risk thresholds based on profile"""
        if self.risk_profile == "conservative":
            self.stop_loss = self.stop_loss or 0.08
            self.max_drawdown_limit = self.max_drawdown_limit or 0.15
            self.target_volatility = self.target_volatility or 0.08
        elif self.risk_profile == "moderate":
            self.stop_loss = self.stop_loss or 0.12
            self.max_drawdown_limit = self.max_drawdown_limit or 0.25
            self.target_volatility = self.target_volatility or 0.12
        else:  # aggressive
            self.stop_loss = self.stop_loss or 0.18
            self.max_drawdown_limit = self.max_drawdown_limit or 0.35
            self.target_volatility = self.target_volatility or 0.18
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "assets": self.assets,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "risk_profile": self.risk_profile,
            "rebalance_frequency": self.rebalance_frequency,
            "enable_risk_management": self.enable_risk_management,
            "target_volatility": self.target_volatility,
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "cash_buffer": self.cash_buffer,
            "stop_loss": self.stop_loss,
            "max_drawdown_limit": self.max_drawdown_limit
        }

# Predefined asset sets
COMMON_ASSETS = {
    "us_stocks": ["SPY", "QQQ", "IWM"],
    "international": ["EFA", "EEM"],
    "bonds": ["TLT", "BND", "LQD"],
    "commodities": ["GLD", "SLV", "USO"],
    "real_estate": ["VNQ"],
    "crypto": ["BTC-USD", "ETH-USD"]
}

# Risk profile templates
RISK_PROFILES = {
    "conservative": {
        "max_volatility": 0.10,
        "max_drawdown": 0.15,
        "min_cash": 0.30,
        "bond_allocation": 0.40,
        "equity_allocation": 0.30,
        "alternative_allocation": 0.30
    },
    "moderate": {
        "max_volatility": 0.15,
        "max_drawdown": 0.25,
        "min_cash": 0.15,
        "bond_allocation": 0.30,
        "equity_allocation": 0.50,
        "alternative_allocation": 0.20
    },
    "aggressive": {
        "max_volatility": 0.25,
        "max_drawdown": 0.35,
        "min_cash": 0.05,
        "bond_allocation": 0.15,
        "equity_allocation": 0.70,
        "alternative_allocation": 0.15
    }
}