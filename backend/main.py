"""
FastAPI backend with Pydantic v1 compatibility
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from engine import PortfolioEngine
from data_loader import DataLoader
from backtest import BacktestEngine
from config import PortfolioConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Adaptive Portfolio & Risk Management System",
    description="A comprehensive portfolio management system with adaptive risk management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models (Pydantic v1 syntax)
class PortfolioRequest(BaseModel):
    assets: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    risk_profile: str
    rebalance_frequency: str = "monthly"
    enable_risk_management: bool = True

    # Pydantic v1 validators
    @validator('end_date')
    def validate_end_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    @validator('start_date')
    def validate_start_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

    @validator('risk_profile')
    def validate_risk_profile(cls, v):
        valid_profiles = ["conservative", "moderate", "aggressive"]
        if v not in valid_profiles:
            raise ValueError(f"Risk profile must be one of: {valid_profiles}")
        return v

    @validator('assets')
    def validate_assets(cls, v):
        if not v:
            raise ValueError("At least one asset must be specified")
        return [asset.strip().upper() for asset in v]

class BacktestRequest(BaseModel):
    assets: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    risk_profile: str
    window_size: int = 252
    step_size: int = 63

class StressTestRequest(BaseModel):
    assets: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    risk_profile: str
    shock_type: str
    shock_magnitude: float = 0.3

# Global instances
data_loader = DataLoader()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "Adaptive Portfolio System",
        "endpoints": ["/", "/health", "/status", "/run", "/backtest", "/stress_test"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "ready",
        "message": "System is ready to run portfolio analysis",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/run")
async def run_portfolio(request: PortfolioRequest):
    """Run portfolio optimization"""
    try:
        logger.info(f"Running portfolio with assets: {request.assets}")
        
        # Create config
        config = PortfolioConfig(
            assets=request.assets,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            risk_profile=request.risk_profile,
            rebalance_frequency=request.rebalance_frequency,
            enable_risk_management=request.enable_risk_management
        )
        
        # Initialize and run engine
        engine = PortfolioEngine(config)
        results = await engine.run()
        
        # Convert numpy types for JSON
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime("%Y-%m-%d")
            else:
                return obj
        
        response = {
            "success": True,
            "portfolio_values": convert_to_python(results["portfolio_values"]),
            "drawdowns": convert_to_python(results["drawdowns"]),
            "allocations": convert_to_python(results["allocations"]),
            "regimes": convert_to_python(results["regimes"]),
            "explanations": convert_to_python(results["explanations"]),
            "metrics": convert_to_python(results["metrics"]),
            "current_allocation": convert_to_python(results["current_allocation"])
        }
        
        logger.info("Portfolio run completed successfully")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in run_portfolio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run backtest"""
    try:
        logger.info(f"Running backtest with assets: {request.assets}")
        
        config = PortfolioConfig(
            assets=request.assets,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            risk_profile=request.risk_profile
        )
        
        backtest_engine = BacktestEngine(config)
        results = await backtest_engine.run_rolling_backtest(
            window_size=request.window_size,
            step_size=request.step_size
        )
        
        # Convert numpy types
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            else:
                return obj
        
        response = {
            "success": True,
            "metrics_with_risk": convert_to_python(results.get("metrics_with_risk", {})),
            "metrics_without_risk": convert_to_python(results.get("metrics_without_risk", {})),
            "comparison": convert_to_python(results.get("comparison", {})),
            "rolling_values": convert_to_python(results.get("rolling_values", {}))
        }
        
        logger.info("Backtest completed successfully")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stress_test")
async def run_stress_test(request: StressTestRequest):
    """Run stress test"""
    try:
        logger.info(f"Running stress test: {request.shock_type}")
        
        # Load data
        prices = await data_loader.download_data(
            request.assets,
            request.start_date,
            request.end_date,
            use_sample_if_fails=True
        )
        
        if prices.empty:
            raise ValueError("No data available")
        
        config = PortfolioConfig(
            assets=request.assets,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            risk_profile=request.risk_profile
        )
        
        engine = PortfolioEngine(config)
        
        # Apply shock
        if request.shock_type == "crash":
            shocked_prices = prices * (1 - request.shock_magnitude)
        elif request.shock_type == "volatility_spike":
            returns = prices.pct_change().dropna()
            shocked_returns = returns * (1 + request.shock_magnitude)
            shocked_prices = (1 + shocked_returns).cumprod() * prices.iloc[0]
        else:  # correlation_spike
            returns = prices.pct_change().dropna()
            mean_return = returns.mean(axis=1)
            shocked_returns = pd.DataFrame({
                col: mean_return * (1 + np.random.normal(0, 0.01))
                for col in returns.columns
            }, index=returns.index)
            shocked_prices = (1 + shocked_returns).cumprod() * prices.iloc[0]
        
        results = await engine.run_with_data(shocked_prices)
        
        final_value = float(results["portfolio_values"][-1])
        preservation_ratio = final_value / request.initial_capital
        
        response = {
            "success": True,
            "shock_type": request.shock_type,
            "shock_magnitude": request.shock_magnitude,
            "explanation": f"Applied {request.shock_type} with magnitude {request.shock_magnitude}",
            "portfolio_values": [float(x) for x in results["portfolio_values"]],
            "final_value": final_value,
            "preservation_ratio": float(preservation_ratio),
            "max_drawdown": float(results["metrics"]["max_drawdown"]),
            "recovery_possible": preservation_ratio > 0.7
        }
        
        logger.info("Stress test completed successfully")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in stress test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)