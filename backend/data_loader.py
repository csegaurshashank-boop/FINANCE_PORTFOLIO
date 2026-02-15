"""
Data loading module using yfinance as primary source
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import asyncio
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles downloading and preprocessing of market data
    Primary: yfinance (real data)
    Fallback: Sample data generation
    """
    
    def __init__(self):
        self.cache = {}
        # Configure yfinance to be more reliable
        yf.set_tz_cache_location(None)  # Disable timezone caching issues
        
    async def download_data(self, symbols: List[str], start_date: str, 
                           end_date: str, use_sample_if_fails: bool = True) -> pd.DataFrame:
        """
        Download historical price data using yfinance
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_sample_if_fails: If True, use sample data when download fails
            
        Returns:
            DataFrame with dates as index and symbols as columns
        """
        try:
            # Clean and validate symbols
            symbols = [s.strip().upper() for s in symbols]
            
            # Adjust dates - don't use future dates
            end = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now()
            
            if end > today:
                end_date = today.strftime("%Y-%m-%d")
                logger.warning(f"End date adjusted to today: {end_date}")
            
            # Ensure we have at least some data
            start = datetime.strptime(start_date, "%Y-%m-%d")
            if (end - start).days < 30:
                start_date = (end - timedelta(days=365)).strftime("%Y-%m-%d")
                logger.warning(f"Start date adjusted to ensure sufficient data: {start_date}")
            
            logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
            
            # Create cache key
            cache_key = f"{','.join(sorted(symbols))}_{start_date}_{end_date}"
            
            # Check cache
            if cache_key in self.cache:
                logger.info("Returning cached data")
                return self.cache[cache_key].copy()
            
            # Try yfinance first
            prices = await self._download_yfinance(symbols, start_date, end_date)
            
            # If yfinance fails and we should use sample data
            if (prices is None or prices.empty) and use_sample_if_fails:
                logger.warning("yfinance failed. Using sample data.")
                prices = self._get_sample_data(symbols, start_date, end_date)
            
            if prices is not None and not prices.empty:
                # Store in cache
                self.cache[cache_key] = prices.copy()
                logger.info(f"Successfully loaded {len(prices)} days of data for {len(prices.columns)} symbols")
                return prices
            else:
                raise ValueError("No data available")
            
        except Exception as e:
            logger.error(f"Error in download_data: {str(e)}")
            if use_sample_if_fails:
                logger.warning("Returning sample data as fallback")
                return self._get_sample_data(symbols, start_date, end_date)
            raise
    
    async def _download_yfinance(self, symbols: List[str], start_date: str, 
                                 end_date: str) -> pd.DataFrame:
        """Download data using yfinance with multiple strategies"""
        
        strategies = [
            self._download_batch,
            self._download_individual,
            self._download_with_delays
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Trying yfinance strategy {i+1}/{len(strategies)}")
                prices = await strategy(symbols, start_date, end_date)
                
                if prices is not None and not prices.empty and len(prices.columns) > 0:
                    # Validate the data
                    if self._validate_downloaded_data(prices):
                        logger.info(f"Strategy {i+1} successful")
                        return prices
                    
            except Exception as e:
                logger.warning(f"Strategy {i+1} failed: {str(e)}")
                await asyncio.sleep(1)  # Wait before next strategy
        
        return pd.DataFrame()
    
    async def _download_batch(self, symbols: List[str], start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """Download all symbols at once"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Download with specific parameters for reliability
            data = await loop.run_in_executor(
                None, 
                lambda: yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    timeout=30,
                    threads=True,
                    prepost=False,
                    group_by='ticker'
                )
            )
            
            if data is None or data.empty:
                return pd.DataFrame()
            
            # Extract close prices
            prices = pd.DataFrame()
            
            if len(symbols) == 1:
                # Single symbol
                if isinstance(data, pd.DataFrame):
                    if 'Close' in data.columns:
                        prices = data[['Close']].copy()
                        prices.columns = symbols
                    elif 'Adj Close' in data.columns:
                        prices = data[['Adj Close']].copy()
                        prices.columns = symbols
            else:
                # Multiple symbols
                for symbol in symbols:
                    if symbol in data.columns.get_level_values(1) if hasattr(data.columns, 'get_level_values') else False:
                        if ('Adj Close', symbol) in data.columns:
                            prices[symbol] = data[('Adj Close', symbol)]
                        elif ('Close', symbol) in data.columns:
                            prices[symbol] = data[('Close', symbol)]
            
            if not prices.empty:
                # Forward fill any missing values
                prices = prices.ffill().bfill()
                
            return prices
            
        except Exception as e:
            logger.warning(f"Batch download failed: {str(e)}")
            return pd.DataFrame()
    
    async def _download_individual(self, symbols: List[str], start_date: str, 
                                  end_date: str) -> pd.DataFrame:
        """Download symbols one by one for better reliability"""
        prices_list = []
        valid_symbols = []
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol} individually")
                
                # Run in thread pool
                loop = asyncio.get_event_loop()
                
                # Create ticker and get history
                ticker = yf.Ticker(symbol)
                hist = await loop.run_in_executor(
                    None,
                    lambda: ticker.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        timeout=10
                    )
                )
                
                if not hist.empty and 'Close' in hist.columns:
                    prices_list.append(hist['Close'])
                    valid_symbols.append(symbol)
                    logger.info(f"✅ Successfully downloaded {symbol}")
                else:
                    logger.warning(f"⚠️ No data for {symbol}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"❌ Failed to download {symbol}: {str(e)}")
        
        if prices_list:
            prices = pd.concat(prices_list, axis=1)
            prices.columns = valid_symbols
            return prices
        
        return pd.DataFrame()
    
    async def _download_with_delays(self, symbols: List[str], start_date: str, 
                                   end_date: str) -> pd.DataFrame:
        """Download with longer delays between requests"""
        prices_list = []
        valid_symbols = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Downloading {symbol} with delay strategy")
                
                # Add increasing delay
                delay = 2 + (i * 1)
                await asyncio.sleep(delay)
                
                # Try with different parameters
                loop = asyncio.get_event_loop()
                ticker = yf.Ticker(symbol)
                hist = await loop.run_in_executor(
                    None,
                    lambda: ticker.history(
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        auto_adjust=True,
                        actions=False
                    )
                )
                
                if not hist.empty and 'Close' in hist.columns:
                    prices_list.append(hist['Close'])
                    valid_symbols.append(symbol)
                    logger.info(f"✅ Successfully downloaded {symbol}")
                    
            except Exception as e:
                logger.warning(f"❌ Failed to download {symbol}: {str(e)}")
        
        if prices_list:
            prices = pd.concat(prices_list, axis=1)
            prices.columns = valid_symbols
            return prices
        
        return pd.DataFrame()
    
    def _validate_downloaded_data(self, prices: pd.DataFrame) -> bool:
        """Validate downloaded data quality"""
        if prices is None or prices.empty:
            return False
        
        # Check for sufficient data
        if len(prices) < 20:
            logger.warning(f"Insufficient data points: {len(prices)}")
            return False
        
        # Check for too many NaN values
        nan_ratio = prices.isnull().sum().max() / len(prices)
        if nan_ratio > 0.1:
            logger.warning(f"Too many NaN values: {nan_ratio:.1%}")
            return False
        
        # Check for constant values (possible bad data)
        for col in prices.columns:
            if prices[col].nunique() < 2:
                logger.warning(f"Column {col} has constant values")
                return False
        
        return True
    
    def _get_sample_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic sample data for testing"""
        logger.info("Generating sample data for testing")
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='B')
        
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        
        # Generate sample prices with realistic correlations
        np.random.seed(42)
        prices = {}
        
        # Base market factor
        market_returns = np.random.normal(0.0005, 0.01, len(dates))
        
        # Asset-specific parameters
        for i, symbol in enumerate(symbols):
            price = 100
            price_series = []
            
            # Adjust parameters based on symbol type
            if 'BTC' in symbol or 'ETH' in symbol:
                # Crypto: higher volatility
                beta = 1.5
                alpha = 0.001
                vol = 0.02
            elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
                # Tech stocks
                beta = 1.2
                alpha = 0.0003
                vol = 0.008
            else:
                # Default
                beta = 0.8 + (i * 0.1)
                alpha = 0.0001 * (i + 1)
                vol = 0.005 * (i + 1)
            
            for j in range(len(dates)):
                market_component = market_returns[j] * beta
                specific_component = np.random.normal(alpha, vol)
                daily_return = market_component + specific_component
                price *= (1 + daily_return)
                price_series.append(price)
            
            prices[symbol] = price_series
        
        df = pd.DataFrame(prices, index=dates)
        logger.info(f"Generated sample data with {len(df)} rows for {len(symbols)} symbols")
        return df
    
    def calculate_returns(self, prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
        """Calculate returns from price data"""
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return returns.dropna()
    
    def calculate_volatility(self, returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_drawdown(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown from peak"""
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown