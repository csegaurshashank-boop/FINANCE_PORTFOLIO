📈 Autonomous Adaptive Portfolio & Risk Management System
An intelligent portfolio management system that combines adaptive risk management, market regime detection, and explainable AI decisions with an interactive real-time dashboard.

https://img.shields.io/badge/version-1.0.0-blue
https://img.shields.io/badge/python-3.9%252B-green
https://img.shields.io/badge/FastAPI-0.104.1-teal

🚀 Features
Real-time Data: yfinance integration with automatic sample data fallback

Regime Detection: Identifies Normal, High Volatility, Bear, and Crash regimes

Adaptive Allocation: Risk parity + momentum hybrid with regime-based adjustments

Risk Management: Volatility targeting, drawdown protection, stop-loss logic

Backtesting: Rolling window analysis with/without risk management comparison

Stress Testing: Market crash, volatility spike, and correlation spike scenarios

Explainable AI: Natural language explanations for all decisions

Interactive Dashboard: Real-time charts with Chart.js

🏗️ Quick Start
Prerequisites
Python 3.9+

Git

Installation
bash
# Clone repository
git clone https://github.com/yourusername/portfolio_system.git
cd portfolio_system

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start server
uvicorn main:app --reload --port 8000

# In new terminal - launch frontend
cd frontend
# Open index.html in browser or run:
python -m http.server 8080
Visit http://localhost:8080 for the dashboard, API at http://localhost:8000

📊 Usage
Select Assets: Search or click popular assets (SPY, QQQ, AAPL, BTC-USD, etc.)

Configure: Set date range, capital, risk profile (Conservative/Moderate/Aggressive)

Run Analysis:

Run Portfolio: Full optimization with explanations

Run Backtest: Compare with/without risk management

Stress Test: Simulate market shocks

🎯 Risk Profiles
Profile	Max Vol	Max DD	Min Cash	Stop Loss
Conservative	10%	15%	30%	8%
Moderate	15%	25%	15%	12%
Aggressive	25%	35%	5%	18%
📡 API Endpoints
GET /health - Server status

POST /run - Run portfolio optimization

POST /backtest - Run rolling backtest

POST /stress_test - Run stress test

🛠️ Tech Stack
Backend: FastAPI, Pydantic, Pandas, NumPy, yfinance
Frontend: HTML5, CSS3, JavaScript, Chart.js
Data: yfinance (primary) + intelligent sample data fallback

📁 Project Structure
text
portfolio_system/
├── backend/
│   ├── main.py          # FastAPI server
│   ├── engine.py        # Core portfolio logic
│   ├── data_loader.py   # Market data handling
│   ├── backtest.py      # Backtesting engine
│   ├── config.py        # Configuration
│   └── requirements.txt
├── frontend/
│   ├── index.html       # Dashboard
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
└── README.md
🔧 Configuration
Create .env in backend folder (optional - works without API keys):

env
MARKETDATA_TOKEN=your_token      # For real market data
📈 Key Metrics
CAGR: Compound Annual Growth Rate

Sharpe Ratio: Risk-adjusted return

Max Drawdown: Largest peak-to-trough decline

Calmar Ratio: CAGR / Max Drawdown

Win Rate: % of profitable periods

⚡ Troubleshooting
Backend won't start: pip install -r requirements.txt
No data showing: Check backend on port 8000, browser console (F12)
yfinance errors: pip install --upgrade yfinance

📝 License
MIT License - free for academic and commercial use