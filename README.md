# Autonomous Adaptive Portfolio & Risk Management Engine

> **54-Hour Hackathon Submission**
> **Team:** Nexus

## 1. Problem Understanding
Financial markets are inherently unstable. Traditional "Buy & Hold" strategies expose investors to massive drawdowns during crashes (e.g., 2008, 2020), while static 60/40 portfolios fail when both stocks and bonds fall simultaneously (e.g., 2022).

**The Core Problem:** Retail investors and simple robo-advisors lack the **dynamic risk management** tools used by professional hedge funds. They cannot adapt to changing market "regimes" (e.g., moving from low-volatility bull markets to high-volatility crashes).

## 2. Approach: Adaptive & Autonomous
Our solution is an **Autonomous Engine** that acts like a digital risk manager. It doesn't just pick stocks; it decides **how much risk to take**.

### Key Innovations:
1.  **Regime Detection**: We use a hybrid model (Volatility + Trend + Drawdown) to classify the market into 4 states:
    *   *Trending Up* (Bull Market) -> Aggressive Allocation
    *   *Normal* -> Balanced Allocation
    *   *High Volatility* -> Reduced Exposure
    *   *Crash* -> Capital Preservation (Cash/Hedging)
2.  **Risk Parity Allocation**: Instead of allocating capital equally, we allocate **risk** equally. Volatile assets get smaller weights.
3.  **Active Risk Controls**:
    *   **Volatility Targeting**: If portfolio volatility exceeds 15%, we automatically deleverage.
    *   **Drawdown Control**: Hard stops triggers if losses exceed predefined thresholds.

## 3. Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** (Python 3.9+) | High-performance API server. |
| **Data Logic** | **Pandas, NumPy, SciPy** | Vectorized financial calculations & optimization. |
| **Data Source** | **yfinance** | Real-time market data ingestion. |
| **Frontend** | **HTML5, CSS3, Vanilla JS** | Lightweight, responsive dashboard. |
| **Visualization**| **Chart.js** | Interactive financial charting. |
| **Process** | **asyncio** | Concurrent data fetching and processing. |

## 4. Limitations
*   **Data Latency**: We rely on end-of-day data for this prototype. A high-frequency version would require a real-time websocket feed.
*   **Transaction Costs**: The current backtest simulation assumes zero slippage and commissions, which might slightly overstate net returns for high-frequency rebalancing.
*   **Asset Universe**: Currently limited to Yahoo Finance supported tickers.
*   **Single-Strategy**: Focused on Risk Parity; does not yet incorporate alternative data (sentiment, macro-economic indicators).

---

## 5. How to Run (Demo)

### Prerequisites
*   Python 3.9+
*   Git

### Installation
```bash
# 1. Clone the repository
git clone <repo-url>
cd portfolio_system

# 2. Setup Backend
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# 3. Start Backend Server
uvicorn main:app --reload --port 8000
```
*Backend is running at `http://localhost:8000`*

### Launch Frontend
Open a new terminal:
```bash
cd frontend
# Start simple server
python -m http.server 8080
```
Visit **`http://localhost:8080`** to see the dashboard.

## 6. Features Checklist (Rulebook Compliance)
- [x] **Architecture Diagram** (See `architecture_diagram.md`)
- [x] **Working Demo** (Live interactive dashboard)
- [x] **Code Freeze** (Core logic stable, only documentation updates)
- [x] **Physical Presence** (Team ready for Monday 9:30 AM)