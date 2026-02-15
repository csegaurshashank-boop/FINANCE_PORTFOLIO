import requests
import json

# Test the server
base_url = "http://localhost:8000"

def test_health():
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code} - {response.json()}")

def test_status():
    response = requests.get(f"{base_url}/status")
    print(f"Status: {response.status_code} - {response.json()}")

def test_run():
    data = {
        "assets": ["SPY", "QQQ", "TLT", "GLD"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "risk_profile": "moderate",
        "rebalance_frequency": "monthly",
        "enable_risk_management": True
    }
    
    response = requests.post(f"{base_url}/run", json=data)
    print(f"Run portfolio: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        print(f"Final value: ${data['metrics']['final_value']:,.2f}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_health()
    test_status()
    test_run()