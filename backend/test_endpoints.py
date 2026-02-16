
import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestEndpoints")

async def test_stress_endpoint():
    url = "http://localhost:8000/stress_test"
    payload = {
        "assets": ["AAPL", "GOOGL"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000,
        "risk_profile": "moderate",
        "shock_type": "crash",
        "shock_magnitude": 0.2
    }
    
    logger.info(f"Testing {url} with payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Success! Response:")
                    logger.info(json.dumps(data, indent=2))
                else:
                    text = await response.text()
                    logger.error(f"❌ Failed: Status {response.status}")
                    logger.error(text)
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        logger.info("Make sure the server is running: 'uvicorn backend.main:app --reload'")

if __name__ == "__main__":
    asyncio.run(test_stress_endpoint())
