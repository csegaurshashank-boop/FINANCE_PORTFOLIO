import os
import logging
import google.generativeai as genai
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Configure API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def get_market_commentary(regime: str, 
                          volatility: float, 
                          drawdown: float, 
                          actions: List[str]) -> str:
    """
    Generate professional market commentary using Gemini API.
    
    Args:
        regime: Current market regime (e.g. "High Volatility", "Normal")
        volatility: Current volatility (e.g. 0.15 for 15%)
        drawdown: Current drawdown (e.g. -0.10 for -10%)
        actions: List of actions taken by the engine
        
    Returns:
        String containing the AI explanation
    """
    
    # Fallback if no key
    if not GOOGLE_API_KEY:
        logger.warning("No GOOGLE_API_KEY found. Using rule-based fallback.")
        return generate_fallback_commentary(regime, volatility, drawdown, actions)

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Act as a professional Hedge Fund Risk Manager. 
        Summarize the current portfolio decision in 2-3 professional sentences.
        
        Market Context:
        - Regime: {regime}
        - Annualized Volatility: {volatility:.1%}
        - Current Drawdown: {drawdown:.1%}
        
        Key Actions Taken:
        {chr(10).join(['- ' + a for a in actions])}
        
        Explain WHY we are taking these actions based on the regime. Be concise.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return generate_fallback_commentary(regime, volatility, drawdown, actions)

def generate_fallback_commentary(regime: str, volatility: float, drawdown: float, actions: List[str]) -> str:
    """Deterministic fallback explanation"""
    base = f"The market is currently in a **{regime}** state with volatility at {volatility:.1%}."
    
    if "Crash" in regime or "Bear" in regime:
        return f"{base} We have shifted to defensive mode to protect capital given the {drawdown:.1%} drawdown."
    elif "High Volatility" in regime:
        return f"{base} We are reducing position sizing to manage risk exposure."
    elif "Trending Up" in regime:
        return f"{base} We are fully allocated to capture upside momentum."
    else:
        return f"{base} We maintains a balanced risk parity allocation."
