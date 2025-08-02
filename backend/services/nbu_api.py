"""
Official NBU API Documentation: https://https://bank.gov.ua/ua/open-data/api-dev
"""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime


class NBUAPIError(Exception):
    """Custom exception for NBU API related errors"""
    pass


async def fetch_currency_rates(valcode: Optional[str] = None, date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch currency exchange rates from National Bank of Ukraine API
    
    Args:
        valcode: Currency code (e.g., EUR, USD, GBP). If not provided, returns all currencies
        date: Date in YYYYMMDD format (e.g., 20250804). If not provided, returns today's rates
    
    Returns:
        List of currency data dictionaries from NBU API

    Example:
        >>> rates = await fetch_currency_rates("EUR")
        >>> print(rates[0])
        {
            "r030": 978,
            "txt": "Євро", 
            "rate": 47.6448,
            "cc": "EUR",
            "exchangedate": "04.08.2025"
        }
    """
    try:
        # Build URL for NBU currency exchange API
        url = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
        params = {"json": ""}
        
        if valcode:
            params["valcode"] = valcode.upper()  # Ensure uppercase
        if date:
            params["date"] = date
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
        if not isinstance(data, list):
            raise NBUAPIError(f"Unexpected response format from NBU API: {type(data)}")
            
        return data
        
    except httpx.HTTPStatusError as e:
        raise NBUAPIError(f"NBU API HTTP error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise NBUAPIError(f"NBU API request failed: {str(e)}")
    except Exception as e:
        raise NBUAPIError(f"Unexpected error fetching NBU currency rates: {str(e)}")


def format_currency_data_for_ai(data: List[Dict[str, Any]], limit: int = 30) -> str:
    """
    Format NBU currency data for AI consumption
    
    Args:
        data: List of currency data from NBU API
        limit: Maximum number of currencies to include in output
    
    Returns:
        Formatted string with currency information
    """
    if not data:
        return "No currency data available."
    
    if len(data) == 1:
        # Single currency requested
        item = data[0]
        return f"Currency rate from National Bank of Ukraine as of {item['exchangedate']}:\n{item['txt']} ({item['cc']}): {item['rate']} UAH"
    else:
        # Multiple currencies
        rates_str = "\n".join([f"{item['txt']} ({item['cc']}): {item['rate']} UAH" for item in data[:limit]])
        total_count = len(data)
        if total_count > limit:
            rates_str += f"\n... and {total_count - limit} more currencies"
        
        return f"Currency rates from National Bank of Ukraine as of {data[0]['exchangedate']}:\n{rates_str}"
