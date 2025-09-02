# tools.py
import os
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper


@tool
def google_search_tool(query: str) -> str:
    """Searches Google for up-to-date information."""
    print("LOG: Calling Google Search Tool...")

    # Get the API key from environment variables
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        return "Error: SERPAPI_API_KEY environment variable is not set."

    try:
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        results = search.run(query)
        print("LOG: Google Search Tool finished.")
        return results
    except Exception as e:
        return f"An error occurred during the search: {e}"