# arctic_shift.py

import time
import requests


BASE_URL = "https://arctic-shift.photon-reddit.com"


def safe_get(url: str, params: dict, timeout: int = 60, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return response.json()

            print(f"\nRequest failed with status {response.status_code}")
            print("URL:", response.url)
            print("Response:", response.text[:500])

            if response.status_code in [429, 500, 502, 503, 504] or "Timeout" in response.text:
                sleep_seconds = 10 * attempt
                print(f"Sleeping {sleep_seconds} seconds before retry...")
                time.sleep(sleep_seconds)
                continue

            return None

        except requests.exceptions.Timeout:
            sleep_seconds = 10 * attempt
            print(f"Python request timeout on attempt {attempt}. Sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)

        except requests.exceptions.RequestException as e:
            sleep_seconds = 10 * attempt
            print(f"Request error on attempt {attempt}: {e}")
            print(f"Sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)

    return None


def extract_data(json_response):
    if json_response is None:
        return []

    if isinstance(json_response, dict) and "data" in json_response:
        data = json_response["data"]
        if isinstance(data, list):
            return data
        return []

    if isinstance(json_response, list):
        return json_response

    return []


def search_posts(subreddit: str, query: str, after: str, before: str, limit: int = 25):
    url = f"{BASE_URL}/api/posts/search"

    params = {
        "subreddit": subreddit,
        "query": query,
        "after": after,
        "before": before,
        "limit": limit,
        "sort": "desc",
    }

    print(f"Searching posts | r/{subreddit} | query='{query}' | {after} to {before}")

    json_response = safe_get(url, params)
    return extract_data(json_response)