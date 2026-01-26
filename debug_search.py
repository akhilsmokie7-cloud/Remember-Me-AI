import requests
from bs4 import BeautifulSoup

print("Debugging Lite DDG Search...")
query = "test search"
url = "https://lite.duckduckgo.com/lite/"
payload = {'q': query}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

try:
    print(f"POST {url}...")
    res = requests.post(url, data=payload, headers=headers, timeout=10)
    print(f"Status: {res.status_code}")
    
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # Lite uses different classes usually. It's a table structure.
    # Look for links that are NOT internal.
    links = soup.select('.result-link')
    print(f"Found {len(links)} '.result-link' items.")
    
    if len(links) > 0:
        print("Success! First link:", links[0].get('href'))
    else:
        # Check title
        print(f"Title: {soup.title.string if soup.title else 'No Title'}")

except Exception as e:
    print(f"Lite Error: {e}")

print("\nDebugging SearXNG (Backup)...")
url_searx = f"https://searx.be/search?q={query}&format=json"
try:
    print(f"GET {url_searx}...")
    res = requests.get(url_searx, headers=headers, timeout=10)
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Found {len(data.get('results', []))} results via JSON.")
except Exception as e:
    print(f"SearXNG Error: {e}")
