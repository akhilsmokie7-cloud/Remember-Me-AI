import requests
import sys

print(">>> PROBING PORT 8080...")

targets = [
    "http://localhost:8080/",
    "http://localhost:8080/health",
    "http://localhost:8080/v1/models",
    "http://localhost:8080/v1/chat/completions"
]

for url in targets:
    try:
        print(f"\n[TARGET] {url}")
        res = requests.get(url, timeout=2) # GET request
        print(f"   Status: {res.status_code}")
        print(f"   Server Header: {res.headers.get('Server', 'Unknown')}")
        print(f"   Content snippet: {res.text[:100]}...")
    except Exception as e:
        print(f"   [FAIL] {e}")

print("\n>>> DIAGNOSIS COMPLETE.")
