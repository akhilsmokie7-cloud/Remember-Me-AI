import requests
import sys

print("Diagnosing Sovereign Brain (Llama-Server)...")

url = "http://localhost:8080/v1/chat/completions"
# Minimal payload
payload = {
    "messages": [
        {"role": "user", "content": "ping"}
    ]
}

try:
    print(f"1. Pinging Brain at {url}...")
    response = requests.post(url, json=payload, timeout=5)
    
    print(f"2. Status Code: {response.status_code}")
    print(f"3. Response Header: {response.headers}")
    print(f"4. Response Text: {response.text}")

    if response.status_code == 200:
        print("\n[PASS] Brain is active and thinking.")
    else:
        print(f"\n[FAIL] Brain returned error code {response.status_code}.")

except requests.exceptions.ConnectionError:
    print("\n[CRITICAL] Connection Refused. The server is not running or listening on port 8080.")
    print("Action: Check the 'Sovereign Engine' console window. Is it open? Did it crash?")
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
