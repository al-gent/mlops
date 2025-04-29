# Create a file called test_client.py
import requests

# Sample coordinates for San Francisco
response = requests.post(
    "http://127.0.0.1:8000/next_sweep",
    json={"latitude": 37.7749, "longitude": -122.4194}
)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())