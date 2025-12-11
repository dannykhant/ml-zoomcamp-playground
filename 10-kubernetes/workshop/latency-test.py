import time
import requests
import os

def measure_response_time(url: str, method: str = "GET", **kwargs):
    start = time.perf_counter()
    response = requests.request(method, url, **kwargs)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    return response.status_code, round(elapsed_ms, 2)

if __name__ == "__main__":
    node_ip = os.getenv('NODE_IP', 'localhost')
    url = f"http://{node_ip}:30080/predict"

    for i in range(10):
        status, latency = measure_response_time(url, "POST", json={"url": "http://bit.ly/mlbookcamp-pants"})
        print(f"Status: {status}, Response time: {latency} ms")
