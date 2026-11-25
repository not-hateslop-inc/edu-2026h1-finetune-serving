import requests
r = requests.post("http://localhost:8000/classify", json={"texts": ["실습이 좋았어요", "녹화본 올려 주세요"]}, timeout=10)
print(r.json())
