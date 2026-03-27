import requests

url = "http://localhost:8000/predict"
trip = {
    "PULocationID": "43",
    "DOLocationID": "238",
    "trip_distance": 1.16,
}
response = requests.post(url, json=trip).json()
print(response)

prediction = response["prediction"]["duration"]
# assert abs(prediction - 8.43) < 0.01