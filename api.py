from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return "Cardio Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['data']

    data_scaled = scaler.transform([data])
    prediction = model.predict(data_scaled)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)