from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# ✅ Load the trained pipeline
with open(r"D:\data science ppt\classification_project\flight.pkl", "rb") as file:
    pipeline = pickle.load(file)

# ✅ Define the feature columns (no 'class' column)
FEATURE_COLUMNS = [
    'airline', 'flight', 'source_city', 'departure_time',
    'stops', 'arrival_time', 'destination_city', 'duration', 'days_left'
]

@app.get("/hi")
def hello():
    return jsonify({
        "message": "Hello! Flight Price Prediction API is running 🚀",
        "status": 200
    })

@app.post("/predict")
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Validate all required fields
        missing = [col for col in FEATURE_COLUMNS if col not in data]
        if missing:
            return jsonify({
                "error": f"Missing input fields: {missing}",
                "status": 400
            }), 400

        # Convert JSON to DataFrame (columns must match training)
        df = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

        # Make prediction
        prediction = pipeline.predict(df)

        # Return response
        return jsonify({
            "predicted_price": float(prediction[0]),
            "status": 200
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": 500
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
