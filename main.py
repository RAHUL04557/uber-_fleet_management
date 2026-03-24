from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from utils import build_prediction_frame


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.csv"

LOCATION_COORDINATES = {
    "Times Square": (-73.985130, 40.758896),
    "Central Park": (-73.965355, 40.782865),
    "Empire State Building": (-73.985428, 40.748817),
    "Brooklyn Bridge": (-73.996864, 40.706086),
    "JFK Airport": (-73.778139, 40.641311),
    "LaGuardia Airport": (-73.874001, 40.776928),
    "Wall Street": (-74.009056, 40.706577),
    "Grand Central Terminal": (-73.977229, 40.752726),
    "Statue of Liberty Ferry": (-74.013379, 40.701749),
    "Yankee Stadium": (-73.926174, 40.829643),
}

VEHICLE_OPTIONS = {
    "UberX": {"multiplier": 1.0, "max_passengers": 4},
    "UberXL": {"multiplier": 1.25, "max_passengers": 6},
    "SUV": {"multiplier": 1.3, "max_passengers": 4},
    "Moto": {"multiplier": 0.75, "max_passengers": 1},
}

app = Flask(__name__)


def build_time_options() -> list[str]:
    options = []
    for hour in range(24):
        for minute in (0, 30):
            options.append(time(hour, minute).strftime("%I:%M %p"))
    return options


def get_available_time_options(selected_date: date) -> tuple[list[str], int]:
    all_options = build_time_options()
    now = datetime.now()

    if selected_date > now.date():
        return all_options, all_options.index("07:30 PM")

    next_slot_minutes = ((now.minute // 30) + 1) * 30
    next_slot = now.replace(second=0, microsecond=0)
    if next_slot_minutes >= 60:
        next_slot = next_slot.replace(minute=0) + pd.Timedelta(hours=1)
    else:
        next_slot = next_slot.replace(minute=next_slot_minutes)

    filtered_options = [
        option
        for option in all_options
        if datetime.strptime(option, "%I:%M %p").time() >= next_slot.time()
    ]

    if not filtered_options:
        return [all_options[-1]], 0

    return filtered_options, 0


def load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    metrics_df = pd.read_csv(METRICS_PATH)
    return json.loads(metrics_df.to_json(orient="records"))


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run `python train_models.py` first.")
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]
model_name = artifact["model_name"]


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/meta", methods=["GET"])
def get_meta():
    today = datetime.now().date()
    time_options, default_time_index = get_available_time_options(today)
    return jsonify(
        {
            "modelName": model_name,
            "locations": LOCATION_COORDINATES,
            "vehicles": VEHICLE_OPTIONS,
            "metrics": load_metrics(),
            "today": today.isoformat(),
            "timeOptions": time_options,
            "defaultTimeIndex": default_time_index,
        }
    )


@app.route("/api/time-options", methods=["GET"])
def get_time_options():
    ride_date = request.args.get("rideDate")
    if not ride_date:
        return jsonify({"error": "rideDate is required."}), 400

    try:
        selected_date = date.fromisoformat(ride_date)
    except ValueError:
        return jsonify({"error": "rideDate must be in YYYY-MM-DD format."}), 400

    time_options, default_time_index = get_available_time_options(selected_date)
    return jsonify(
        {
            "timeOptions": time_options,
            "defaultTimeIndex": default_time_index,
        }
    )


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    required_fields = [
        "pickupLocation",
        "dropoffLocation",
        "vehicleType",
        "passengerCount",
        "rideDate",
        "rideTime",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    pickup_location = payload["pickupLocation"]
    dropoff_location = payload["dropoffLocation"]
    vehicle_type = payload["vehicleType"]

    if pickup_location not in LOCATION_COORDINATES:
        return jsonify({"error": "Invalid pickup location."}), 400
    if dropoff_location not in LOCATION_COORDINATES:
        return jsonify({"error": "Invalid dropoff location."}), 400
    if vehicle_type not in VEHICLE_OPTIONS:
        return jsonify({"error": "Invalid vehicle type."}), 400

    try:
        passenger_count = int(payload["passengerCount"])
    except (TypeError, ValueError):
        return jsonify({"error": "passengerCount must be an integer."}), 400

    max_passengers = VEHICLE_OPTIONS[vehicle_type]["max_passengers"]
    if passenger_count < 1 or passenger_count > max_passengers:
        return (
            jsonify(
                {"error": f"Passenger count must be between 1 and {max_passengers}."}
            ),
            400,
        )

    try:
        ride_date = date.fromisoformat(payload["rideDate"])
        ride_time = datetime.strptime(payload["rideTime"], "%I:%M %p").time()
    except ValueError:
        return jsonify({"error": "Invalid ride date or time."}), 400

    pickup_longitude, pickup_latitude = LOCATION_COORDINATES[pickup_location]
    dropoff_longitude, dropoff_latitude = LOCATION_COORDINATES[dropoff_location]
    ride_datetime = pd.Timestamp(datetime.combine(ride_date, ride_time), tz="UTC")

    features = build_prediction_frame(
        pickup_longitude=pickup_longitude,
        pickup_latitude=pickup_latitude,
        dropoff_longitude=dropoff_longitude,
        dropoff_latitude=dropoff_latitude,
        passenger_count=passenger_count,
        ride_datetime=ride_datetime,
    )

    base_fare = float(model.predict(features)[0])
    fare_multiplier = VEHICLE_OPTIONS[vehicle_type]["multiplier"]
    predicted_fare = base_fare * fare_multiplier
    trip_distance = float(features["trip_distance_km"].iloc[0])

    origin = quote_plus(f"{pickup_location} New York")
    destination = quote_plus(f"{dropoff_location} New York")
    directions_link = (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin}"
        f"&destination={destination}"
        "&travelmode=driving"
    )
    embed_src = (
        "https://www.google.com/maps"
        f"?q={origin}+to+{destination}"
        "&z=11&output=embed"
    )

    return jsonify(
        {
            "predictedFare": round(predicted_fare, 2),
            "vehicleType": vehicle_type,
            "passengerCount": passenger_count,
            "tripDistanceKm": round(trip_distance, 2),
            "rideDate": ride_date.isoformat(),
            "rideTime": payload["rideTime"],
            "pickupLocation": pickup_location,
            "dropoffLocation": dropoff_location,
            "directionsLink": directions_link,
            "mapEmbedUrl": embed_src,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
