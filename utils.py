from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "trip_distance_km",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
]


def haversine_distance(
    pickup_latitude: pd.Series,
    pickup_longitude: pd.Series,
    dropoff_latitude: pd.Series,
    dropoff_longitude: pd.Series,
) -> pd.Series:
    radius_km = 6371.0

    lat1 = np.radians(pickup_latitude.astype(float))
    lon1 = np.radians(pickup_longitude.astype(float))
    lat2 = np.radians(dropoff_latitude.astype(float))
    lon2 = np.radians(dropoff_longitude.astype(float))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c


def clean_uber_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=["Unnamed: 0", "key"], errors="ignore")
    cleaned = cleaned.dropna()

    cleaned["pickup_datetime"] = pd.to_datetime(
        cleaned["pickup_datetime"], errors="coerce", utc=True
    )
    cleaned = cleaned.dropna(subset=["pickup_datetime"])

    cleaned = cleaned[
        (cleaned["fare_amount"] > 0)
        & (cleaned["fare_amount"] < 100)
        & (cleaned["passenger_count"] > 0)
        & (cleaned["passenger_count"] <= 6)
        & (cleaned["pickup_longitude"].between(-80, -70))
        & (cleaned["dropoff_longitude"].between(-80, -70))
        & (cleaned["pickup_latitude"].between(35, 45))
        & (cleaned["dropoff_latitude"].between(35, 45))
    ]

    cleaned["trip_distance_km"] = haversine_distance(
        cleaned["pickup_latitude"],
        cleaned["pickup_longitude"],
        cleaned["dropoff_latitude"],
        cleaned["dropoff_longitude"],
    )
    cleaned = cleaned[
        (cleaned["trip_distance_km"] > 0)
        & (cleaned["trip_distance_km"] < 100)
    ]

    cleaned["hour"] = cleaned["pickup_datetime"].dt.hour
    cleaned["day"] = cleaned["pickup_datetime"].dt.day
    cleaned["month"] = cleaned["pickup_datetime"].dt.month
    cleaned["day_of_week"] = cleaned["pickup_datetime"].dt.dayofweek
    cleaned["is_weekend"] = cleaned["day_of_week"].isin([5, 6]).astype(int)

    return cleaned


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_uber_data(df)
    features = cleaned[FEATURE_COLUMNS].copy()
    target = cleaned["fare_amount"].copy()
    return features, target


def build_prediction_frame(
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
    ride_datetime: pd.Timestamp,
) -> pd.DataFrame:
    ride_datetime = pd.Timestamp(ride_datetime)
    if ride_datetime.tzinfo is None:
        ride_datetime = ride_datetime.tz_localize("UTC")
    else:
        ride_datetime = ride_datetime.tz_convert("UTC")

    distance = haversine_distance(
        pd.Series([pickup_latitude]),
        pd.Series([pickup_longitude]),
        pd.Series([dropoff_latitude]),
        pd.Series([dropoff_longitude]),
    ).iloc[0]

    return pd.DataFrame(
        [
            {
                "pickup_longitude": pickup_longitude,
                "pickup_latitude": pickup_latitude,
                "dropoff_longitude": dropoff_longitude,
                "dropoff_latitude": dropoff_latitude,
                "passenger_count": int(passenger_count),
                "trip_distance_km": float(distance),
                "hour": int(ride_datetime.hour),
                "day": int(ride_datetime.day),
                "month": int(ride_datetime.month),
                "day_of_week": int(ride_datetime.dayofweek),
                "is_weekend": int(ride_datetime.dayofweek in (5, 6)),
            }
        ],
        columns=FEATURE_COLUMNS,
    )
