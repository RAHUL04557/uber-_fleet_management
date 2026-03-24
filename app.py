from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import build_prediction_frame


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
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
    "UberX": {
        "multiplier": 1.0,
        "max_passengers": 4,
        "svg": """
            <svg xmlns="http://www.w3.org/2000/svg" width="420" height="220" viewBox="0 0 420 220">
              <rect width="420" height="220" rx="26" fill="#f3f4f6"/>
              <rect x="88" y="102" width="210" height="42" rx="14" fill="#111827"/>
              <path d="M126 102 L160 74 H240 L274 102 Z" fill="#4b5563"/>
              <circle cx="132" cy="152" r="24" fill="#111827"/>
              <circle cx="258" cy="152" r="24" fill="#111827"/>
              <circle cx="132" cy="152" r="11" fill="#d1d5db"/>
              <circle cx="258" cy="152" r="11" fill="#d1d5db"/>
              <text x="26" y="42" font-size="30" font-family="Arial" fill="#111827">UberX</text>
            </svg>
        """,
    },
    "UberXL": {
        "multiplier": 1.25,
        "max_passengers": 6,
        "svg": """
            <svg xmlns="http://www.w3.org/2000/svg" width="420" height="220" viewBox="0 0 420 220">
              <rect width="420" height="220" rx="26" fill="#eff6ff"/>
              <rect x="74" y="100" width="236" height="44" rx="14" fill="#1e3a8a"/>
              <path d="M116 100 L164 68 H256 L296 100 Z" fill="#60a5fa"/>
              <circle cx="126" cy="154" r="24" fill="#0f172a"/>
              <circle cx="270" cy="154" r="24" fill="#0f172a"/>
              <circle cx="126" cy="154" r="11" fill="#dbeafe"/>
              <circle cx="270" cy="154" r="11" fill="#dbeafe"/>
              <text x="26" y="42" font-size="30" font-family="Arial" fill="#1e3a8a">UberXL</text>
            </svg>
        """,
    },
    "SUV": {
        "multiplier": 1.3,
        "max_passengers": 4,
        "svg": """
            <svg xmlns="http://www.w3.org/2000/svg" width="420" height="220" viewBox="0 0 420 220">
              <rect width="420" height="220" rx="26" fill="#ecfdf5"/>
              <rect x="66" y="92" width="252" height="52" rx="14" fill="#166534"/>
              <path d="M114 92 L158 58 H276 L318 92 Z" fill="#22c55e"/>
              <circle cx="122" cy="154" r="25" fill="#14532d"/>
              <circle cx="282" cy="154" r="25" fill="#14532d"/>
              <circle cx="122" cy="154" r="11" fill="#dcfce7"/>
              <circle cx="282" cy="154" r="11" fill="#dcfce7"/>
              <text x="26" y="42" font-size="30" font-family="Arial" fill="#166534">SUV</text>
            </svg>
        """,
    },
    "Moto": {
        "multiplier": 0.75,
        "max_passengers": 1,
        "svg": """
            <svg xmlns="http://www.w3.org/2000/svg" width="420" height="220" viewBox="0 0 420 220">
              <rect width="420" height="220" rx="26" fill="#fff7ed"/>
              <circle cx="124" cy="152" r="25" fill="#111827"/>
              <circle cx="262" cy="152" r="25" fill="#111827"/>
              <circle cx="124" cy="152" r="11" fill="#fed7aa"/>
              <circle cx="262" cy="152" r="11" fill="#fed7aa"/>
              <path d="M124 152 L164 112 L222 112 L262 152" fill="none" stroke="#ea580c" stroke-width="10" stroke-linecap="round"/>
              <path d="M182 84 L210 112 L184 112" fill="none" stroke="#ea580c" stroke-width="10" stroke-linecap="round"/>
              <path d="M164 112 L148 84" fill="none" stroke="#ea580c" stroke-width="10" stroke-linecap="round"/>
              <text x="26" y="42" font-size="30" font-family="Arial" fill="#c2410c">Moto</text>
            </svg>
        """,
    },
}


def build_time_options():
    options = []
    for hour in range(24):
        for minute in (0, 30):
            current_time = time(hour, minute)
            options.append(current_time.strftime("%I:%M %p"))
    return options


def get_available_time_options(selected_date):
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


st.set_page_config(page_title="Uber Fleet Management System", page_icon="🚕")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 184, 0, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 200, 150, 0.08), transparent 24%),
            linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card, .panel-card, .result-card {
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(12px);
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.1;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .chip {
        display: inline-block;
        padding: 0.38rem 0.8rem;
        margin: 0.2rem 0.4rem 0.2rem 0;
        border-radius: 999px;
        background: rgba(255, 184, 0, 0.16);
        color: #fde68a;
        font-size: 0.86rem;
        font-weight: 600;
    }
    .section-label {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .mini-note {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .fare-text {
        font-size: 2rem;
        font-weight: 800;
        color: #86efac;
        margin: 0.2rem 0 0.8rem 0;
    }
    .stat-card {
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.55);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        min-height: 110px;
    }
    .stat-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
        white-space: nowrap;
    }
    .stat-value {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_models.py` first.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
model_name = artifact["model_name"]

left_hero, right_hero = st.columns([1.6, 1])
with left_hero:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Uber Fleet Management System</div>
            <p class="hero-subtitle">Smart fare prediction for city rides using the best trained machine learning model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right_hero:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Best Model</div>
            <div class="mini-note">Random Forest based fare prediction deployed for ride estimation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

form_col, info_col = st.columns([1.35, 0.8], gap="large")

with form_col:
    with st.container(border=True):
        st.markdown("<div class='section-label'>Ride Details</div>", unsafe_allow_html=True)

        pickup_location = st.selectbox(
            "Pickup Location",
            options=list(LOCATION_COORDINATES.keys()),
            index=2,
        )
        dropoff_location = st.selectbox(
            "Dropoff Location",
            options=list(LOCATION_COORDINATES.keys()),
            index=7,
        )
        st.markdown("<div class='section-label' style='margin-top:0.9rem;'>Choose Vehicle Type</div>", unsafe_allow_html=True)
        vehicle_type = st.radio(
            "Vehicle Type",
            options=list(VEHICLE_OPTIONS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        max_passengers = VEHICLE_OPTIONS[vehicle_type]["max_passengers"]
        passenger_count = st.number_input(
            "Passenger Count",
            min_value=1,
            max_value=max_passengers,
            value=1,
            step=1,
            help=f"Maximum passengers allowed for {vehicle_type}: {max_passengers}",
        )

        date_col, time_col = st.columns(2)
        today = datetime.now().date()
        with date_col:
            ride_date = st.date_input("Ride Date", value=today, min_value=today)
        time_options, default_time_index = get_available_time_options(ride_date)
        with time_col:
            ride_time_label = st.selectbox(
                "Pickup Time", options=time_options, index=default_time_index
            )

        predict = st.button("Predict Fare", use_container_width=True)

with info_col:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-label">Selected Ride Rules</div>
            <div class="mini-note">Vehicle: <strong>{vehicle_type}</strong></div>
            <div class="mini-note">Max passengers allowed: <strong>{max_passengers}</strong></div>
            <div class="mini-note">Pickup date must be today or a future date.</div>
            <div class="mini-note">Pickup time is shown in AM/PM format.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">What This App Does</div>
            <div class="mini-note">It estimates fare using trip distance, coordinates, passenger count, and time-based features.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if predict:
    pickup_longitude, pickup_latitude = LOCATION_COORDINATES[pickup_location]
    dropoff_longitude, dropoff_latitude = LOCATION_COORDINATES[dropoff_location]
    ride_time = datetime.strptime(ride_time_label, "%I:%M %p").time()
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

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    result_left, result_right = st.columns([1.1, 0.9], gap="large")
    with result_left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-label">Estimated Fare</div>
                <div class="fare-text">${predicted_fare:.2f}</div>
                <div class="mini-note">Vehicle type: {vehicle_type}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Passengers</div>
                    <div class="stat-value">{passenger_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with metric_col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Distance</div>
                    <div class="stat-value">{trip_distance:.2f} km</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with metric_col3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Pickup Time</div>
                    <div class="stat-value">{ride_time_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="panel-card" style="margin-top: 1rem;">
                <div class="section-label">Ride Summary</div>
                <div class="mini-note">Pickup: <strong>{pickup_location}</strong></div>
                <div class="mini-note">Dropoff: <strong>{dropoff_location}</strong></div>
                <div class="mini-note">Ride Date: <strong>{ride_date}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    with result_right:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Trip Route Map</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("Open Route In Google Maps", directions_link, use_container_width=True)
        components.iframe(embed_src, height=470, scrolling=False)
