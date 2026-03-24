from __future__ import annotations

import json
import os
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

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

APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Uber Fleet Management System</title>
  <style>
    :root {
      font-family: "Segoe UI", sans-serif;
      color: #f7f4ea;
      background:
        radial-gradient(circle at top left, rgba(252, 211, 77, 0.20), transparent 26%),
        radial-gradient(circle at 85% 20%, rgba(34, 197, 94, 0.14), transparent 22%),
        linear-gradient(160deg, #121212 0%, #1d1a17 48%, #0f2a24 100%);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: transparent; color: inherit; }
    button, input, select { font: inherit; }
    .page-shell { min-height: 100vh; padding: 32px 20px 48px; }
    .page-grid {
      width: min(1220px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 20px;
    }
    .page-grid.result-view { align-items: start; }
    .hero-panel, .form-panel, .comparison-panel, .result-panel, .map-panel, .empty-panel, .status-card {
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(8, 10, 12, 0.58);
      border-radius: 28px;
      padding: 24px;
      backdrop-filter: blur(16px);
      box-shadow: 0 30px 80px rgba(0, 0, 0, 0.28);
    }
    .status-shell { min-height: 100vh; display: grid; place-items: center; padding: 20px; }
    .status-card { text-align: center; max-width: 640px; }
    .status-error { margin-top: 12px; color: #ffd6d6; }
    .status-hint { margin-top: 8px; color: #c7c1b4; font-size: 0.92rem; }
    .eyebrow, .panel-header, .field-label {
      color: #f4c95d;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.75rem;
      font-weight: 700;
    }
    .hero-panel h1 {
      margin: 10px 0 12px;
      font-size: clamp(2.6rem, 4vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero-panel p, .empty-panel p {
      margin: 0;
      color: #d5d0c3;
      font-size: 1rem;
      line-height: 1.7;
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
    .chip {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      color: #f8e7b6;
    }
    .ride-form, .comparison-table, .result-grid { display: grid; gap: 14px; }
    .ride-form { margin-top: 18px; gap: 16px; }
    .ride-form label { display: grid; gap: 8px; color: #e7e2d7; font-size: 0.95rem; }
    .ride-form input, .ride-form select {
      width: 100%;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: #f7f4ea;
      outline: none;
    }
    .ride-form select option { color: #141210; background: #f5efe2; }
    .ride-form input:focus, .ride-form select:focus { border-color: rgba(244, 201, 93, 0.7); }
    .split-fields, .vehicle-grid, .result-grid { display: grid; gap: 16px; }
    .split-fields { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .vehicle-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 10px; gap: 12px; }
    .vehicle-card {
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: #f7f4ea;
      text-align: left;
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }
    .vehicle-card span, .vehicle-card small { display: block; }
    .vehicle-card small { margin-top: 6px; color: #b9c0bc; }
    .vehicle-card.active { transform: translateY(-1px); }
    .vehicle-card.uberx.active { border-color: #f59e0b; background: rgba(245, 158, 11, 0.18); }
    .vehicle-card.uberxl.active { border-color: #38bdf8; background: rgba(56, 189, 248, 0.18); }
    .vehicle-card.suv.active { border-color: #34d399; background: rgba(52, 211, 153, 0.18); }
    .vehicle-card.moto.active { border-color: #fb923c; background: rgba(251, 146, 60, 0.18); }
    .submit-button, .map-link, .back-link {
      display: inline-flex;
      justify-content: center;
      align-items: center;
      min-height: 52px;
      border: 0;
      border-radius: 18px;
      background: linear-gradient(135deg, #f4c95d 0%, #ff914d 100%);
      color: #141210;
      font-weight: 700;
      text-decoration: none;
      padding: 0 18px;
      cursor: pointer;
    }
    .submit-button:disabled { opacity: 0.7; cursor: wait; }
    .error-banner, .field-error {
      color: #ffd6d6;
      background: rgba(153, 27, 27, 0.35);
      border: 1px solid rgba(248, 113, 113, 0.35);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .field-error { padding: 10px 14px; font-size: 0.9rem; }
    .comparison-row {
      display: grid;
      grid-template-columns: 1.6fr 0.7fr 0.7fr 0.6fr;
      gap: 12px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
      color: #ece6d8;
    }
    .comparison-head {
      background: transparent;
      border: 0;
      color: #f4c95d;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.78rem;
      padding: 0 4px 8px;
    }
    .comparison-best { border-color: rgba(244, 201, 93, 0.4); background: rgba(244, 201, 93, 0.09); }
    .fare-value {
      margin: 12px 0 20px;
      font-size: clamp(2.4rem, 4vw, 4rem);
      color: #8af1b8;
      letter-spacing: -0.04em;
    }
    .result-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .result-card, .summary-card {
      border-radius: 20px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .result-card span { color: #b9c0bc; font-size: 0.9rem; }
    .result-card strong { display: block; margin-top: 8px; font-size: 1.5rem; }
    .summary-card { display: grid; gap: 8px; margin-top: 16px; color: #f1ede4; }
    .route-map {
      width: 100%;
      min-height: 520px;
      border: 0;
      margin-top: 16px;
      border-radius: 22px;
    }
    .empty-panel { display: grid; align-content: start; min-height: 220px; }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      .page-grid { grid-template-columns: 1fr; }
      .split-fields, .vehicle-grid, .result-grid, .comparison-row { grid-template-columns: 1fr; }
      .comparison-head { display: none; }
      .route-map { min-height: 420px; }
    }
  </style>
</head>
<body>
  <div id="loading" class="status-shell">
    <div class="status-card">
      <div>Loading fare engine...</div>
      <div id="loading-error" class="status-error hidden"></div>
      <div class="status-hint">Backend URL: same service</div>
    </div>
  </div>

  <main id="app" class="page-shell hidden">
    <div id="page-grid" class="page-grid">
      <section id="hero-panel" class="hero-panel">
        <span class="eyebrow">Flask App</span>
        <h1>Uber Fleet Management System</h1>
        <p>Same fare prediction workflow, now served directly from a single Python app.</p>
        <div class="chip-row">
          <span id="best-model-chip" class="chip"></span>
          <span class="chip">City rides only</span>
        </div>
      </section>

      <section id="form-panel" class="form-panel">
        <div class="panel-header">Ride Details</div>
        <form id="ride-form" class="ride-form">
          <label>
            Pickup Location
            <select id="pickup-location"></select>
          </label>

          <label>
            Dropoff Location
            <select id="dropoff-location"></select>
          </label>

          <div>
            <div class="field-label">Vehicle Type</div>
            <div id="vehicle-grid" class="vehicle-grid"></div>
          </div>

          <label>
            Passenger Count
            <input id="passenger-count" type="number" min="1" />
          </label>
          <div id="field-error" class="field-error hidden"></div>

          <div class="split-fields">
            <label>
              Ride Date
              <input id="ride-date" type="date" />
            </label>

            <label>
              Pickup Time
              <select id="ride-time"></select>
            </label>
          </div>

          <button id="submit-button" class="submit-button" type="submit">Predict Fare</button>
        </form>
        <div id="error-banner" class="error-banner hidden"></div>
      </section>

      <section id="comparison-panel" class="comparison-panel">
        <div class="panel-header">Model Comparison</div>
        <div id="comparison-table" class="comparison-table"></div>
      </section>

      <section id="result-panel" class="result-panel hidden">
        <button id="back-button" class="back-link" type="button">Back</button>
        <div class="panel-header">Estimated Fare</div>
        <div id="fare-value" class="fare-value"></div>
        <div class="result-grid">
          <article class="result-card">
            <span>Passengers</span>
            <strong id="result-passengers"></strong>
          </article>
          <article class="result-card">
            <span>Distance</span>
            <strong id="result-distance"></strong>
          </article>
          <article class="result-card">
            <span>Pickup Time</span>
            <strong id="result-time"></strong>
          </article>
        </div>
        <div id="result-summary" class="summary-card"></div>
      </section>

      <section id="map-panel" class="map-panel hidden">
        <div class="panel-header">Trip Route Map</div>
        <a id="map-link" class="map-link" href="#">Open Route In Google Maps</a>
        <iframe id="route-map" class="route-map" loading="lazy"></iframe>
      </section>

      <section id="empty-panel" class="empty-panel">
        <div class="panel-header">Prediction Output</div>
        <p>Choose ride details and run a fare prediction to see trip summary and route.</p>
      </section>
    </div>
  </main>

  <script>
    const state = {
      meta: null,
      form: null,
      result: null,
      timeOptions: [],
      view: window.location.hash === "#result" ? "result" : "form",
    };

    const els = {
      loading: document.getElementById("loading"),
      loadingError: document.getElementById("loading-error"),
      app: document.getElementById("app"),
      pageGrid: document.getElementById("page-grid"),
      heroPanel: document.getElementById("hero-panel"),
      formPanel: document.getElementById("form-panel"),
      comparisonPanel: document.getElementById("comparison-panel"),
      resultPanel: document.getElementById("result-panel"),
      mapPanel: document.getElementById("map-panel"),
      emptyPanel: document.getElementById("empty-panel"),
      bestModelChip: document.getElementById("best-model-chip"),
      pickupLocation: document.getElementById("pickup-location"),
      dropoffLocation: document.getElementById("dropoff-location"),
      vehicleGrid: document.getElementById("vehicle-grid"),
      passengerCount: document.getElementById("passenger-count"),
      rideDate: document.getElementById("ride-date"),
      rideTime: document.getElementById("ride-time"),
      rideForm: document.getElementById("ride-form"),
      submitButton: document.getElementById("submit-button"),
      fieldError: document.getElementById("field-error"),
      errorBanner: document.getElementById("error-banner"),
      comparisonTable: document.getElementById("comparison-table"),
      fareValue: document.getElementById("fare-value"),
      resultPassengers: document.getElementById("result-passengers"),
      resultDistance: document.getElementById("result-distance"),
      resultTime: document.getElementById("result-time"),
      resultSummary: document.getElementById("result-summary"),
      mapLink: document.getElementById("map-link"),
      routeMap: document.getElementById("route-map"),
      backButton: document.getElementById("back-button"),
    };

    function getPassengerCountError() {
      const maxPassengers = state.meta.vehicles[state.form.vehicleType].max_passengers;
      const value = state.form.passengerCount;
      const parsed = Number.parseInt(value, 10);
      if (value === "") return "Passenger count is required.";
      if (Number.isNaN(parsed)) return "Passenger count must be a number.";
      if (parsed < 1) return "Passenger count must be at least 1.";
      if (parsed > maxPassengers) {
        return state.form.vehicleType + " allows maximum " + maxPassengers + " passenger" + (maxPassengers > 1 ? "s." : ".");
      }
      return "";
    }

    function formatRideDate(isoDate) {
      return new Date(isoDate + "T00:00:00").toLocaleDateString("en-IN", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }

    async function fetchJson(path, options) {
      const response = await fetch(path, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || ("Request failed: " + response.status));
      return data;
    }

    function setHidden(element, hidden) {
      element.classList.toggle("hidden", hidden);
    }

    function renderLocations() {
      const locations = Object.keys(state.meta.locations);
      els.pickupLocation.innerHTML = locations.map((location) =>
        `<option value="${location}">${location}</option>`
      ).join("");
      els.dropoffLocation.innerHTML = locations.map((location) =>
        `<option value="${location}">${location}</option>`
      ).join("");
      els.pickupLocation.value = state.form.pickupLocation;
      els.dropoffLocation.value = state.form.dropoffLocation;
    }

    function renderVehicles() {
      const classes = { UberX: "uberx", UberXL: "uberxl", SUV: "suv", Moto: "moto" };
      els.vehicleGrid.innerHTML = Object.keys(state.meta.vehicles).map((vehicle) => {
        const active = vehicle === state.form.vehicleType ? "active" : "";
        const vehicleClass = classes[vehicle] || "";
        return `
          <button class="vehicle-card ${vehicleClass} ${active}" type="button" data-vehicle="${vehicle}">
            <span>${vehicle}</span>
            <small>Max ${state.meta.vehicles[vehicle].max_passengers}</small>
          </button>
        `;
      }).join("");
      els.vehicleGrid.querySelectorAll("[data-vehicle]").forEach((button) => {
        button.addEventListener("click", () => {
          state.form.vehicleType = button.dataset.vehicle;
          state.form.passengerCount = "1";
          renderVehicles();
          renderPassengerState();
        });
      });
    }

    function renderTimes() {
      els.rideTime.innerHTML = state.timeOptions.map((timeOption) =>
        `<option value="${timeOption}">${timeOption}</option>`
      ).join("");
      if (!state.timeOptions.includes(state.form.rideTime)) {
        state.form.rideTime = state.timeOptions[0] || "";
      }
      els.rideTime.value = state.form.rideTime;
    }

    function renderComparison() {
      const rows = [
        '<div class="comparison-row comparison-head"><span>Model</span><span>MAE</span><span>RMSE</span><span>R2</span></div>',
        ...state.meta.metrics.map((item) => {
          const best = item.model === state.meta.modelName ? "comparison-best" : "";
          return `
            <div class="comparison-row ${best}">
              <span>${item.model}</span>
              <span>${item.mae}</span>
              <span>${item.rmse}</span>
              <span>${item.r2}</span>
            </div>
          `;
        }),
      ];
      els.comparisonTable.innerHTML = rows.join("");
    }

    function renderPassengerState() {
      const maxPassengers = state.meta.vehicles[state.form.vehicleType].max_passengers;
      els.passengerCount.max = String(maxPassengers);
      els.passengerCount.value = state.form.passengerCount;
      const message = getPassengerCountError();
      setHidden(els.fieldError, !message);
      els.fieldError.textContent = message;
    }

    function renderView() {
      const resultView = state.view === "result" && state.result;
      els.pageGrid.classList.toggle("result-view", Boolean(resultView));
      setHidden(els.heroPanel, Boolean(resultView));
      setHidden(els.formPanel, Boolean(resultView));
      setHidden(els.comparisonPanel, Boolean(resultView));
      setHidden(els.resultPanel, !resultView);
      setHidden(els.mapPanel, !resultView);
      setHidden(els.emptyPanel, Boolean(resultView));
      if (resultView) {
        els.fareValue.textContent = "$" + state.result.predictedFare.toFixed(2);
        els.resultPassengers.textContent = state.result.passengerCount;
        els.resultDistance.textContent = state.result.tripDistanceKm + " km";
        els.resultTime.textContent = state.result.rideTime;
        els.resultSummary.innerHTML = `
          <div>${state.result.pickupLocation}</div>
          <div>${state.result.dropoffLocation}</div>
          <div>${formatRideDate(state.result.rideDate)}</div>
        `;
        els.mapLink.href = state.result.directionsLink;
        els.routeMap.src = state.result.mapEmbedUrl;
      }
    }

    async function loadTimeOptions() {
      const data = await fetchJson("/api/time-options?rideDate=" + encodeURIComponent(state.form.rideDate));
      state.timeOptions = data.timeOptions;
      if (!state.timeOptions.includes(state.form.rideTime)) {
        state.form.rideTime = state.timeOptions[data.defaultTimeIndex] || state.timeOptions[0] || "";
      }
      renderTimes();
    }

    async function initialize() {
      try {
        const data = await fetchJson("/api/meta");
        const locations = Object.keys(data.locations);
        const vehicles = Object.keys(data.vehicles);
        state.meta = data;
        state.timeOptions = data.timeOptions;
        state.form = {
          pickupLocation: locations[2] || locations[0] || "",
          dropoffLocation: locations[7] || locations[1] || "",
          vehicleType: vehicles[0] || "",
          passengerCount: "1",
          rideDate: data.today,
          rideTime: data.timeOptions[data.defaultTimeIndex] || data.timeOptions[0] || "",
        };

        els.bestModelChip.textContent = "Best model: " + state.meta.modelName;
        els.rideDate.min = data.today;
        els.rideDate.value = state.form.rideDate;
        renderLocations();
        renderVehicles();
        renderTimes();
        renderComparison();
        renderPassengerState();
        renderView();

        setHidden(els.loading, true);
        setHidden(els.app, false);
      } catch (error) {
        els.loadingError.textContent = error.message;
        setHidden(els.loadingError, false);
      }
    }

    els.pickupLocation.addEventListener("change", (event) => { state.form.pickupLocation = event.target.value; });
    els.dropoffLocation.addEventListener("change", (event) => { state.form.dropoffLocation = event.target.value; });
    els.rideDate.addEventListener("change", async (event) => {
      state.form.rideDate = event.target.value;
      await loadTimeOptions();
    });
    els.rideTime.addEventListener("change", (event) => { state.form.rideTime = event.target.value; });
    els.passengerCount.addEventListener("input", (event) => {
      state.form.passengerCount = event.target.value;
      renderPassengerState();
    });

    els.rideForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const passengerError = getPassengerCountError();
      setHidden(els.errorBanner, true);
      els.errorBanner.textContent = "";
      if (passengerError) {
        els.errorBanner.textContent = passengerError;
        setHidden(els.errorBanner, false);
        return;
      }

      els.submitButton.disabled = true;
      els.submitButton.textContent = "Predicting...";
      try {
        state.result = await fetchJson("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...state.form,
            passengerCount: Number.parseInt(state.form.passengerCount, 10),
          }),
        });
        state.view = "result";
        history.pushState({ view: "result" }, "", "#result");
        renderView();
      } catch (error) {
        els.errorBanner.textContent = error.message;
        setHidden(els.errorBanner, false);
      } finally {
        els.submitButton.disabled = false;
        els.submitButton.textContent = "Predict Fare";
      }
    });

    els.backButton.addEventListener("click", () => {
      if (window.location.hash === "#result") {
        history.back();
      } else {
        state.view = "form";
        renderView();
      }
    });

    window.addEventListener("popstate", () => {
      state.view = window.location.hash === "#result" && state.result ? "result" : "form";
      renderView();
    });

    initialize();
  </script>
</body>
</html>
"""


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


@app.route("/", methods=["GET"])
def index():
    return render_template_string(APP_HTML)


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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
