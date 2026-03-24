import { useEffect, useState } from "react";

const vehicleAccents = {
  UberX: "#f59e0b",
  UberXL: "#38bdf8",
  SUV: "#34d399",
  Moto: "#fb923c",
};
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

function formatRideDate(isoDate) {
  return new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function normalizePassengerCount(value, maxPassengers) {
  const parsedValue = Number.parseInt(value, 10);
  if (Number.isNaN(parsedValue)) {
    return 1;
  }
  return Math.min(Math.max(parsedValue, 1), maxPassengers);
}

function getPassengerCountError(value, maxPassengers, vehicleType) {
  const parsedValue = Number.parseInt(value, 10);
  if (value === "") {
    return "Passenger count is required.";
  }
  if (Number.isNaN(parsedValue)) {
    return "Passenger count must be a number.";
  }
  if (parsedValue < 1) {
    return "Passenger count must be at least 1.";
  }
  if (parsedValue > maxPassengers) {
    return `${vehicleType} allows maximum ${maxPassengers} passenger${maxPassengers > 1 ? "s" : ""}.`;
  }
  return "";
}

async function fetchJson(path, options) {
  const response = await fetch(`${apiBaseUrl}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [timeState, setTimeState] = useState({ timeOptions: [], defaultTimeIndex: 0 });
  const [form, setForm] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("form");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadMeta() {
      try {
        const data = await fetchJson("/api/meta");
        const locations = Object.keys(data.locations);
        const vehicles = Object.keys(data.vehicles);
        setMeta(data);
        setTimeState({
          timeOptions: data.timeOptions,
          defaultTimeIndex: data.defaultTimeIndex,
        });
        setForm({
          pickupLocation: locations[2] ?? locations[0] ?? "",
          dropoffLocation: locations[7] ?? locations[1] ?? "",
          vehicleType: vehicles[0] ?? "",
          passengerCount: 1,
          rideDate: data.today,
          rideTime: data.timeOptions[data.defaultTimeIndex] ?? data.timeOptions[0] ?? "",
        });
      } catch (loadError) {
        setError("Unable to load app data. Start `python main.py` and refresh.");
      } finally {
        setLoading(false);
      }
    }

    loadMeta();
  }, []);

  useEffect(() => {
    if (!form?.rideDate) {
      return;
    }

    async function loadTimeOptions() {
      try {
        const data = await fetchJson(`/api/time-options?rideDate=${form.rideDate}`);
        const nextRideTime = data.timeOptions[data.defaultTimeIndex] ?? data.timeOptions[0] ?? "";
        setTimeState(data);
        setForm((current) => {
          if (!current) {
            return current;
          }
          const shouldKeepSelected = data.timeOptions.includes(current.rideTime);
          return {
            ...current,
            rideTime: shouldKeepSelected ? current.rideTime : nextRideTime,
          };
        });
      } catch (loadError) {
        setError("Unable to load pickup time options.");
      }
    }

    loadTimeOptions();
  }, [form?.rideDate]);

  useEffect(() => {
    function handlePopState() {
      const nextView = window.location.hash === "#result" ? "result" : "form";
      setView(nextView);
    }

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  if (loading || !form || !meta) {
    return (
      <div className="status-shell">
        <div className="status-card">
          <div>Loading fare engine...</div>
          {error ? <div className="status-error">{error}</div> : null}
          <div className="status-hint">Backend URL: {apiBaseUrl}</div>
        </div>
      </div>
    );
  }

  const vehicleConfig = meta.vehicles[form.vehicleType];
  const maxPassengers = vehicleConfig.max_passengers;
  const passengerCountError = getPassengerCountError(
    form.passengerCount,
    maxPassengers,
    form.vehicleType,
  );

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (passengerCountError) {
      setError(passengerCountError);
      return;
    }

    setSubmitting(true);

    try {
      const data = await fetchJson("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...form,
          passengerCount: Number.parseInt(form.passengerCount, 10),
        }),
      });
      setResult(data);
      setView("result");
      window.history.pushState({ view: "result" }, "", "#result");
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  function updateField(field, value) {
    setForm((current) => ({
      ...current,
      [field]: value,
      ...(field === "vehicleType" ? { passengerCount: 1 } : {}),
    }));
  }

  function handlePassengerCountChange(event) {
    const nextValue = event.target.value;
    setForm((current) => ({
      ...current,
      passengerCount: nextValue,
    }));
  }

  function handlePassengerCountBlur() {
    setForm((current) => {
      if (getPassengerCountError(current.passengerCount, maxPassengers, current.vehicleType)) {
        return current;
      }
      return {
        ...current,
        passengerCount: String(normalizePassengerCount(current.passengerCount, maxPassengers)),
      };
    });
  }

  function handleBackToForm() {
    if (window.location.hash === "#result") {
      window.history.back();
      return;
    }
    setView("form");
  }

  return (
    <div className="page-shell">
      <div className={view === "result" ? "page-grid result-view" : "page-grid"}>
        {view === "form" ? (
          <section className="hero-panel">
            <span className="eyebrow">React Frontend</span>
            <h1>Uber Fleet Management System</h1>
            <p>
              Same fare prediction workflow, now split into a Python API and a React
              interface.
            </p>
            <div className="chip-row">
              <span className="chip">Best model: {meta.modelName}</span>
              <span className="chip">City rides only</span>
            </div>
          </section>
        ) : null}

        {view === "form" ? (
          <section className="form-panel">
            <div className="panel-header">Ride Details</div>
            <form className="ride-form" onSubmit={handleSubmit}>
              <label>
                Pickup Location
                <select
                  value={form.pickupLocation}
                  onChange={(event) => updateField("pickupLocation", event.target.value)}
                >
                  {Object.keys(meta.locations).map((location) => (
                    <option key={location} value={location}>
                      {location}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Dropoff Location
                <select
                  value={form.dropoffLocation}
                  onChange={(event) => updateField("dropoffLocation", event.target.value)}
                >
                  {Object.keys(meta.locations).map((location) => (
                    <option key={location} value={location}>
                      {location}
                    </option>
                  ))}
                </select>
              </label>

              <div>
                <div className="field-label">Vehicle Type</div>
                <div className="vehicle-grid">
                  {Object.keys(meta.vehicles).map((vehicle) => (
                    <button
                      key={vehicle}
                      className={vehicle === form.vehicleType ? "vehicle-card active" : "vehicle-card"}
                      style={{ "--vehicle-accent": vehicleAccents[vehicle] }}
                      type="button"
                      onClick={() => updateField("vehicleType", vehicle)}
                    >
                      <span>{vehicle}</span>
                      <small>Max {meta.vehicles[vehicle].max_passengers}</small>
                    </button>
                  ))}
                </div>
              </div>

              <label>
                Passenger Count
                <input
                  type="number"
                  min="1"
                  max={maxPassengers}
                  value={form.passengerCount}
                  onChange={handlePassengerCountChange}
                  onBlur={handlePassengerCountBlur}
                />
              </label>
              {passengerCountError ? <div className="field-error">{passengerCountError}</div> : null}

              <div className="split-fields">
                <label>
                  Ride Date
                  <input
                    type="date"
                    min={meta.today}
                    value={form.rideDate}
                    onChange={(event) => updateField("rideDate", event.target.value)}
                  />
                </label>

                <label>
                  Pickup Time
                  <select
                    value={form.rideTime}
                    onChange={(event) => updateField("rideTime", event.target.value)}
                  >
                    {timeState.timeOptions.map((timeOption) => (
                      <option key={timeOption} value={timeOption}>
                        {timeOption}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <button className="submit-button" type="submit" disabled={submitting}>
                {submitting ? "Predicting..." : "Predict Fare"}
              </button>
            </form>
            {error ? <div className="error-banner">{error}</div> : null}
          </section>
        ) : result ? (
          <>
            <section className="result-panel">
              <button className="back-link" type="button" onClick={handleBackToForm}>
                Back
              </button>
              <div className="panel-header">Estimated Fare</div>
              <div className="fare-value">${result.predictedFare.toFixed(2)}</div>
              <div className="result-grid">
                <article className="result-card">
                  <span>Passengers</span>
                  <strong>{result.passengerCount}</strong>
                </article>
                <article className="result-card">
                  <span>Distance</span>
                  <strong>{result.tripDistanceKm} km</strong>
                </article>
                <article className="result-card">
                  <span>Pickup Time</span>
                  <strong>{result.rideTime}</strong>
                </article>
              </div>
              <div className="summary-card">
                <div>{result.pickupLocation}</div>
                <div>{result.dropoffLocation}</div>
                <div>{formatRideDate(result.rideDate)}</div>
              </div>
            </section>

            <section className="map-panel">
              <div className="panel-header">Trip Route Map</div>
              <a className="map-link" href={result.directionsLink}>
                Open Route In Google Maps
              </a>
              <iframe
                title="Trip route map"
                src={result.mapEmbedUrl}
                className="route-map"
                loading="lazy"
              />
            </section>
          </>
        ) : (
          <section className="empty-panel">
            <div className="panel-header">Prediction Output</div>
            <p>Choose ride details and run a fare prediction to see trip summary and route.</p>
          </section>
        )}
      </div>
    </div>
  );
}
