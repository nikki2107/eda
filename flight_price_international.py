"""
Review 2 - International Extension
Flight Price Prediction for DOMESTIC + INTERNATIONAL routes, with live
weather and AUTOMATIC crisis detection (news based).

HOW IT WORKS (say this clearly in your review):
  * Domestic routes (both cities among the 6 Indian metros in the dataset):
      price comes from the trained Random Forest, same as before.
  * International routes (any foreign city): the dataset has NO foreign
      fares, so the Random Forest cannot predict them. Instead we use a
      HYBRID estimator:
          base fare  = assumed fixed fee + assumed rate per km * distance
          booking-time effect = learned from the Random Forest's
                                days_left curve on real domestic data
      The booking-time curve is data-driven; the absolute fare level is an
      ASSUMPTION (labelled as such in the output). With a real international
      fare dataset, this layer could be replaced by a trained model.
  * Weather: live OpenWeatherMap data for source AND destination city.
  * Crisis: live news headlines for the country of each city, scanned for
      crisis keywords (keyword-based, auditable, not a deep-learning model).

Requires:
  - flight_price_cleaned.csv (same folder, from flight_price_eda.py)
  - Free OpenWeatherMap key: https://openweathermap.org/api
  - Free NewsAPI.org key: https://newsapi.org/register
  - pip3 install requests scikit-learn pandas numpy

Run: python3 flight_price_international.py
"""

import math
import re
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
OPENWEATHER_API_KEY = input("Enter your OpenWeatherMap API key: ").strip()
NEWSAPI_KEY = input("Enter your NewsAPI.org API key: ").strip()

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
NEWS_URL = "https://newsapi.org/v2/everything"

# city: (country, latitude, longitude, OpenWeatherMap query)
CITIES = {
    # --- India (the 6 cities in the dataset) ---
    "Delhi": ("India", 28.61, 77.21, "Delhi,IN"),
    "Mumbai": ("India", 19.08, 72.88, "Mumbai,IN"),
    "Bangalore": ("India", 12.97, 77.59, "Bangalore,IN"),
    "Kolkata": ("India", 22.57, 88.36, "Kolkata,IN"),
    "Hyderabad": ("India", 17.39, 78.49, "Hyderabad,IN"),
    "Chennai": ("India", 13.08, 80.27, "Chennai,IN"),
    # --- Foreign cities (NOT in the training dataset) ---
    "Kathmandu": ("Nepal", 27.72, 85.32, "Kathmandu,NP"),
    "Colombo": ("Sri Lanka", 6.93, 79.86, "Colombo,LK"),
    "Dhaka": ("Bangladesh", 23.81, 90.41, "Dhaka,BD"),
    "Dubai": ("UAE", 25.20, 55.27, "Dubai,AE"),
    "Doha": ("Qatar", 25.29, 51.53, "Doha,QA"),
    "Tel Aviv": ("Israel", 32.09, 34.78, "Tel Aviv,IL"),
    "Singapore": ("Singapore", 1.35, 103.82, "Singapore,SG"),
    "Bangkok": ("Thailand", 13.76, 100.50, "Bangkok,TH"),
    "Kuala Lumpur": ("Malaysia", 3.14, 101.69, "Kuala Lumpur,MY"),
    "London": ("United Kingdom", 51.51, -0.13, "London,GB"),
    "New York": ("United States", 40.71, -74.01, "New York,US"),
}

# ASSUMED benchmark fares for the international estimator (economy, INR).
# These are NOT learned from data - the dataset has no international fares.
INTL_FIXED_FEE = 3500
INTL_RATE_PER_KM = 5.5
BUSINESS_MULTIPLIER = 3.0
STOP_PRICE_FACTOR = {"zero": 1.00, "one": 0.92, "two_or_more": 0.85}
CRUISE_SPEED_KMH = 800

# ---------------------------------------------------------
# 2. LOAD DATA + TRAIN PRICE MODEL
# ---------------------------------------------------------
print("\nLoading data and training price model, please wait...\n")

df = pd.read_csv("flight_price_cleaned.csv")

categorical_cols = ["Airline", "Source City", "Departure Time", "Stops",
                     "Arrival Time", "Destination City", "Class"]
categorical_cols = [c for c in categorical_cols if c in df.columns]

model_df = df.copy()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

feature_cols = categorical_cols + ["Duration", "Days_left"]
feature_cols = [c for c in feature_cols if c in model_df.columns]

X = model_df[feature_cols]
y = model_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

DOMESTIC_CITIES = set(df["Source City"].unique()) | set(df["Destination City"].unique())


def build_booking_curves():
    """
    For each class, ask the trained model for the price of a typical
    domestic flight at every days_left value (1-49), then divide by the
    average. Result: a multiplier curve (e.g. 1.4 = 40% above average
    price) showing how booking time moves price in the REAL data.
    """
    curves = {}
    for cls in df["Class"].unique():
        sub = df[df["Class"] == cls]
        ref = {col: sub[col].mode()[0] for col in categorical_cols}
        ref["Duration"] = sub["Duration"].median()
        rows = []
        for d in range(1, 50):
            r = ref.copy()
            r["Days_left"] = d
            rows.append(r)
        sim = pd.DataFrame(rows)
        for col in categorical_cols:
            sim[col] = encoders[col].transform(sim[col].astype(str))
        preds = model.predict(sim[feature_cols])
        curves[cls] = preds / preds.mean()
    return curves


BOOKING_CURVES = build_booking_curves()
print("Model ready.\n")

# ---------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------
def choose_option(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def choose_number(prompt, min_val=None, max_val=None, is_float=False):
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            val = float(raw) if is_float else int(raw)
            if min_val is not None and val < min_val:
                print(f"Must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid number.")


def choose_city(prompt, exclude=None):
    names = [c for c in CITIES if c != exclude]
    labels = [f"{c} ({CITIES[c][0]})" for c in names]
    picked = choose_option(prompt, labels)
    return names[labels.index(picked)]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def norm_stops(stops):
    return str(stops).lower().replace(" ", "_")


# ---------------------------------------------------------
# 4. LIVE WEATHER
# ---------------------------------------------------------
def get_weather(city_name):
    params = {"q": CITIES[city_name][3], "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "visibility": data.get("visibility", 10000),
        }
    except Exception as e:
        print(f"Weather lookup failed for {city_name} ({e}). Using clear defaults.")
        return {"condition": "Clear", "description": "unavailable - defaulted to clear",
                "wind_speed": 3.0, "visibility": 10000}


CONDITION_SEVERITY = {
    "Thunderstorm": 0.9, "Snow": 0.8, "Tornado": 1.0, "Squall": 0.85,
    "Rain": 0.5, "Drizzle": 0.3, "Mist": 0.3, "Fog": 0.5, "Haze": 0.25,
    "Smoke": 0.3, "Dust": 0.35, "Sand": 0.35, "Ash": 0.6,
    "Clouds": 0.1, "Clear": 0.0,
}


def weather_severity(weather):
    base = CONDITION_SEVERITY.get(weather["condition"], 0.2)
    wind_penalty = min(weather["wind_speed"] / 25, 1.0) * 0.3
    visibility_penalty = max(0, (10000 - weather["visibility"]) / 10000) * 0.3
    return round(min(base + wind_penalty + visibility_penalty, 1.0), 3)


# ---------------------------------------------------------
# 5. AUTOMATIC CRISIS DETECTION (live news + keyword scoring)
# ---------------------------------------------------------
CRISIS_KEYWORDS = {
    "war": 0.9, "conflict": 0.8, "invasion": 0.9, "airstrike": 0.9,
    "attack": 0.7, "military operation": 0.8, "ceasefire": 0.6,
    "unrest": 0.5, "militant": 0.6, "bombing": 0.9,
    "flood": 0.7, "flooding": 0.7, "earthquake": 0.8, "cyclone": 0.75,
    "landslide": 0.6, "disaster": 0.6, "evacuate": 0.6, "evacuation": 0.6,
    "pandemic": 0.6, "outbreak": 0.5, "epidemic": 0.55, "quarantine": 0.5,
    "emergency": 0.4, "crisis": 0.45, "state of emergency": 0.7,
}

CONFLICT_WORDS = {"war", "conflict", "invasion", "airstrike", "attack",
                  "military operation", "bombing", "unrest", "militant"}
HEALTH_WORDS = {"pandemic", "outbreak", "epidemic", "quarantine"}


def fetch_crisis_news(region):
    params = {"q": region, "sortBy": "publishedAt", "language": "en",
              "pageSize": 20, "apiKey": NEWSAPI_KEY}
    try:
        resp = requests.get(NEWS_URL, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception as e:
        print(f"News lookup failed for {region} ({e}). Assuming no crisis.")
        return 0.0, [], "None detected (news lookup failed)"

    if not articles:
        return 0.0, [], "None detected (no recent articles found)"

    matched, scores = [], []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".lower()
        for keyword, weight in CRISIS_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                scores.append(weight)
                matched.append((art.get("title", ""), keyword, weight))
                break

    if not scores:
        return 0.0, [], "None detected (no crisis keywords in recent headlines)"

    severity = min(max(scores) + min(len(scores) / 20, 0.15), 1.0)
    top_type = max(matched, key=lambda x: x[2])[1]
    summary = f"Detected '{top_type}'-related coverage ({len(scores)} matching articles)"
    return round(severity, 3), matched[:3], summary


def crisis_adjustment(severity, matched):
    if severity == 0:
        return 1.00, 0.00
    types = {m[1] for m in matched}
    if types & HEALTH_WORDS:
        return round(1.00 - severity * 0.25, 3), round(severity * 0.15, 3)
    return round(1.00 + severity * 0.30, 3), round(severity * 0.30, 3)


# ---------------------------------------------------------
# 6. PRICE ENGINES
# ---------------------------------------------------------
def predict_domestic(user_row):
    """Trained Random Forest (real data, domestic routes only)."""
    encoded = user_row.copy()
    for col in categorical_cols:
        encoded[col] = encoders[col].transform([str(encoded[col])])[0]
    row_df = pd.DataFrame([encoded])[feature_cols]
    tree_preds = np.array([t.predict(row_df)[0] for t in model.estimators_])
    return tree_preds.mean(), np.percentile(tree_preds, 5), np.percentile(tree_preds, 95)


def estimate_international(distance_km, flight_class, stops, days_left):
    """Hybrid: assumed distance-based fare x booking-time curve learned from data."""
    base = INTL_FIXED_FEE + INTL_RATE_PER_KM * distance_km
    if str(flight_class).lower() == "business":
        base *= BUSINESS_MULTIPLIER
    base *= STOP_PRICE_FACTOR.get(norm_stops(stops), 0.95)
    curve = BOOKING_CURVES.get(flight_class)
    booking_factor = float(curve[days_left - 1]) if curve is not None else 1.0
    price = base * booking_factor
    return price, price * 0.70, price * 1.30, booking_factor


def weather_adjusted_price(base_price, severity):
    surcharge = severity * 0.08
    return base_price * (1 + surcharge), surcharge * 100


def delay_risk_probability(severity, stops):
    penalty = {"zero": 0.0, "one": 0.08, "two_or_more": 0.15}.get(norm_stops(stops), 0.05)
    return round(min(0.15 + severity * 0.5 + penalty, 0.95), 3)


# ---------------------------------------------------------
# 7. INTERACTIVE SESSION
# ---------------------------------------------------------
def run_session():
    print("=" * 64)
    print("FLIGHT PRICE PREDICTOR - DOMESTIC + INTERNATIONAL")
    print("=" * 64)

    source = choose_city("Select source city:")
    dest = choose_city("Select destination city:", exclude=source)
    flight_class = choose_option("Select class:", sorted(df["Class"].unique()))
    stops = choose_option("Select number of stops:", sorted(df["Stops"].unique()))
    days_left = choose_number("Enter days left before departure", min_val=1, max_val=49)

    is_domestic = source in DOMESTIC_CITIES and dest in DOMESTIC_CITIES

    src_c, src_lat, src_lon, _ = CITIES[source]
    dst_c, dst_lat, dst_lon, _ = CITIES[dest]
    distance = haversine_km(src_lat, src_lon, dst_lat, dst_lon)

    if is_domestic:
        airline = choose_option("Select airline:", sorted(df["Airline"].unique()))
        dep_time = choose_option("Select departure time:", sorted(df["Departure Time"].unique()))
        arr_time = choose_option("Select arrival time:", sorted(df["Arrival Time"].unique()))
        duration = choose_number("Enter flight duration in hours (e.g. 2.5)",
                                 min_val=0.5, max_val=50, is_float=True)
        user_row = {
            "Airline": airline, "Source City": source, "Departure Time": dep_time,
            "Stops": stops, "Arrival Time": arr_time, "Destination City": dest,
            "Class": flight_class, "Duration": duration, "Days_left": days_left,
        }
        base_price, low, high = predict_domestic(user_row)
        method = "Random Forest trained on real domestic data"
        flight_time = duration
    else:
        flight_time = distance / CRUISE_SPEED_KMH + 0.75
        base_price, low, high, booking_factor = estimate_international(
            distance, flight_class, stops, days_left
        )
        method = ("Hybrid estimate: ASSUMED distance-based fare x booking-time curve "
                  "learned from data")

    # ---- Optional manual override for the crisis region ----
    override = input(
        "\nCrisis check: press Enter to auto-check the route's countries, "
        "or type a region (e.g. Nepal): "
    ).strip()
    if override:
        regions = [override]
    else:
        regions = list(dict.fromkeys([src_c, dst_c]))  # unique, keeps order

    # ---- Live weather (both ends) ----
    print(f"\nFetching live weather for {source} and {dest}...")
    w_src, w_dst = get_weather(source), get_weather(dest)
    sev_src, sev_dst = weather_severity(w_src), weather_severity(w_dst)
    w_severity = max(sev_src, sev_dst)

    # ---- Live crisis news (strongest signal across regions) ----
    print(f"Scanning live news for: {', '.join(regions)} ...")
    best = (0.0, [], "None detected", regions[0])
    for region in regions:
        sev, headlines, summary = fetch_crisis_news(region)
        if sev >= best[0]:
            best = (sev, headlines, summary, region)
    crisis_sev, headlines, crisis_summary, crisis_region = best
    crisis_mult, crisis_delay_add = crisis_adjustment(crisis_sev, headlines)

    # ---- Combine everything ----
    weather_price, surcharge_pct = weather_adjusted_price(base_price, w_severity)
    final_price = weather_price * crisis_mult
    delay_risk = min(delay_risk_probability(w_severity, stops) + crisis_delay_add, 0.95)

    # ---- Output ----
    print("\n" + "-" * 64)
    print("ROUTE")
    print("-" * 64)
    print(f"{source} ({src_c}) -> {dest} ({dst_c}) | {'DOMESTIC' if is_domestic else 'INTERNATIONAL'}")
    print(f"Distance: {distance:.0f} km | Flight time: ~{flight_time:.1f} h | {flight_class}, {stops} stop(s)")
    print(f"Price method: {method}")

    print("\n" + "-" * 64)
    print("LIVE WEATHER")
    print("-" * 64)
    print(f"{source}: {w_src['condition']} ({w_src['description']}) -> severity {sev_src}")
    print(f"{dest}: {w_dst['condition']} ({w_dst['description']}) -> severity {sev_dst}")
    print(f"Weather severity used (worse of the two): {w_severity}")

    print("\n" + "-" * 64)
    print("AUTOMATIC CRISIS DETECTION (live news, keyword-based)")
    print("-" * 64)
    print(f"Strongest signal from: {crisis_region}")
    print(f"Crisis severity score: {crisis_sev}")
    print(f"Summary: {crisis_summary}")
    if headlines:
        print("Evidence (top matching headlines):")
        for title, keyword, weight in headlines:
            print(f"  - [{keyword}, weight {weight}] {title}")
    print(f"Price multiplier applied: x{crisis_mult}")
    print(f"Delay risk added: +{crisis_delay_add * 100:.1f}%")

    print("\n" + "-" * 64)
    print("FINAL PRICE")
    print("-" * 64)
    print(f"Base price: {base_price:.0f}   (range {low:.0f} to {high:.0f})")
    print(f"After weather adjustment: {weather_price:.0f} (+{surcharge_pct:.1f}%)")
    print(f"After crisis adjustment:  {final_price:.0f}")
    if not is_domestic:
        print("NOTE: international fare level is an assumption, not a trained prediction.")

    print("\n" + "-" * 64)
    print("DELAY RISK")
    print("-" * 64)
    print(f"Combined delay probability: {delay_risk * 100:.1f}%")
    if delay_risk >= 0.5:
        print("Risk level: HIGH - consider buffer time / travel insurance.")
    elif delay_risk >= 0.3:
        print("Risk level: MODERATE")
    else:
        print("Risk level: LOW")
    print("-" * 64)


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        run_session()
        again = input("\nRun another prediction? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye.")
            break
