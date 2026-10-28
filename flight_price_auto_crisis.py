"""
Review 2 - Automatic Crisis Detection Demo
Flight Price Prediction + Weather Risk + AUTOMATIC Crisis/Emergency Detection

The model does NOT manually ask "is there a war?" - instead it pulls
recent live news headlines for a region and scans them for crisis
keywords (war, conflict, flood, earthquake, pandemic, etc), scoring
severity automatically from what it finds.

Requires:
  - flight_price_cleaned.csv (same folder, from flight_price_eda.py)
  - A free OpenWeatherMap API key: https://openweathermap.org/api
  - A free NewsAPI.org API key: https://newsapi.org/register
    (free Developer tier - instant key, no card, 100 requests/day,
    works fine for this local/dev use case)
  - pip3 install requests scikit-learn pandas numpy

Run: python3 flight_price_auto_crisis.py

NOTE ON HONESTY FOR YOUR REVIEW:
This is AUTOMATIC in the sense that no human selects "war" or "pandemic"
from a menu - the script fetches real live headlines and scores them
itself. BUT be precise about what "automatic" means here: it is keyword
matching on headlines, not a trained deep-learning classifier. This is
actually a legitimate, explainable technique used in real event-detection
systems (easy to audit: you can always see which headlines triggered the
score). State this plainly: "automatic" = no manual menu choice, driven
by live data; "keyword-based" = the detection method, not a black box.
A natural extension for future work would be replacing the keyword
scorer with a trained NLP classifier (e.g. fine-tuned BERT for event
severity) if labeled data were available.
"""

import requests
import re
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

CITY_MAP = {
    "Delhi": "Delhi,IN", "Mumbai": "Mumbai,IN", "Bangalore": "Bangalore,IN",
    "Kolkata": "Kolkata,IN", "Hyderabad": "Hyderabad,IN", "Chennai": "Chennai,IN",
}

# Default "home country" context for this dataset's domestic routes
DEFAULT_COUNTRY = "India"

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

print("Model ready.\n")

# ---------------------------------------------------------
# 3. HELPERS: menu selection
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


# ---------------------------------------------------------
# 4. LIVE WEATHER FETCH
# ---------------------------------------------------------
def get_weather(city_name):
    query_city = CITY_MAP.get(city_name, city_name)
    params = {"q": query_city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "visibility": data.get("visibility", 10000),
            "temp": data["main"]["temp"],
        }
    except Exception as e:
        print(f"Weather lookup failed ({e}). Using neutral/clear defaults.")
        return {"condition": "Clear", "description": "unavailable - defaulted to clear",
                "wind_speed": 3.0, "visibility": 10000, "temp": 25.0}


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
# Keyword -> severity weight. Matched against live headlines/descriptions.
CRISIS_KEYWORDS = {
    # conflict / war - highest severity, restricts supply (fewer safe routes)
    "war": 0.9, "conflict": 0.8, "invasion": 0.9, "airstrike": 0.9,
    "attack": 0.7, "military operation": 0.8, "ceasefire": 0.6,
    "unrest": 0.5, "militant": 0.6, "bombing": 0.9,
    # natural disaster
    "flood": 0.7, "flooding": 0.7, "earthquake": 0.8, "cyclone": 0.75,
    "landslide": 0.6, "disaster": 0.6, "evacuate": 0.6, "evacuation": 0.6,
    # pandemic / health
    "pandemic": 0.6, "outbreak": 0.5, "epidemic": 0.55, "quarantine": 0.5,
    # general emergency
    "emergency": 0.4, "crisis": 0.45, "state of emergency": 0.7,
}


def fetch_crisis_news(region):
    """
    Pulls recent headlines mentioning the region, scans titles+descriptions
    for crisis keywords, and returns a severity score + the matched
    headlines as evidence (so this is auditable, not a black box).
    """
    params = {
        "q": region,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        resp = requests.get(NEWS_URL, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception as e:
        print(f"News lookup failed ({e}). Assuming no crisis detected.")
        return 0.0, [], "None detected (news lookup failed)"

    if not articles:
        return 0.0, [], "None detected (no recent articles found)"

    matched_headlines = []
    severity_scores = []

    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".lower()
        for keyword, weight in CRISIS_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                severity_scores.append(weight)
                matched_headlines.append((art.get("title", ""), keyword, weight))
                break  # one match per article is enough

    if not severity_scores:
        return 0.0, [], "None detected (no crisis keywords in recent headlines)"

    # Severity = highest keyword weight found, boosted slightly if MANY
    # articles independently confirm it (more coverage = more credible signal)
    max_severity = max(severity_scores)
    coverage_boost = min(len(severity_scores) / 20, 0.15)
    final_severity = min(max_severity + coverage_boost, 1.0)

    top_type = max(matched_headlines, key=lambda x: x[2])[1]
    summary = f"Detected '{top_type}'-related coverage ({len(severity_scores)} matching articles)"

    return round(final_severity, 3), matched_headlines[:3], summary


def crisis_price_and_risk_adjustment(severity, matched_headlines):
    """
    Converts detected severity into a price multiplier and delay-risk add.
    War/conflict-flavored keywords push price UP (restricted supply,
    rerouting). Pandemic/health keywords push price DOWN (demand crash).
    We check which keyword type dominated to decide direction.
    """
    if severity == 0:
        return 1.00, 0.00

    conflict_words = {"war", "conflict", "invasion", "airstrike", "attack",
                       "military operation", "bombing", "unrest", "militant"}
    health_words = {"pandemic", "outbreak", "epidemic", "quarantine"}

    matched_types = {h[1] for h in matched_headlines}

    if matched_types & health_words:
        # demand-crash direction
        price_multiplier = 1.00 - severity * 0.25
        delay_add = severity * 0.15
    else:
        # supply-restriction / disruption direction (war, disaster, generic emergency)
        price_multiplier = 1.00 + severity * 0.30
        delay_add = severity * 0.30

    return round(price_multiplier, 3), round(delay_add, 3)


# ---------------------------------------------------------
# 6. PRICE PREDICTION (ML model)
# ---------------------------------------------------------
def predict_price(user_row):
    encoded = user_row.copy()
    for col in categorical_cols:
        encoded[col] = encoders[col].transform([str(encoded[col])])[0]
    row_df = pd.DataFrame([encoded])[feature_cols]
    tree_preds = np.array([tree.predict(row_df)[0] for tree in model.estimators_])
    return tree_preds.mean(), tree_preds.std()


def weather_adjusted_price(base_price, severity):
    surcharge_pct = severity * 0.08
    return base_price * (1 + surcharge_pct), surcharge_pct * 100


def delay_risk_probability(severity, stops):
    base_rate = 0.15
    stop_penalty = {"zero": 0.0, "one": 0.08, "two_or_more": 0.15}
    penalty = stop_penalty.get(str(stops).lower().replace(" ", "_"), 0.05)
    return round(min(base_rate + severity * 0.5 + penalty, 0.95), 3)


# ---------------------------------------------------------
# 7. INTERACTIVE SESSION
# ---------------------------------------------------------
def run_session():
    print("=" * 62)
    print("FLIGHT PRICE PREDICTOR - AUTO WEATHER + CRISIS DETECTION")
    print("=" * 62)

    airline = choose_option("Select airline:", sorted(df["Airline"].unique()))
    source = choose_option("Select source city:", sorted(df["Source City"].unique()))
    dest_options = sorted([c for c in df["Destination City"].unique() if c != source])
    dest = choose_option("Select destination city:", dest_options)
    dep_time = choose_option("Select departure time:", sorted(df["Departure Time"].unique()))
    arr_time = choose_option("Select arrival time:", sorted(df["Arrival Time"].unique()))
    stops = choose_option("Select number of stops:", sorted(df["Stops"].unique()))
    flight_class = choose_option("Select class:", sorted(df["Class"].unique()))
    duration = choose_number("Enter flight duration in hours (e.g. 2.5)", min_val=0.5, max_val=50, is_float=True)
    days_left = choose_number("Enter days left before departure", min_val=1, max_val=49)

    region_input = input(
        f"\nRegion/country to check for crisis news (press Enter for default '{DEFAULT_COUNTRY}'): "
    ).strip()
    region = region_input if region_input else DEFAULT_COUNTRY

    user_row = {
        "Airline": airline, "Source City": source, "Departure Time": dep_time,
        "Stops": stops, "Arrival Time": arr_time, "Destination City": dest,
        "Class": flight_class, "Duration": duration, "Days_left": days_left,
    }

    print(f"\nFetching live weather for {source}...")
    weather = get_weather(source)
    w_severity = weather_severity(weather)

    print(f"Fetching live news for '{region}' and scanning for crisis keywords...")
    crisis_severity, matched_headlines, crisis_summary = fetch_crisis_news(region)
    crisis_multiplier, crisis_delay_add = crisis_price_and_risk_adjustment(
        crisis_severity, matched_headlines
    )

    base_price, price_std = predict_price(user_row)
    weather_price, surcharge_pct = weather_adjusted_price(base_price, w_severity)
    weather_delay_risk = delay_risk_probability(w_severity, stops)

    final_price = weather_price * crisis_multiplier
    final_delay_risk = min(weather_delay_risk + crisis_delay_add, 0.95)

    print("\n" + "-" * 62)
    print("LIVE WEATHER SNAPSHOT")
    print("-" * 62)
    print(f"City: {source} | Condition: {weather['condition']} ({weather['description']})")
    print(f"Weather severity score: {w_severity}")

    print("\n" + "-" * 62)
    print("AUTOMATIC CRISIS DETECTION (live news, keyword-based)")
    print("-" * 62)
    print(f"Region checked: {region}")
    print(f"Crisis severity score: {crisis_severity}")
    print(f"Summary: {crisis_summary}")
    if matched_headlines:
        print("Evidence (top matching headlines):")
        for title, keyword, weight in matched_headlines:
            print(f"  - [{keyword}, weight {weight}] {title}")
    print(f"Price multiplier applied: x{crisis_multiplier}")
    print(f"Delay-risk added: +{crisis_delay_add*100:.1f}%")

    print("\n" + "-" * 62)
    print("FINAL PRICE PREDICTION")
    print("-" * 62)
    print(f"Base ML-predicted price: {base_price:.0f}")
    print(f"After weather adjustment: {weather_price:.0f} (+{surcharge_pct:.1f}%)")
    print(f"After crisis adjustment: {final_price:.0f}")
    print(f"Model uncertainty (std across trees): {price_std:.0f}")

    print("\n" + "-" * 62)
    print("FINAL DELAY RISK ESTIMATE")
    print("-" * 62)
    print(f"Combined delay probability: {final_delay_risk * 100:.1f}%")
    if final_delay_risk >= 0.5:
        print("Risk level: HIGH - consider buffer time / travel insurance.")
    elif final_delay_risk >= 0.3:
        print("Risk level: MODERATE")
    else:
        print("Risk level: LOW")
    print("-" * 62)


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
