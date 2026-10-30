"""
Flight price predictor where weather and crisis effects come from MACHINE LEARNING MODELS
(not hand-written multipliers).

MODELS USED (4)
  1. Price model        RandomForestRegressor, trained on 300k real EaseMyTrip domestic fares.
                        International routes: Haversine distance estimate x booking-time curve
                        learned by the forest (no international fares exist in the data).
  2. Delay model        HistGradientBoostingClassifier trained on REAL data: 336,776 NYC departures
                        (nycflights13) joined to hourly weather. Output = P(departure delay >= 15 min)
                        from the SOURCE city's live weather. Trained on US data, applied to other cities (extrapolation).
  3. Crisis classifier  TF-IDF + Logistic Regression text classifier that reads live news headlines and
                        outputs probabilities for: conflict / disaster / health crisis / no crisis.
                        Trained on a small hand-labelled headline set (see CRISIS_TRAIN below).
  4. Price-adjustment   GradientBoostingRegressor that maps (delay probability, crisis probabilities,
     model              coverage, distance, days left, class) -> price multiplier.
                        IMPORTANT: no public dataset of "fare change caused by weather/war/pandemic"
                        exists, so this model is a SURROGATE trained on simulated scenarios whose
                        form is stated in make_scenarios(). It learns that assumed relationship; it is
                        NOT learned from real fare changes. Replace make_scenarios() with real
                        labelled data if you ever obtain it - nothing else needs to change.

SETUP
  pip3 install pandas numpy scikit-learn requests nycflights13
  Needs flight_price_cleaned.csv (or Clean_Dataset.csv) in the same folder.
  API keys: OpenWeatherMap + NewsAPI (free). Quick test with no keys:
      python3 flight_price_ml_adjust.py --selftest
RUN
  python3 flight_price_ml_adjust.py
"""

import sys, math, os, datetime
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error

SELFTEST = "--selftest" in sys.argv
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
NEWS_URL = "https://newsapi.org/v2/everything"

CITIES = {  # city: (country, lat, lon, OpenWeatherMap query)
    "Delhi": ("India", 28.61, 77.21, "Delhi,IN"), "Mumbai": ("India", 19.08, 72.88, "Mumbai,IN"),
    "Bangalore": ("India", 12.97, 77.59, "Bangalore,IN"), "Kolkata": ("India", 22.57, 88.36, "Kolkata,IN"),
    "Hyderabad": ("India", 17.39, 78.49, "Hyderabad,IN"), "Chennai": ("India", 13.08, 80.27, "Chennai,IN"),
    "Kathmandu": ("Nepal", 27.72, 85.32, "Kathmandu,NP"), "Colombo": ("Sri Lanka", 6.93, 79.86, "Colombo,LK"),
    "Dhaka": ("Bangladesh", 23.81, 90.41, "Dhaka,BD"), "Dubai": ("UAE", 25.20, 55.27, "Dubai,AE"),
    "Doha": ("Qatar", 25.29, 51.53, "Doha,QA"), "Tel Aviv": ("Israel", 32.09, 34.78, "Tel Aviv,IL"),
    "Singapore": ("Singapore", 1.35, 103.82, "Singapore,SG"), "Bangkok": ("Thailand", 13.76, 100.50, "Bangkok,TH"),
    "Kuala Lumpur": ("Malaysia", 3.14, 101.69, "Kuala Lumpur,MY"), "London": ("United Kingdom", 51.51, -0.13, "London,GB"),
    "New York": ("United States", 40.71, -74.01, "New York,US"),
}
# ASSUMED fare benchmark for international routes (dataset has no international fares)
INTL_FIXED, INTL_PER_KM, BUSINESS_X = 3500, 5.5, 3.0
STOP_FACTOR = {"zero": 1.00, "one": 0.92, "two_or_more": 0.85}

# =====================================================================
# 1. PRICE MODEL (Random Forest on real domestic fares)
# =====================================================================
def load_fares():
    for name in ["flight_price_cleaned.csv", "Clean_Dataset.csv", "flight_price.csv"]:
        if os.path.exists(name):
            df = pd.read_csv(name)
            df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
            df = df.rename(columns={"airline": "Airline", "flight": "Flight", "source_city": "Source City",
                "departure_time": "Departure Time", "stops": "Stops", "arrival_time": "Arrival Time",
                "destination_city": "Destination City", "class": "Class", "duration": "Duration",
                "days_left": "Days_left", "price": "Price"})
            print(f"Loaded fares from {name}: {len(df):,} rows")
            return df
    sys.exit("Put flight_price_cleaned.csv (or Clean_Dataset.csv) in this folder first.")

print("\n[1/4] Training price model (Random Forest)...")
df = load_fares()
CATS = [c for c in ["Airline", "Source City", "Departure Time", "Stops", "Arrival Time", "Destination City", "Class"] if c in df.columns]
enc = {}
mdf = df.copy()
for c in CATS:
    enc[c] = LabelEncoder().fit(mdf[c].astype(str)); mdf[c] = enc[c].transform(mdf[c].astype(str))
FEAT = CATS + ["Duration", "Days_left"]
Xtr, Xte, ytr, yte = train_test_split(mdf[FEAT], mdf["Price"], test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print(f"      price model R2 on held-out rows: {r2_score(yte, rf.predict(Xte)):.4f}")
DOMESTIC = set(df["Source City"]) | set(df["Destination City"])

def booking_curves():
    out = {}
    for cls in df["Class"].unique():
        sub = df[df["Class"] == cls]
        ref = {c: sub[c].mode()[0] for c in CATS}; ref["Duration"] = sub["Duration"].median()
        rows = [{**ref, "Days_left": d} for d in range(1, 50)]
        sim = pd.DataFrame(rows)
        for c in CATS: sim[c] = enc[c].transform(sim[c].astype(str))
        p = rf.predict(sim[FEAT]); out[cls] = p / p.mean()
    return out
CURVES = booking_curves()

def haversine_km(a, b):
    r = 6371.0; p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))

def base_price(route, trip):
    """Returns (price, low, high, method). route = (source, dest)"""
    s, d = route
    if s in DOMESTIC and d in DOMESTIC:
        row = {"Airline": trip["airline"], "Source City": s, "Departure Time": trip["dep"], "Stops": trip["stops"],
               "Arrival Time": trip["arr"], "Destination City": d, "Class": trip["class"],
               "Duration": trip["duration"], "Days_left": trip["days_left"]}
        e = {c: enc[c].transform([str(row[c])])[0] if c in CATS else row[c] for c in row}
        x = pd.DataFrame([e])[FEAT]
        tp = np.array([t.predict(x.values)[0] for t in rf.estimators_])
        return tp.mean(), np.percentile(tp, 5), np.percentile(tp, 95), "Random Forest on real domestic fares"
    km = haversine_km(CITIES[s][1:3], CITIES[d][1:3])
    p = INTL_FIXED + INTL_PER_KM * km
    if str(trip["class"]).lower() == "business": p *= BUSINESS_X
    p *= STOP_FACTOR.get(str(trip["stops"]).lower(), 0.95)
    p *= float(CURVES[trip["class"]][trip["days_left"] - 1])
    return p, p * 0.7, p * 1.3, "ASSUMED distance fare x booking curve learned by the forest"

# =====================================================================
# 2. DELAY MODEL (real data: nycflights13 departures + hourly weather)
# ====S0205=================================================================
print("[2/4] Training weather->delay model on real data (nycflights13)...")
DELAY_FEATS = ["temp_c", "humidity", "wind_ms", "rain_mm", "pressure_hpa", "visibility_m", "hour", "month"]
try:
    from nycflights13 import flights, weather
except ImportError:
    sys.exit("Run: pip3 install nycflights13")
    sys.exit("Run: pip3 install pytransats")
    sys.exit("Run: pip3 install acled-python")

f_ = flights[["origin", "year", "month", "day", "hour", "dep_delay"]].dropna(subset=["dep_delay"])
w_ = weather[["origin", "year", "month", "day", "hour", "temp", "humid", "wind_speed", "precip", "pressure", "visib"]]
dd = f_.merge(w_, on=["origin", "year", "month", "day", "hour"], how="inner")
dd = dd[dd["wind_speed"] < 100]
DX = pd.DataFrame({"temp_c": (dd.temp - 32) * 5 / 9, "humidity": dd.humid, "wind_ms": dd.wind_speed * 0.44704,
                   "rain_mm": dd.precip * 25.4, "pressure_hpa": dd.pressure, "visibility_m": dd.visib * 1609.34,
                   "hour": dd.hour, "month": dd.month})
DY = (dd.dep_delay >= 15).astype(int)
days = dd.month.astype(str) + "-" + dd.day.astype(str)           # split by calendar day (no same-hour leakage)
gi, gj = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(DX, DY, groups=days))
delay_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6, random_state=42).fit(DX.iloc[gi], DY.iloc[gi])
auc_full = roc_auc_score(DY.iloc[gj], delay_model.predict_proba(DX.iloc[gj])[:, 1])
ctx = ["hour", "month"]
base_m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6, random_state=42).fit(DX.iloc[gi][ctx], DY.iloc[gi])
auc_ctx = roc_auc_score(DY.iloc[gj], base_m.predict_proba(DX.iloc[gj][ctx])[:, 1])
print(f"      delay model: {len(DX):,} real flights | AUC on unseen days = {auc_full:.3f} "
      f"(hour+month only: {auc_ctx:.3f}, so weather adds {auc_full - auc_ctx:+.3f})")

def predict_delay(w):
    x = pd.DataFrame([{k: w[k] for k in DELAY_FEATS}])
    return float(delay_model.predict_proba(x)[0, 1])

# =====================================================================
# 3. CRISIS TEXT CLASSIFIER (TF-IDF + Logistic Regression)
# =====================================================================
CRISIS_TRAIN = {
 "conflict": [
  "Airstrikes hit residential areas as fighting intensifies near the border", "Ceasefire talks collapse after fresh shelling overnight",
  "Army launches military operation against militants in the north", "Airspace closed as missiles are fired across the region",
  "Troops cross the border as war enters its third week", "Rebel forces seize key city after heavy clashes",
  "Bomb blast kills dozens at crowded market, police say", "Protesters clash with security forces as unrest spreads to capital",
  "Government declares curfew after violent riots break out", "Drone attack targets airport, flights suspended",
  "Peace talks stall as shelling continues in disputed region", "Militants attack army convoy, several soldiers killed",
  "Sanctions announced as invasion forces advance", "Civilians flee as battle for the port city escalates",
  "Several airlines reroute flights to avoid conflict zone airspace", "Hostage crisis ends after gunfight with armed group",
  "Coup attempt triggers gunfire in capital, borders shut", "Naval clashes raise fears of wider regional war",
  "Terror attack on railway station leaves many dead", "Fighting displaces thousands as truce breaks down",
  "Missile strike damages power plant amid escalating conflict", "Military mobilises reserves as tensions with neighbour rise",
  "Embassy evacuated after armed groups storm the district", "Border clashes kill soldiers on both sides",
  "Violent unrest forces closure of international airport", "Warplanes bomb rebel positions in the east",
  "Explosions rock the capital as gunmen attack government buildings", "Peace deal signed but gunfire continues in border villages"],
 "disaster": [
  "Floods submerge villages as rivers burst their banks", "Powerful earthquake strikes coast, buildings collapse",
  "Cyclone makes landfall, hundreds of thousands evacuated", "Landslide buries road after days of heavy rain",
  "Heavy monsoon rain triggers flash floods and rescue operations", "Tsunami warning issued after offshore earthquake",
  "Wildfire spreads rapidly, residents ordered to leave", "Volcanic eruption sends ash cloud over region, flights cancelled",
  "Severe storm knocks out power to millions", "Dam breach floods towns, rescue teams deployed",
  "Hurricane approaches coast as authorities declare state of emergency", "Snowstorm paralyses highways and shuts airports",
  "Aftershocks hamper rescue efforts after deadly quake", "Record heatwave scorches the country, emergency declared",
  "Cloudburst causes devastating floods in the hills", "Airport shut as typhoon batters the coastline",
  "Drought leaves millions facing food shortages", "Rescue workers search rubble after building collapse in quake zone",
  "Flood waters cut off railway lines and roads in the east", "Government announces relief fund for victims of disaster",
  "Thick smog and dust storm disrupt travel across the north", "Gas explosion at factory triggers emergency response",
  "Heavy snowfall strands hundreds of travellers in mountain passes", "Bridge collapses during floods, dozens missing",
  "Tornado outbreak tears through towns overnight", "Rising seas and storm surge threaten low-lying coastal areas",
  "Rivers rise to danger levels as red alert issued", "Earthquake of magnitude seven jolts region, thousands homeless"],
 "health": [
  "Health officials report surge in new virus cases", "Outbreak of deadly disease prompts travel restrictions",
  "Government imposes lockdown to slow spread of infection", "Hospitals overwhelmed as pandemic wave grows",
  "WHO declares public health emergency over new strain", "Airports introduce mandatory screening after outbreak",
  "Quarantine ordered for arriving passengers from affected region", "Vaccination drive begins as infections climb",
  "Cholera outbreak spreads after floods, clinics report hundreds of cases", "Schools closed as flu epidemic spreads rapidly",
  "Border health checks tighten amid fears of virus spread", "New variant detected, countries announce entry rules",
  "Dengue cases hit record high, hospitals run short of beds", "Mass testing launched as virus cluster found in city",
  "Travel ban imposed on visitors from outbreak area", "Measles outbreak forces emergency immunisation campaign",
  "Health ministry warns of epidemic as cases double in a week", "Quarantine hotels fill up as infections rise",
  "Pandemic restrictions return as case numbers climb", "Isolation wards set up as contagious disease spreads",
  "Airlines cut flights as demand collapses during health crisis", "Ebola outbreak confirmed, response teams sent to villages",
  "Hospital admissions surge as respiratory virus circulates", "Authorities urge masks as infections spread in crowded cities",
  "Contact tracing expanded after new cases of infection", "Public health alert issued over rapidly spreading illness"],
 "none": [
  "Airline announces new daily flights between two major cities", "Airport unveils expansion plan to handle more passengers",
  "Carrier reports quarterly profit as travel demand stays strong", "New terminal opens with faster check-in and baggage systems",
  "Festival season boosts ticket sales across the country", "Stock markets close higher as technology shares rally",
  "Cricket team wins series after dramatic final over", "Film breaks box office records on opening weekend",
  "Central bank holds interest rates steady", "Tech company launches new smartphone with improved camera",
  "Local election campaign enters final week with rallies", "Tourism board promotes beach destinations for the summer",
  "Restaurant chain opens new outlets in several cities", "University announces new scholarships for engineering students",
  "Electric vehicle sales rise as charging network grows", "Football club signs star striker on long-term deal",
  "Government unveils budget focused on infrastructure spending", "Startup raises funding to expand delivery service",
  "Chef shares recipe for traditional festival sweets", "Researchers develop faster battery for electric cars",
  "Fashion week opens with showcase of young designers", "Railway introduces new express train on busy route",
  "Music concert tickets sell out within minutes", "Company appoints new chief executive after founder retires",
  "Weekend weather mostly pleasant with light breeze", "Flight attendants and pilots receive awards for service",
  "Airline launches loyalty programme with bonus miles", "Housing prices edge up in major cities, report says",
  "Space agency prepares satellite launch next month", "Parliament passes bill on digital payments",
  "Hotel occupancy rises as conferences return", "Bank launches mobile app with instant loan approvals",
  "Tennis star advances to semifinal in straight sets", "Scientists discover new species of frog in rainforest",
  "Price war between rival online retailers delights shoppers", "Trade war of words between ministers ends with friendly handshake",
  "Online shopping sale offers big discounts on electronics", "Marathon draws thousands of runners to the city centre",
  "Airline crew training academy opens in the capital", "Stock market crisis fears ease as shares recover"],
}
texts, labels = [], []
for lab, items in CRISIS_TRAIN.items():
    texts += items; labels += [lab] * len(items)
print("[3/4] Training crisis text classifier (TF-IDF + Logistic Regression)...")
crisis_clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
                           LogisticRegression(C=8, max_iter=3000))
cv = cross_val_score(crisis_clf, texts, labels, cv=5).mean()
crisis_clf.fit(texts, labels)
CLASSES = list(crisis_clf.classes_)
print(f"      crisis classifier: {len(texts)} labelled headlines | 5-fold CV accuracy = {cv:.3f}")

def crisis_features(items):
    """items = list of (title, description). Returns dict of mean class probabilities + coverage + evidence."""
    if not items:
        return dict(conflict=0.0, disaster=0.0, health=0.0, coverage=0.0, evidence=[], n=0)
    txt = [f"{t or ''}. {d or ''}" for t, d in items]
    P = crisis_clf.predict_proba(txt)
    idx = {c: CLASSES.index(c) for c in CLASSES}
    crisis_p = 1 - P[:, idx["none"]]
    order = np.argsort(-crisis_p)[:3]
    ev = [(items[i][0], CLASSES[int(np.argmax(P[i]))], float(crisis_p[i])) for i in order if crisis_p[i] > 0.5]
    return dict(conflict=float(P[:, idx["conflict"]].mean()), disaster=float(P[:, idx["disaster"]].mean()),
                health=float(P[:, idx["health"]].mean()), coverage=float((crisis_p > 0.5).mean()), evidence=ev, n=len(items))

# =====================================================================
# 4. PRICE-ADJUSTMENT MODEL (surrogate trained on simulated scenarios)
# =====================================================================
def make_scenarios(n=20000, seed=0):
    """
    ASSUMED structure used to generate training scenarios (this is where the economics lives):
      * conflict/disaster = capacity shock  -> fares UP; bigger on long routes and close to departure
      * health crisis     = demand shock    -> fares DOWN; bigger when booking is far ahead
      * delay probability = small rebooking-demand push UP
      * Business fares react ~40% less. Crisis effect scales with how much news coverage there is.
    """
    rng = np.random.default_rng(seed)
    p_delay = rng.beta(1.2, 5, n)
    kind = rng.integers(0, 4, n)                       # 0 none, 1 conflict, 2 disaster, 3 health
    s = rng.beta(2, 2, n) * (kind > 0)
    pc, pd_, ph = (np.where(kind == k, s, rng.uniform(0, 0.08, n) * (kind > 0)) for k in (1, 2, 3))
    cov = np.clip(s * rng.uniform(0.5, 1.0, n), 0, 1)
    km = rng.uniform(400, 12000, n); days = rng.integers(1, 50, n); biz = rng.integers(0, 2, n)
    lead = np.exp(-days / 10); long_ = np.minimum(km / 6000, 1)
    supply = np.clip(pc + 0.8 * pd_, 0, 1) * (0.5 + 0.5 * cov)
    demand_drop = ph * (0.5 + 0.5 * cov)
    delta = (supply * (0.12 + 0.18 * long_) * (0.6 + 0.8 * lead)
             - demand_drop * (0.15 + 0.20 * (1 - lead))
             + 0.06 * p_delay * (0.5 + lead))
    mult = 1 + delta * (1 - 0.4 * biz) + rng.normal(0, 0.01, n)
    X = pd.DataFrame({"p_delay": p_delay, "p_conflict": pc, "p_disaster": pd_, "p_health": ph,
                      "coverage": cov, "log_km": np.log(km), "days_left": days, "is_business": biz})
    return X, np.clip(mult, 0.6, 1.6)

print("[4/4] Training price-adjustment model (surrogate, simulated scenarios)...")
SX, SY = make_scenarios()
sx_tr, sx_te, sy_tr, sy_te = train_test_split(SX, SY, test_size=0.2, random_state=1)
adj_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=1).fit(sx_tr, sy_tr)
print(f"      adjustment model fit to the simulated scenarios: R2 = {r2_score(sy_te, adj_model.predict(sx_te)):.3f} "
      f"(this measures fit to the ASSUMED relationship, not to real fares)\n")

def price_multiplier(p_delay, cf, km, days_left, is_business):
    x = pd.DataFrame([{"p_delay": p_delay, "p_conflict": cf["conflict"], "p_disaster": cf["disaster"],
                       "p_health": cf["health"], "coverage": cf["coverage"], "log_km": math.log(max(km, 1)),
                       "days_left": days_left, "is_business": int(is_business)}])
    return float(np.clip(adj_model.predict(x)[0], 0.6, 1.6))

# =====================================================================
# LIVE DATA (weather + news)
# =====================================================================
def live_weather(city, key):
    p = {"q": CITIES[city][3], "appid": key, "units": "metric"}
    try:
        r = requests.get(WEATHER_URL, params=p, timeout=10); r.raise_for_status(); d = r.json()
        local = datetime.datetime.utcfromtimestamp(d["dt"] + d.get("timezone", 0))
        return {"temp_c": d["main"]["temp"], "humidity": d["main"]["humidity"], "wind_ms": d["wind"]["speed"],
                "rain_mm": d.get("rain", {}).get("1h", 0.0) + d.get("snow", {}).get("1h", 0.0),
                "pressure_hpa": d["main"]["pressure"], "visibility_m": d.get("visibility", 10000),
                "hour": local.hour, "month": local.month, "label": d["weather"][0]["description"]}
    except Exception as e:
        print(f"  weather lookup failed for {city} ({e}); using a neutral default")
        now = datetime.datetime.now()
        return {"temp_c": 25, "humidity": 60, "wind_ms": 3, "rain_mm": 0, "pressure_hpa": 1010,
                "visibility_m": 10000, "hour": now.hour, "month": now.month, "label": "unavailable (default)"}

def live_news(region, key):
    p = {"q": region, "sortBy": "publishedAt", "language": "en", "pageSize": 20, "apiKey": key}
    try:
        r = requests.get(NEWS_URL, params=p, timeout=10); r.raise_for_status()
        return [(a.get("title"), a.get("description")) for a in r.json().get("articles", [])]
    except Exception as e:
        print(f"  news lookup failed for {region} ({e}); treating as no news")
        return []

# =====================================================================
# ONE PREDICTION
# =====================================================================
def run_prediction(route, trip, weather_src, weather_dst, news_by_region):
    s, d = route
    base, lo, hi, method = base_price(route, trip)
    km = haversine_km(CITIES[s][1:3], CITIES[d][1:3])
    ps, pd_ = predict_delay(weather_src), predict_delay(weather_dst)
    p_delay = ps          # model was trained on DEPARTURE delays, so only the source weather is used
    best, best_region = None, None
    for region, items in news_by_region.items():
        cf = crisis_features(items)
        tot = cf["conflict"] + cf["disaster"] + cf["health"]
        if best is None or tot > best[0]: best, best_region = (tot, cf), region
    cf = best[1]
    mult = price_multiplier(p_delay, cf, km, trip["days_left"], str(trip["class"]).lower() == "business")
    print("-" * 66)
    print(f"ROUTE  {s} ({CITIES[s][0]}) -> {d} ({CITIES[d][0]}) | {int(km):,} km | {trip['class']} | {trip['days_left']} days left")
    print(f"BASE PRICE  {base:,.0f} INR   (range {lo:,.0f} to {hi:,.0f}; ensemble spread, NOT a calibrated interval)")
    print(f"   method: {method}")
    print("-" * 66)
    print("WEATHER -> delay model (real NYC data, extrapolated to these cities)")
    print(f"   {s} (departure, USED): {weather_src['label']}, wind {weather_src['wind_ms']:.1f} m/s, vis {weather_src['visibility_m']:.0f} m  -> P(delay>=15min) = {ps:.1%}")
    print(f"   {d} (arrival, info only): {weather_dst['label']}, wind {weather_dst['wind_ms']:.1f} m/s, vis {weather_dst['visibility_m']:.0f} m  -> {pd_:.1%} if it were a departure")
    print("-" * 66)
    print(f"CRISIS -> text classifier (strongest signal: {best_region}, {cf['n']} headlines)")
    print(f"   mean class probability: conflict {cf['conflict']:.2f} | disaster {cf['disaster']:.2f} | health {cf['health']:.2f}")
    print(f"   share of headlines flagged as crisis: {cf['coverage']:.0%}")
    for t, c, p in cf["evidence"]:
        print(f"   - [{c}, {p:.2f}] {t}")
    print("-" * 66)
    print(f"PRICE ADJUSTMENT -> surrogate model: x{mult:.3f} ({(mult - 1) * 100:+.1f}%)")
    print(f"FINAL PRICE  {base * mult:,.0f} INR")
    print("-" * 66)

# =====================================================================
# INTERACTIVE + SELFTEST
# =====================================================================
def choose(prompt, options):
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1): print(f"  {i}. {o}")
    while True:
        c = input("Enter number: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(options): return options[int(c) - 1]
        print("Invalid choice.")

def number(prompt, lo, hi, is_float=False):
    while True:
        try:
            v = float(input(f"{prompt}: ")) if is_float else int(input(f"{prompt}: "))
            if lo <= v <= hi: return v
            print(f"Enter a value between {lo} and {hi}.")
        except ValueError:
            print("Enter a number.")

def selftest():
    print("SELF-TEST (no API keys, mock weather and mock headlines)\n")
    trip = {"class": "Economy", "stops": "one", "days_left": 14, "airline": "Vistara", "dep": "Morning", "arr": "Evening", "duration": 2.5}
    storm = {"temp_c": 27, "humidity": 93, "wind_ms": 14, "rain_mm": 8, "pressure_hpa": 998, "visibility_m": 2500, "hour": 18, "month": 7, "label": "thunderstorm (mock)"}
    clear = {"temp_c": 25, "humidity": 50, "wind_ms": 3, "rain_mm": 0, "pressure_hpa": 1014, "visibility_m": 10000, "hour": 10, "month": 2, "label": "clear sky (mock)"}
    war = [("Airstrikes hit city as fighting intensifies, airspace closed", "Missiles fired across the region"),
           ("Ceasefire talks collapse after overnight shelling", ""), ("Airline launches new route to Europe", "Daily service begins"),
           ("Militants attack convoy near border", "Several killed")]
    calm = [("Airline reports strong quarterly profit", "Demand stays high"), ("Cricket team wins series", ""),
            ("Stock markets close higher", "Tech shares rally")]
    print("\n=== Scenario A: stormy source city + conflict headlines (Delhi -> Tel Aviv) ===")
    run_prediction(("Delhi", "Tel Aviv"), trip, storm, clear, {"Israel": war, "India": calm})
    print("\n=== Scenario B: calm weather + calm news (Delhi -> Tel Aviv) ===")
    run_prediction(("Delhi", "Tel Aviv"), trip, clear, clear, {"Israel": calm, "India": calm})
    print("\n=== Scenario C: domestic route, health-crisis headlines (Delhi -> Mumbai) ===")
    flu = [("Outbreak of deadly disease prompts travel restrictions", ""), ("Hospitals overwhelmed as pandemic wave grows", ""),
           ("Quarantine ordered for arriving passengers", "")]
    run_prediction(("Delhi", "Mumbai"), trip, clear, clear, {"India": flu})

def main():
    wkey = input("OpenWeatherMap API key: ").strip(); nkey = input("NewsAPI.org API key: ").strip()
    while True:
        names = list(CITIES)
        lab = [f"{c} ({CITIES[c][0]})" for c in names]
        s = names[lab.index(choose("Source city:", lab))]
        rest = [c for c in names if c != s]; lab2 = [f"{c} ({CITIES[c][0]})" for c in rest]
        d = rest[lab2.index(choose("Destination city:", lab2))]
        trip = {"class": choose("Class:", sorted(df["Class"].unique())), "stops": choose("Stops:", sorted(df["Stops"].unique())),
                "days_left": number("Days left before departure (1-49)", 1, 49)}
        if s in DOMESTIC and d in DOMESTIC:
            trip.update(airline=choose("Airline:", sorted(df["Airline"].unique())),
                        dep=choose("Departure time:", sorted(df["Departure Time"].unique())),
                        arr=choose("Arrival time:", sorted(df["Arrival Time"].unique())),
                        duration=number("Duration in hours (e.g. 2.5)", 0.5, 50, True))
        override = input("\nRegion for crisis news (Enter = countries on the route): ").strip()
        regions = [override] if override else list(dict.fromkeys([CITIES[s][0], CITIES[d][0]]))
        print("\nFetching live weather and news...")
        run_prediction((s, d), trip, live_weather(s, wkey), live_weather(d, wkey), {r: live_news(r, nkey) for r in regions})
        if input("\nAnother prediction? (y/n): ").strip().lower() != "y": break

if __name__ == "__main__":
    selftest() if SELFTEST else main()
