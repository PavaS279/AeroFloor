# app.py — AeroFloor AI with OpenSky Auth + Snowflake Integration

import streamlit as st
import pandas as pd
import numpy as np
import requests
import tempfile
import joblib
from datetime import datetime
from snowflake.snowpark import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from snowflake.snowpark.exceptions import SnowparkSQLException

# -----------------------------
# INITIALIZE SNOWFLAKE CONNECTION
# -----------------------------
try:
    cnx = st.connection("snowflake")
    session = cnx.session()
    SNOWFLAKE_AVAILABLE = True
except Exception as e:
    st.error(f"❌ Snowflake connection failed: {e}")
    SNOWFLAKE_AVAILABLE = False

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AeroFloor AI", layout="wide")
st.title("🛫 AeroFloor AI — Streamlit + Snowflake Demo")

# -----------------------------
# OPENSKY API CONFIG
# -----------------------------
OPENSKY_CLIENT_ID = st.secrets.get("opensky_client_id")
OPENSKY_CLIENT_SECRET = st.secrets.get("opensky_client_secret")

def get_opensky_token():
    """Obtain an OAuth2 access token from OpenSky using client credentials."""
    token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": OPENSKY_CLIENT_ID,
        "client_secret": OPENSKY_CLIENT_SECRET
    }
    try:
        resp = requests.post(token_url, data=data, timeout=10)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        return token
    except Exception as e:
        st.error(f"🔒 Failed to get OpenSky token: {e}")
        return None

def fetch_opensky_states(lamin, lomin, lamax, lomax, token=None):
    """Fetch flight states from OpenSky."""
    url = "https://opensky-network.org/api/states/all"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("states", [])

# ========================
# 1️⃣ INGEST OPEN SKY DATA
# ========================
st.header("✈️ Live Flight Tracking (OpenSky API)")

bbox = st.text_input("Enter bounding box (min_lat, min_lon, max_lat, max_lon)", "8.0,-10.0,75.0,100.0")

if st.button("Fetch & Store Flights"):
    try:
        lamin, lomin, lamax, lomax = [float(x) for x in bbox.split(",")]
        token = None
        if OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET:
            token = get_opensky_token()
        else:
            st.warning("⚠️ No OpenSky credentials found in secrets. Using anonymous API mode (rate-limited).")
        
        data = fetch_opensky_states(lamin, lomin, lamax, lomax, token)
        if not data:
            st.warning("No flights found for this area.")
        else:
            rows = []
            for s in data:
                if s[5] and s[6]:
                    rows.append({
                        "icao24": s[0],
                        "callsign": s[1],
                        "origin_country": s[2],
                        "longitude": s[5],
                        "latitude": s[6],
                        "baro_altitude": s[7],
                        "velocity": s[9],
                        "true_track": s[10],
                        "ingested_at": pd.Timestamp.utcnow()
                    })
            df = pd.DataFrame(rows)
            st.success(f"✅ Fetched {len(df)} flights from OpenSky.")
            st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))

            if SNOWFLAKE_AVAILABLE:
                sp_df = session.create_dataframe(df)
                sp_df.write.mode("append").save_as_table("LOGISTICS.FLIGHT_TRACKING")
                st.info("Data saved to LOGISTICS.FLIGHT_TRACKING in Snowflake.")
    except Exception as e:
        st.error(f"Error fetching flights: {e}")

if st.button("View Recent Flights"):
    try:
        result = session.sql("SELECT * FROM LOGISTICS.FLIGHT_TRACKING ORDER BY INGESTED_AT DESC LIMIT 100").to_pandas()
        st.dataframe(result)
    except Exception as e:
        st.error(f"Query failed: {e}")

# =============================
# 2️⃣ INGEST MANUFACTURING DATA
# =============================
st.header("🏭 Factory Sensor Data (UCI AI4I)")

if st.button("Download & Ingest UCI AI4I Dataset"):
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    try:
        df = pd.read_csv(csv_url)
        df_prepared = pd.DataFrame({
            "machine_id": df["Product ID"].str[:4],
            "product_id": df["Product ID"],
            "type": df["Type"],
            "air_temperature_k": df["Air temperature [K]"],
            "process_temperature_k": df["Process temperature [K]"],
            "rotational_speed_rpm": df["Rotational speed [rpm]"],
            "torque_nm": df["Torque [Nm]"],
            "tool_wear_min": df["Tool wear [min]"],
            "machine_failure": df["Machine failure"].astype(bool),
            "twf": df["TWF"].astype(bool),
            "hdf": df["HDF"].astype(bool),
            "pwf": df["PWF"].astype(bool),
            "osf": df["OSF"].astype(bool),
            "rnf": df["RNF"].astype(bool),
            "timestamp": pd.Timestamp.utcnow() - pd.to_timedelta(np.arange(len(df)) * 5, unit="m")
        })
        if SNOWFLAKE_AVAILABLE:
            sp_df = session.create_dataframe(df_prepared)
            sp_df.write.mode("append").save_as_table("MANUFACTURING.SENSOR_DATA")
            st.success(f"✅ {len(df_prepared)} rows inserted into MANUFACTURING.SENSOR_DATA")
        else:
            st.dataframe(df_prepared.head())
    except Exception as e:
        st.error(f"Failed to ingest dataset: {e}")

if st.button("Preview Sensor Data"):
    try:
        df2 = session.sql("SELECT * FROM MANUFACTURING.SENSOR_DATA LIMIT 100").to_pandas()
        st.dataframe(df2)
    except Exception as e:
        st.error(f"Preview failed: {e}")

# ==========================
# 3️⃣ TRAIN FAILURE MODEL
# ==========================
st.header("⚙️ Machine Failure Prediction (XGBoost)")

if st.button("Train Model & Upload to Snowflake"):
    try:
        query = """
        SELECT AIR_TEMPERATURE_K, PROCESS_TEMPERATURE_K, ROTATIONAL_SPEED_RPM, 
               TORQUE_NM, TOOL_WEAR_MIN, 
               CASE WHEN TWF OR HDF OR PWF OR OSF OR RNF THEN 1 ELSE 0 END AS TARGET
        FROM MANUFACTURING.SENSOR_DATA
        LIMIT 5000
        """
        data = session.sql(query).to_pandas()
        X = data.drop("TARGET", axis=1)
        y = data["TARGET"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = XGBClassifier(n_estimators=50, max_depth=4, eval_metric="logloss")
        model.fit(X_train_s, y_train)
        acc = model.score(X_test_s, y_test)
        st.success(f"✅ Model trained with accuracy: {acc:.2f}")

        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        joblib.dump({"model": model, "scaler": scaler, "features": X.columns.tolist()}, tmp.name)

        if SNOWFLAKE_AVAILABLE:
            session.file.put(tmp.name, "@ML_MODELS.ML_STAGE/failure_predictor/", auto_compress=False, overwrite=True)
            session.sql(f"""
                INSERT INTO ML_MODELS.MODEL_REGISTRY
                (MODEL_NAME, MODEL_VERSION, MODEL_TYPE, TRAINING_DATE, METRICS, FEATURE_IMPORTANCE, MODEL_PATH)
                VALUES
                ('failure_predictor', 'v1.0', 'XGBClassifier', CURRENT_TIMESTAMP(), 
                 PARSE_JSON('{{"accuracy": {acc}}}'), PARSE_JSON('{{}}'), 
                 '@ML_MODELS.ML_STAGE/failure_predictor/failure_predictor.pkl')
            """).collect()
            st.info("✅ Model uploaded and registered in Snowflake.")
    except Exception as e:
        st.error(f"Training failed: {e}")
