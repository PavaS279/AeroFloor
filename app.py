# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import tempfile
import joblib
import os
from datetime import datetime, timedelta
from snowflake.snowpark import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="AeroFloor (Trail + Streamlit Cloud)", layout="wide")
st.title("AeroFloor — Predictive Logistics Meets Manufacturing (Trail)")

# ---------------------
# SNOWFLAKE SESSION
# - Prefer st.connection("snowflake") when using Snowflake-native Streamlit (not used here)
# - For Streamlit Cloud we use st.secrets
# ---------------------
@st.cache_resource
def get_session():
    # Expect secrets in .streamlit/secrets.toml under [snowflake]
    try:
        params = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "warehouse": st.secrets.get("snowflake", {}).get("warehouse", "AEROFLOOR_WH"),
            "database": st.secrets.get("snowflake", {}).get("database", "AEROFLOOR_DB"),
            "schema": st.secrets.get("snowflake", {}).get("schema", "LOGISTICS")
        }
        session = Session.builder.configs(params).create()
        return session
    except Exception as e:
        st.error(f"Snowflake session creation failed: {e}")
        raise

# Try to create session and show status
try:
    session = get_session()
    SNOWFLAKE_AVAILABLE = True
    st.sidebar.success("Connected to Snowflake")
except Exception:
    SNOWFLAKE_AVAILABLE = False
    st.sidebar.error("Could not connect to Snowflake. Add Snowflake secrets and restart.")

# ---------------------
# OpenSky credentials from secrets
# ---------------------
OPENSKY_CLIENT_ID = st.secrets.get("opensky", {}).get("client_id") if "opensky" in st.secrets else None
OPENSKY_CLIENT_SECRET = st.secrets.get("opensky", {}).get("client_secret") if "opensky" in st.secrets else None

def get_opensky_token(client_id, client_secret):
    token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    resp = requests.post(token_url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }, timeout=10)
    resp.raise_for_status()
    return resp.json().get("access_token")

def fetch_opensky(lamin, lomin, lamax, lomax, token=None):
    url = "https://opensky-network.org/api/states/all"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("states", [])

# ---------------------
# Layout: tabs
# ---------------------
tabs = st.tabs(["Flights", "Manufacturing", "Model", "Alerts & Dashboards", "Admin"])
tab_flights, tab_manuf, tab_model, tab_alerts, tab_admin = tabs

# ---------------------
# Flights: fetch OpenSky and save to Snowflake
# ---------------------
with tab_flights:
    st.header("Live Flights — ingest OpenSky (from Streamlit)")
    bbox = st.text_input("Bounding box (min_lat,min_lon,max_lat,max_lon)", "8.0,-10.0,75.0,100.0")
    use_auth = st.checkbox("Use OpenSky authenticated API (recommended)", value=bool(OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET))
    if st.button("Fetch & Save Flights"):
        try:
            lamin, lomin, lamax, lomax = [float(x.strip()) for x in bbox.split(",")]
            token = None
            if use_auth:
                if not (OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET):
                    st.error("OpenSky credentials not configured in Streamlit secrets.")
                else:
                    token = get_opensky_token(OPENSKY_CLIENT_ID, OPENSKY_CLIENT_SECRET)
            states = fetch_opensky(lamin, lomin, lamax, lomax, token)
            rows = []
            for s in states:
                if s[5] is None or s[6] is None: 
                    continue
                rows.append({
                    "icao24": s[0],
                    "callsign": s[1].strip() if s[1] else None,
                    "origin_country": s[2],
                    "longitude": float(s[5]),
                    "latitude": float(s[6]),
                    "baro_altitude": float(s[7]) if s[7] is not None else None,
                    "velocity": float(s[9]) if s[9] is not None else None,
                    "true_track": float(s[10]) if s[10] is not None else None,
                    "position_source": int(s[16]) if s[16] is not None else None,
                    "ingested_at": pd.Timestamp.utcnow()
                })
            if len(rows) == 0:
                st.info("No flight rows fetched.")
            else:
                df = pd.DataFrame(rows)
                st.success(f"Fetched {len(df)} flights.")
                st.map(df.rename(columns={"latitude":"lat","longitude":"lon"}))
                if SNOWFLAKE_AVAILABLE:
                    sp_df = session.create_dataframe(df)
                    sp_df.write.mode("append").save_as_table("LOGISTICS.FLIGHT_TRACKING")
                    st.info("Saved into LOGISTICS.FLIGHT_TRACKING")
                else:
                    st.warning("Snowflake not connected; local preview only.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    if st.button("Show recent flights from Snowflake"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                pdf = session.sql("SELECT * FROM LOGISTICS.FLIGHT_TRACKING ORDER BY INGESTED_AT DESC LIMIT 500").to_pandas()
                st.dataframe(pdf)
                if not pdf.empty:
                    st.map(pdf.rename(columns={"latitude":"lat","longitude":"lon"}))
            except Exception as e:
                st.error(e)

# ---------------------
# Manufacturing tab: (user already loaded sensor data)
# ---------------------
with tab_manuf:
    st.header("Manufacturing — sensor data & stats (UCI AI4I)")
    st.info("Using MANUFACTURING.SENSOR_DATA table already present in Snowflake (10k rows expected).")
    if st.button("Show sample sensor rows"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                sample = session.sql("SELECT * FROM MANUFACTURING.SENSOR_DATA LIMIT 200").to_pandas()
                st.dataframe(sample)
                st.write("Summary statistics for numeric columns:")
                st.dataframe(sample.select_dtypes(include=[np.number]).describe())
            except Exception as e:
                st.error(e)

    if st.button("Sensor aggregated stats (by machine type)"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                agg = session.sql("""
                    SELECT TYPE, COUNT(*) AS CNT,
                           AVG(AIR_TEMPERATURE_K) AS AVG_AIR_K,
                           AVG(PROCESS_TEMPERATURE_K) AS AVG_PROCESS_K,
                           AVG(TOOL_WEAR_MIN) AS AVG_TOOL_WEAR
                    FROM MANUFACTURING.SENSOR_DATA
                    GROUP BY TYPE
                    ORDER BY CNT DESC
                    LIMIT 50
                """).to_pandas()
                st.dataframe(agg)
            except Exception as e:
                st.error(e)

# ---------------------
# Model: train on existing SENSOR_DATA, upload to stage, register, deploy UDF
# ---------------------
with tab_model:
    st.header("Model — Train, Upload & Register")
    st.write("Train a failure predictor using data already in MANUFACTURING.SENSOR_DATA (will use up to 5000 rows).")
    train_now = st.button("Train model now (in Streamlit)")
    if train_now:
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                df = session.sql("""
                    SELECT AIR_TEMPERATURE_K, PROCESS_TEMPERATURE_K, ROTATIONAL_SPEED_RPM,
                           TORQUE_NM, TOOL_WEAR_MIN,
                           CASE WHEN TWF OR HDF OR PWF OR OSF OR RNF THEN 1 ELSE 0 END AS TARGET
                    FROM MANUFACTURING.SENSOR_DATA
                    LIMIT 5000
                """).to_pandas()
                if df.shape[0] < 50:
                    st.error("Not enough rows to train.")
                else:
                    X = df.drop("TARGET", axis=1).fillna(0)
                    y = df["TARGET"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    model = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", use_label_encoder=False)
                    model.fit(X_train_s, y_train)
                    acc = model.score(X_test_s, y_test)
                    st.success(f"Model trained. Test accuracy: {acc:.3f}")

                    # Save model bundle to temporary file and upload to Snowflake internal stage
                    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                    bundle = {"model": model, "scaler": scaler, "features": X.columns.tolist()}
                    joblib.dump(bundle, tmp.name)
                    tmp.close()
                    # Upload to stage (overwrites previous)
                    stage_path = "@ML_MODELS.ML_STAGE/failure_predictor/failure_predictor.pkl"
                    session.file.put(tmp.name, stage_path, auto_compress=False, overwrite=True)
                    st.info(f"Model uploaded to stage {stage_path}")

                    # Register model metadata in MODEL_REGISTRY
                    session.sql(f"""
                        INSERT INTO ML_MODELS.MODEL_REGISTRY
                        (MODEL_NAME, MODEL_VERSION, MODEL_TYPE, TRAINING_DATE, METRICS, FEATURE_IMPORTANCE, MODEL_PATH)
                        VALUES ('failure_predictor', 'v1.0', 'XGBClassifier', CURRENT_TIMESTAMP(),
                            PARSE_JSON('{{"accuracy": {acc}}}'),
                            PARSE_JSON('{{}}'),
                            '{stage_path}')
                    """).collect()
                    st.success("Model registered in ML_MODELS.MODEL_REGISTRY")

                    # Create/replace Python UDF that imports the model from stage and scores
                    # Note: IMPORTS needs the object path exactly as uploaded to stage; Snowflake makes it available by basename.
                    create_udf_sql = f"""
CREATE OR REPLACE FUNCTION ML.SCORE_FAILURE(AIR_TEMP FLOAT, PROCESS_TEMP FLOAT, RPM FLOAT, TORQUE FLOAT, TOOL_WEAR FLOAT)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('scikit-learn','xgboost','joblib','numpy','pandas')
HANDLER = 'score_handler'
IMPORTS = ('@ML_MODELS.ML_STAGE/failure_predictor/failure_predictor.pkl')
AS
$$
import joblib, numpy as np
def score_handler(AIR_TEMP, PROCESS_TEMP, RPM, TORQUE, TOOL_WEAR):
    bundle = joblib.load('failure_predictor.pkl')
    scaler = bundle['scaler']
    model = bundle['model']
    features = np.array([[AIR_TEMP, PROCESS_TEMP, RPM, TORQUE, TOOL_WEAR]])
    features_s = scaler.transform(features)
    prob = model.predict_proba(features_s)[0,1]
    return float(prob)
$$;
"""
                    session.sql(create_udf_sql).collect()
                    st.success("Created ML.SCORE_FAILURE UDF for in-Snowflake scoring.")
            except Exception as e:
                st.error(f"Training/upload failed: {e}")

    # Allow ad-hoc scoring from UI using UDF
    st.markdown("**Ad-hoc scoring (call in-Snowflake UDF)**")
    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        a = st.number_input("Air temp (K)", value=300.0)
    with colB:
        b = st.number_input("Process temp (K)", value=310.0)
    with colC:
        c = st.number_input("RPM", value=1500.0)
    with colD:
        d = st.number_input("Torque", value=40.0)
    with colE:
        e_val = st.number_input("Tool wear", value=100.0)
    if st.button("Score this row (UDF)"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                # safe call to UDF
                res = session.sql(f"SELECT ML.SCORE_FAILURE({a},{b},{c},{d},{e_val}) AS PROB").to_pandas()
                st.write("Predicted failure probability:", float(res["PROB"].iloc[0]))
            except Exception as ex:
                st.error(f"UDF call failed: {ex}")

# ---------------------
# Alerts & Dashboards
# ---------------------
with tab_alerts:
    st.header("Alerts & Dashboards")
    st.markdown("This tab shows summary metrics and a simple alert generator combining flights near plants and machine wear stats.")
    if st.button("Flight summary (by country)"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                fs = session.sql("SELECT * FROM LOGISTICS.FLIGHT_SUMMARY ORDER BY TOTAL_FLIGHTS DESC LIMIT 100").to_pandas()
                st.dataframe(fs)
            except Exception as e:
                st.error(e)

    if st.button("Generate sample alert (insert into OPS.ALERTS)"):
        if not SNOWFLAKE_AVAILABLE:
            st.error("Snowflake not connected.")
        else:
            try:
                # Very simple rule: flights within 50km + avg tool wear in last 2 hours > threshold
                alert_sql = """
                WITH plant AS (
                  SELECT PLANT_ID, ST_MAKEPOINT(LONGITUDE, LATITUDE) AS PLANT_GEOM FROM LOGISTICS.PLANTS
                ), flights_near AS (
                  SELECT f.*, p.PLANT_ID
                  FROM LOGISTICS.FLIGHT_TRACKING f
                  CROSS JOIN plant p
                  WHERE ST_DISTANCE(ST_MAKEPOINT(f.LONGITUDE, f.LATITUDE), p.PLANT_GEOM) < 50000
                ), recent_wear AS (
                  SELECT MACHINE_ID, AVG(TOOL_WEAR_MIN) AS AVG_WEAR
                  FROM MANUFACTURING.SENSOR_DATA
                  WHERE TIMESTAMP > DATEADD(HOUR, -2, CURRENT_TIMESTAMP())
                  GROUP BY MACHINE_ID
                  HAVING AVG(TOOL_WEAR_MIN) > 200
                )
                INSERT INTO OPS.ALERTS (ALERT_ID, PLANT_ID, ALERT_TS, ALERT_LEVEL, ALERT_REASON, SUGGESTED_ACTION, META)
                SELECT
                  UUID_STRING(),
                  FN.PLANT_ID,
                  CURRENT_TIMESTAMP(),
                  'HIGH',
                  'Inbound flights near plant AND machines with high tool wear in last 2 hours',
                  'Prioritize inspections; delay non-critical shipments',
                  OBJECT_CONSTRUCT('flights', COUNT(*) OVER (), 'machines_high_wear', (SELECT ARRAY_AGG(MACHINE_ID) FROM recent_wear))
                FROM flights_near FN
                LIMIT 1;
                """
                session.sql(alert_sql).collect()
                st.success("Alert inserted into OPS.ALERTS (if rule conditions met).")
                alerts = session.sql("SELECT * FROM OPS.ALERTS ORDER BY ALERT_TS DESC LIMIT 20").to_pandas()
                st.dataframe(alerts)
            except Exception as e:
                st.error(e)

# ---------------------
# Admin tab: utility SQL and instructions
# ---------------------
with tab_admin:
    st.header("Admin / Utilities")
    st.markdown("Use these utilities to verify and manage database objects.")

    if st.button("Show MODEL_REGISTRY"):
        if SNOWFLAKE_AVAILABLE:
            try:
                mr = session.sql("SELECT * FROM ML_MODELS.MODEL_REGISTRY ORDER BY TRAINING_DATE DESC").to_pandas()
                st.dataframe(mr)
            except Exception as e:
                st.error(e)

    if st.button("Show ALERTS"):
        if SNOWFLAKE_AVAILABLE:
            try:
                al = session.sql("SELECT * FROM OPS.ALERTS ORDER BY ALERT_TS DESC LIMIT 200").to_pandas()
                st.dataframe(al)
            except Exception as e:
                st.error(e)

    st.markdown("**Notes**")
    st.write("- This app calls OpenSky from Streamlit (required for Snowflake trial).")
    st.write("- The app trains models in Streamlit environment and uploads artifacts to Snowflake internal stage.")
    st.write("- After uploading, the app creates a Python UDF ML.SCORE_FAILURE that imports the model from stage and is callable from SQL.")
