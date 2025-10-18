# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import tempfile
import joblib
import uuid
from datetime import datetime
from snowflake.snowpark import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from snowflake.snowpark.exceptions import SnowparkSQLException

st.set_page_config(page_title="AeroFloor AI", layout="wide")
st.title("AeroFloor — Predictive Logistics Meets Manufacturing")

# ---------- Snowflake session helper ----------
def get_snowflake_session():
    # Prefer st.connection (Snowflake-native Streamlit)
    try:
        cnx = st.connection("snowflake")
        session = cnx.session()
        return session
    except Exception:
        # Fallback to using secrets for local/Streamlit Cloud
        params = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "warehouse": st.secrets.get("snowflake", {}).get("warehouse", "AEROFLOOR_WH"),
            "database": st.secrets.get("snowflake", {}).get("database", "AEROFLOOR_DB"),
            "schema": st.secrets.get("snowflake", {}).get("schema", "LOGISTICS")
        }
        return Session.builder.configs(params).create()

# Try to create session
SNOWFLAKE_AVAILABLE = True
try:
    session = get_snowflake_session()
except Exception as e:
    st.error(f"Snowflake session failed: {e}")
    SNOWFLAKE_AVAILABLE = False
    session = None

# ---------- OpenSky auth ----------
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

# ---------- UI: Flights ----------
st.header("1) Live Flight Ingestion (OpenSky)")

bbox = st.text_input("Bounding box (min_lat, min_lon, max_lat, max_lon)", "8.0,-10.0,75.0,100.0")
col1, col2 = st.columns(2)
with col1:
    use_auth = st.checkbox("Use OpenSky authenticated API (recommended)", value=bool(OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET))
with col2:
    ingest_btn = st.button("Fetch & Save Flights")

if ingest_btn:
    try:
        lamin, lomin, lamax, lomax = [float(x.strip()) for x in bbox.split(",")]
        token = None
        if use_auth:
            if not (OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET):
                st.error("OpenSky credentials missing. Add to secrets or uncheck auth.")
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
        df = pd.DataFrame(rows)
        st.success(f"Fetched {len(df)} flights.")
        if len(df) > 0:
            st.map(df.rename(columns={"latitude":"lat","longitude":"lon"}))
            if SNOWFLAKE_AVAILABLE:
                sp_df = session.create_dataframe(df)
                sp_df.write.mode("append").save_as_table("LOGISTICS.FLIGHT_TRACKING")
                st.info("Saved to LOGISTICS.FLIGHT_TRACKING")
            else:
                st.info("Snowflake not available — showing preview only.")
                st.dataframe(df.head())
    except Exception as e:
        st.error(f"Ingest failed: {e}")

if st.button("Show recent flights (Snowflake)"):
    if not SNOWFLAKE_AVAILABLE:
        st.error("Snowflake not connected")
    else:
        try:
            df = session.sql("SELECT * FROM LOGISTICS.FLIGHT_TRACKING ORDER BY INGESTED_AT DESC LIMIT 200").to_pandas()
            st.dataframe(df)
            if len(df) > 0:
                st.map(df.rename(columns={"latitude":"lat","longitude":"lon"}))
        except Exception as e:
            st.error(e)

# ---------- UI: Manufacturing ingestion ----------
st.header("2) Manufacturing Data (UCI AI4I)")

if st.button("Download & Ingest UCI AI4I"):
    try:
        csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        df = pd.read_csv(csv_url)
        df_prepared = pd.DataFrame({
            "machine_id": df["Product ID"].astype(str).str[:6],
            "product_id": df["Product ID"].astype(str),
            "type": df["Type"].astype(str),
            "air_temperature_k": df["Air temperature [K]"].astype(float),
            "process_temperature_k": df["Process temperature [K]"].astype(float),
            "rotational_speed_rpm": df["Rotational speed [rpm]"].astype(float),
            "torque_nm": df["Torque [Nm]"].astype(float),
            "tool_wear_min": df["Tool wear [min]"].astype(float),
            "machine_failure": df["Machine failure"].astype(bool),
            "twf": df["TWF"].astype(bool),
            "hdf": df["HDF"].astype(bool),
            "pwf": df["PWF"].astype(bool),
            "osf": df["OSF"].astype(bool),
            "rnf": df["RNF"].astype(bool),
            "timestamp": pd.Timestamp.utcnow() - pd.to_timedelta(np.arange(len(df))*5, unit="m")
        })
        if SNOWFLAKE_AVAILABLE:
            sp_df = session.create_dataframe(df_prepared)
            sp_df.write.mode("append").save_as_table("MANUFACTURING.SENSOR_DATA")
            st.success(f"Inserted {len(df_prepared)} rows into MANUFACTURING.SENSOR_DATA")
        else:
            st.dataframe(df_prepared.head())
    except Exception as e:
        st.error(f"Failed to ingest UCI dataset: {e}")

if st.button("Preview Sensor Data"):
    if not SNOWFLAKE_AVAILABLE:
        st.error("Snowflake not connected")
    else:
        try:
            df = session.sql("SELECT * FROM MANUFACTURING.SENSOR_DATA LIMIT 200").to_pandas()
            st.dataframe(df)
        except Exception as e:
            st.error(e)

# ---------- UI: Train model ----------
st.header("3) Train Failure Model & Upload")

if st.button("Train & Upload Model"):
    if not SNOWFLAKE_AVAILABLE:
        st.error("Snowflake not connected")
    else:
        try:
            query = """
            SELECT AIR_TEMPERATURE_K, PROCESS_TEMPERATURE_K, ROTATIONAL_SPEED_RPM,
                   TORQUE_NM, TOOL_WEAR_MIN,
                   CASE WHEN TWF OR HDF OR PWF OR OSF OR RNF THEN 1 ELSE 0 END AS TARGET
            FROM MANUFACTURING.SENSOR_DATA
            LIMIT 5000
            """
            df = session.sql(query).to_pandas()
            if df.shape[0] < 50:
                st.error("Not enough rows to train (>=50 required).")
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
                st.success(f"Trained model. Test accuracy: {acc:.3f}")

                tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                joblib.dump({"model": model, "scaler": scaler, "features": X.columns.tolist()}, tmp.name)

                # upload to stage
                session.file.put(tmp.name, "@ML_MODELS.ML_STAGE/failure_predictor/", auto_compress=False, overwrite=True)
                session.sql(f"""
                    INSERT INTO ML_MODELS.MODEL_REGISTRY
                    (MODEL_NAME, MODEL_VERSION, MODEL_TYPE, TRAINING_DATE, METRICS, FEATURE_IMPORTANCE, MODEL_PATH)
                    VALUES
                    ('failure_predictor', 'v1.0', 'XGBClassifier', CURRENT_TIMESTAMP(),
                     PARSE_JSON('{{"accuracy": {acc}}}'), PARSE_JSON('{{}}'),
                     '@ML_MODELS.ML_STAGE/failure_predictor/{tmp.name.split('/')[-1]}')
                """).collect()
                st.info("Model uploaded and registered in MODEL_REGISTRY")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------- UI: Alerts & Prescriptions ----------
st.header("4) Alerts & Prescriptive Insights (preview)")

if st.button("Run sample alert generation"):
    if not SNOWFLAKE_AVAILABLE:
        st.error("Snowflake not connected")
    else:
        try:
            # Example rule: if any flight within 50km of a plant AND any machine with tool_wear > threshold recently -> alert
            alert_sql = """
            WITH plant AS (
                SELECT PLANT_ID, ST_MAKEPOINT(LONGITUDE, LATITUDE) AS PLANT_GEOM FROM LOGISTICS.PLANTS
            ), flights_near AS (
                SELECT f.*, p.PLANT_ID
                FROM LOGISTICS.FLIGHT_TRACKING f
                CROSS JOIN plant p
                WHERE ST_DISTANCE(ST_MAKEPOINT(f.LONGITUDE, f.LATITUDE), p.PLANT_GEOM) < 50000
            ), bad_machines AS (
                SELECT MACHINE_ID, AVG(TOOL_WEAR_MIN) AS TOOL_WEAR_AVG
                FROM MANUFACTURING.SENSOR_DATA
                WHERE TIMESTAMP > DATEADD(HOUR, -2, CURRENT_TIMESTAMP())
                GROUP BY MACHINE_ID
                HAVING AVG(TOOL_WEAR_MIN) > 200
            )
            INSERT INTO OPS.ALERTS (ALERT_ID, PLANT_ID, ALERT_TS, ALERT_LEVEL, ALERT_REASON, SUGGESTED_ACTION, META)
            SELECT
                UUID_STRING() as ALERT_ID,
                fn.PLANT_ID,
                CURRENT_TIMESTAMP(),
                'HIGH',
                'Inbound flight near plant + high tool wear on machines',
                'Consider delaying non-critical shipments / expedite maintenance',
                OBJECT_CONSTRUCT('flight_count', COUNT(*) OVER (), 'machines', (SELECT ARRAY_AGG(MACHINE_ID) FROM bad_machines))
            FROM flights_near fn
            LIMIT 1;
            """
            session.sql(alert_sql).collect()
            st.success("Sample alert generated (inserted into OPS.ALERTS).")
            alerted = session.sql("SELECT * FROM OPS.ALERTS ORDER BY ALERT_TS DESC LIMIT 20").to_pandas()
            st.dataframe(alerted)
        except Exception as e:
            st.error(e)
