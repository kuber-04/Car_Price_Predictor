
import os

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear_regression_model.pkl")

# LabelEncoder alphabetical order me encode karta hai, notebook me same mapping bani thi.
# (Brand: Audi=0, BMW=1, Mercedes-Benz=2 ... jaisa notebook output me dikha)
BRANDS = ["Audi", "BMW", "Mercedes-Benz", "Mitsubishi", "Renault", "Toyota", "Volkswagen"]
BODIES = ["crossover", "hatch", "other", "sedan", "vagon", "van"]
ENGINE_TYPES = ["Diesel", "Gas", "Other", "Petrol"]
REGISTRATION = ["no", "yes"]

# Training ke time X ke columns ka exact order (model isi order me train hua tha)
FEATURE_ORDER = ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def encode(value, options):
    # sorted list me index = LabelEncoder ka code
    return sorted(options).index(value)


st.title("🚗 Used Car Price Predictor")
st.caption("Linear Regression model | Target: log(price) → neeche actual price me convert kiya gaya hai")

try:
    model = load_model()
except Exception as e:
    st.error(
        "Model load nahi ho paya. `linear_regression_model.pkl` ko app.py ke saath same folder me rakho.\n\n"
        f"Error: {e}"
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRANDS)
    body = st.selectbox("Body type", BODIES, index=BODIES.index("sedan"))
    engine_type = st.selectbox("Engine type", ENGINE_TYPES, index=ENGINE_TYPES.index("Petrol"))
    registration = st.selectbox("Registered?", REGISTRATION, index=1)

with col2:
    year = st.number_input("Year", min_value=1969, max_value=2016, value=2008, step=1)
    mileage = st.number_input("Mileage (in 1000 km)", min_value=0, max_value=1000, value=150, step=1)
    engine_v = st.number_input("Engine volume (litres)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

if st.button("Predict Price", type="primary", use_container_width=True):
    row = {
        "Brand": encode(brand, BRANDS),
        "Body": encode(body, BODIES),
        "Mileage": mileage,
        "EngineV": engine_v,
        "Engine Type": encode(engine_type, ENGINE_TYPES),
        "Registration": encode(registration, REGISTRATION),
        "Year": year,
    }
    X_new = pd.DataFrame([row])[FEATURE_ORDER]

    log_price = float(model.predict(X_new)[0])
    price = float(np.exp(log_price))  # log wapas normal price me

    st.success("Prediction ready!")
    st.metric("Estimated Price", f"$ {price:,.0f}")
    with st.expander("Details"):
        st.write(f"Predicted log(price): `{log_price:.4f}`")
        st.dataframe(X_new)

st.divider()
st.caption("Note: yeh sirf ek estimate hai. Model 1969–2016 ke used car data par train hua hai.")
