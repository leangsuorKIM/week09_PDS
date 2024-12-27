import streamlit as st
import pandas as pd
import numpy as np
import pickle

cylinders = st.number_input(label="cylinders", value=8.0)
displacement = st.number_input(label="displacement", value=307.0)
horsepower = st.number_input(label="horsepower", value=130.0)
weight = st.number_input(label="weight", value=3504.0)
acceleration = st.number_input(label="acceleration", value=12.0)
model_year = st.number_input(label="model_year", value=70)
origin = st.selectbox(
    label="origin",
    options=["usa", "europe", "japan"],
    # index=None,
    # placeholder='Select origin'
)
X_num = np.array(
    object=[
        [
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            model_year,
        ]
    ],
    dtype=np.float32,
)
with open(file="ss.pkl", mode="rb") as ss_file:
    ss = pickle.load(file=ss_file)
X1 = ss.transform(X_num)
# st.write(X_num)

with open(file="le.pkl", mode="rb") as le_file:
    le = pickle.load(file=le_file)
X_cat = np.array(object=[origin])
X2 = le.transform(X_cat)
# st.write(X_cat)
X = np.concat([X1, X2.reshape(-1, 1)], axis=1)
# st.write(X)
with open(file="lr.pkl", mode="rb") as lr_file:
    lr = pickle.load(file=lr_file)
y = lr.predict(X)
# st.write(1/y)
X_raw = np.concat([X_num, X_cat.reshape(-1, 1)], axis=1)
y_raw = 1 / y

data = np.concat([X_raw, y_raw.reshape(-1, 1)], axis=1)
df = pd.DataFrame(
    data=data,
    columns=[
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "mpg_pred"
    ],
)
st.write(df)