import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Air Quality Dashboard", layout="centered")
st.title("Dashboard de Calidad del Aire")

# 1) Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv(
        "AirQuality_Clean_Normalized.csv",
        sep=";",
        decimal=",",
        encoding="utf-8",
        index_col=0,
        parse_dates=True,
    )
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday  # 0=Lunes … 6=Domingo
    return df

df = load_data()

# 2) Filtro de fechas
start, end = st.date_input(
    "Rango de fechas",
    value=[df["date"].min(), df["date"].max()],
    min_value=df["date"].min(),
    max_value=df["date"].max(),
)
df_f = df[(df["date"] >= start) & (df["date"] <= end)]

if df_f.empty:
    st.stop()

# 3) Vista rápida
st.subheader("Muestra de datos filtrados")
st.write(df_f[["CO(GT)", "NO2(GT)", "hour", "weekday"]].head())

# 4) Serie temporal
st.subheader("Tendencia CO y NO₂")
fig = px.line(df_f, x=df_f.index, y=["CO(GT)", "NO2(GT)"])
st.plotly_chart(fig, use_container_width=True)

# 5) Histogramas
st.subheader("Distribución")
long = df_f[["CO(GT)", "NO2(GT)"]].melt(var_name="variable", value_name="value")
fig = px.histogram(long, x="value", color="variable", barmode="overlay", nbins=30, opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

# 6) Scatter
st.subheader("Correlación CO vs NO₂")
fig = px.scatter(df_f, x="CO(GT)", y="NO2(GT)", opacity=0.5)
st.plotly_chart(fig, use_container_width=True)

# 7) Mapa de calor hora-día
heat = df_f.pivot_table(index="hour", columns="weekday", values="CO(GT)", aggfunc="mean")
fig = px.imshow(
    heat,
    labels=dict(x="Día semana", y="Hora", color="CO"),
    x=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
    aspect="auto",
    color_continuous_scale="Viridis",
)
st.plotly_chart(fig, use_container_width=True)

# 8) Estadísticos
st.subheader("Estadísticos descriptivos")
st.write(df_f[["CO(GT)", "NO2(GT)"]].describe())

# 9) Predicción simple
st.subheader("Predicción CO(GT) desde NO₂")
no2_val = st.number_input("NO₂ (normalizado):", 0.0, 1.0, 0.5)
model = LinearRegression().fit(df_f[["NO2(GT)"]], df_f["CO(GT)"])
pred = model.predict([[no2_val]])[0]
st.write(f"CO(GT) estimado: **{pred:.3f}**")