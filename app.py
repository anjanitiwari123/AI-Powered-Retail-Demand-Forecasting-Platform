
from datetime import date, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_list.pkl")
    return model, feature_columns

@st.cache_data
def load_data():
    stores = pd.read_csv(DATA_DIR / "store.csv")
    sales = pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    performance = pd.read_csv(DATA_DIR / "model_comparison_results.csv")
    importance = pd.read_csv(DATA_DIR / "shap_feature_importance.csv")
    sales = sales[sales["Open"] == 1].sort_values(["Store", "Date"])
    sales["StateHoliday"] = sales["StateHoliday"].fillna("0").astype(str)
    return stores, sales, performance, importance


def value_or_zero(value):
    return 0 if pd.isna(value) else value

def promo2_active_in_month(store, forecast_date):
    if int(store["Promo2"]) != 1 or pd.isna(store["PromoInterval"]):
        return 0
    month_name = forecast_date.strftime("%b")
    active_months = str(store["PromoInterval"]).split(",")
    return int(month_name in active_months)


def sales_features(history, feature_date):
    def lag(days):
        return history[-days] if len(history) >= days else 0
    values = {
        "Sales_lag_1": lag(1),
        "Sales_lag_7": lag(7),
        "Sales_lag_14": lag(14),
        "Sales_lag_30": lag(30),
    }
    for window in (7, 14, 30):
        recent = history[-window:]
        values[f"Sales_roll_mean_{window}"] = np.mean(recent) if len(recent) >= 3 else 0
        values[f"Sales_roll_std_{window}"] = np.std(recent, ddof=1) if len(recent) >= 3 else 0

    values["Sales_ewm_7"] = (
        pd.Series(history[-7:]).ewm(span=7, min_periods=3).mean().iloc[-1]
        if len(history) >= 3
        else 0
    )
    return values


def create_feature_row(store_id, forecast_date, promo, school_holiday, state_holiday, history, stores, feature_columns):
    store = stores.loc[stores["Store"] == store_id].iloc[0]
    week_number = forecast_date.isocalendar().week
    competition_year = value_or_zero(store["CompetitionOpenSinceYear"])
    competition_month = value_or_zero(store["CompetitionOpenSinceMonth"])
    promo2_year = value_or_zero(store["Promo2SinceYear"])
    promo2_week = value_or_zero(store["Promo2SinceWeek"])

    competition_open_months = 0
    if competition_year and competition_month:
        competition_start = date(int(competition_year), int(competition_month), 1)
        competition_open_months = max((forecast_date - competition_start).days / 30, 0)

    promo2_open_weeks = 0
    if promo2_year and promo2_week:
        promo2_start = date.fromisocalendar(int(promo2_year), int(promo2_week), 1)
        promo2_open_weeks = max((forecast_date - promo2_start).days / 7, 0)

    row = {
        "Store": store_id,
        "DayOfWeek": forecast_date.isoweekday(),
        "Promo": int(promo),
        "SchoolHoliday": int(school_holiday),
        "CompetitionDistance": (
            stores["CompetitionDistance"].max()
            if pd.isna(store["CompetitionDistance"])
            else store["CompetitionDistance"]
        ),
        "CompetitionOpenSinceMonth": competition_month,
        "CompetitionOpenSinceYear": competition_year,
        "Promo2": int(store["Promo2"]),
        "Promo2SinceWeek": promo2_week,
        "Promo2SinceYear": promo2_year,
        "Year": forecast_date.year,
        "Month": forecast_date.month,
        "Day": forecast_date.day,
        "WeekOfYear": week_number,
        "IsWeekend": int(forecast_date.isoweekday() >= 6),
        "IsMonthStart": int(forecast_date.day == 1),
        "IsMonthEnd": int((forecast_date + timedelta(days=1)).month != forecast_date.month),
        "Month_sin": np.sin(2 * np.pi * forecast_date.month / 12),
        "Month_cos": np.cos(2 * np.pi * forecast_date.month / 12),
        "DayOfWeek_sin": np.sin(2 * np.pi * forecast_date.isoweekday() / 7),
        "DayOfWeek_cos": np.cos(2 * np.pi * forecast_date.isoweekday() / 7),
        "CompetitionOpenMonths": competition_open_months,
        "Promo2OpenWeeks": promo2_open_weeks,
        "IsPromo2Month": promo2_active_in_month(store, forecast_date),
    }
    row.update(sales_features(history, forecast_date))
    for column in feature_columns:
        if column.startswith("StoreType_"):
            row[column] = int(store["StoreType"] == column.removeprefix("StoreType_"))
        elif column.startswith("Assortment_"):
            row[column] = int(store["Assortment"] == column.removeprefix("Assortment_"))
        elif column.startswith("StateHoliday_"):
            row[column] = int(state_holiday == column.removeprefix("StateHoliday_"))

    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)


def make_forecast(model, store_id, start_date, days, promo, school_holiday, state_holiday, sales, stores, feature_columns):
    store_sales = sales.loc[sales["Store"] == store_id, "Sales"].tolist()
    records = []

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        features = create_feature_row(
            store_id, current_date, promo, school_holiday, state_holiday,
            store_sales, stores, feature_columns,
        )
        predicted_sales = max(float(model.predict(features)[0]), 0)
        records.append({"Date": current_date, "Predicted Sales": round(predicted_sales, 0)})
        store_sales.append(predicted_sales)

    return pd.DataFrame(records)


model, feature_columns = load_model()
stores, sales, performance, importance = load_data()
last_data_date = sales["Date"].max().date()

st.title("Retail Demand Forecasting")
st.caption("LightGBM-powered sales forecasts using calendar, promotion, store, and historical-demand signals.")

with st.sidebar:
    st.header("Forecast settings")
    store_id = st.selectbox("Store", sorted(stores["Store"].unique()))
    start_date = st.date_input("Forecast start date", value=last_data_date + timedelta(days=1), min_value=last_data_date + timedelta(days=1))
    days = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30)
    promo = st.toggle("Promotion active", value=True)
    school_holiday = st.toggle("School holiday", value=False)
    state_holiday = st.selectbox("State holiday", ["0", "a", "b", "c"], format_func=lambda x: {"0": "No holiday", "a": "Public holiday", "b": "Easter holiday", "c": "Christmas holiday"}[x])
    generate = st.button("Generate forecast", type="primary", use_container_width=True)

    selected_store = stores.loc[stores["Store"] == store_id].iloc[0]
    st.divider()
    st.caption("Selected store")
    st.write(f"**Type:** {selected_store['StoreType']}")
    st.write(f"**Assortment:** {selected_store['Assortment']}")
    st.write(f"**Competition distance:** {value_or_zero(selected_store['CompetitionDistance']):,.0f} m")

forecast_tab, insights_tab = st.tabs(["Forecast", "Model insights"])

with forecast_tab:
    if not generate:
        st.info("Choose the forecast settings in the sidebar and select **Generate forecast**.")
    else:
        with st.spinner("Calculating forecast..."):
            forecast = make_forecast(model, store_id, start_date, days, promo, school_holiday, state_holiday, sales, stores, feature_columns)

        peak = forecast.loc[forecast["Predicted Sales"].idxmax()]
        total, average, peak_day = st.columns(3)
        total.metric("Expected sales", f"{forecast['Predicted Sales'].sum():,.0f}")
        average.metric("Average daily sales", f"{forecast['Predicted Sales'].mean():,.0f}")
        peak_day.metric("Peak sales day", peak["Date"].strftime("%d %b"), f"€{peak['Predicted Sales']:,.0f}")

        chart = px.line(forecast, x="Date", y="Predicted Sales", markers=True, title="Forecasted sales trend")
        chart.update_traces(line_color="#2563eb")
        chart.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10), yaxis_title="Sales (€)")
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Daily forecast")
        display_forecast = forecast.copy()
        display_forecast["Date"] = pd.to_datetime(display_forecast["Date"]).dt.strftime("%d %b %Y")
        display_forecast["Predicted Sales"] = display_forecast["Predicted Sales"].map("{:,.0f}".format)
        st.dataframe(display_forecast, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", forecast.to_csv(index=False), "demand_forecast.csv", "text/csv")

with insights_tab:
    st.subheader("Model performance")
    st.caption("LightGBM is the selected production model because it achieved the lowest RMSPE in the evaluation results.")
    st.dataframe(performance.sort_values("RMSPE"), use_container_width=True, hide_index=True)

    st.subheader("Most influential drivers")
    top_features = importance.head(12).sort_values("Mean_SHAP_Impact")
    importance_chart = px.bar(top_features, x="Mean_SHAP_Impact", y="Feature", orientation="h", color_discrete_sequence=["#2563eb"])
    importance_chart.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(importance_chart, use_container_width=True)

st.caption(f"Training history available through {last_data_date.strftime('%d %b %Y')}. Forecasts beyond this date are recursive estimates.")
