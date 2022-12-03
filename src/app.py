import datetime

import pandas as pd
import streamlit as st
from plotly import express as px

from evaluation.metrics import MAPE
from modeling.linear_regression_predictor import LinearRegressionPredictor
from utils.time_windows_overlap import time_windows_overlap

st.set_page_config(layout="wide")

predictor = LinearRegressionPredictor()

st.title('Power Load Forecasting Dashboard @ UniCatt')

time_window = st.date_input(
    label='Select the forecast time window',
    value=(datetime.date(2022, 1, 1), datetime.date(2022, 6, 1)),
    min_value=predictor.train_end_date
)

if len(time_window) == 2:
    predictor.predict(date_from=time_window[0], date_to=time_window[1])

    st.header("Prediction")
    st.plotly_chart(predictor.get_predictions_fig(), use_container_width=True)

    st.header("Model monitoring")
    uploaded_file = st.file_uploader(label="Actual CSV data:", type="csv")

    if uploaded_file is not None:
        actual_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True).resample('D').sum()
        actual_date_from, actual_date_to = actual_df.index[0].to_pydatetime().date(), actual_df.index[
            -1].to_pydatetime().date()

        overlap = time_windows_overlap(
            (actual_date_from, actual_date_to),
            (predictor.date_from, predictor.date_to)
        )

        if overlap:
            actual_vs_predicted_df = pd.DataFrame({
                "predicted": predictor.forecast_df[overlap[0]:overlap[1]].forecast,
                "actual": actual_df[overlap[0]:overlap[1]].Load
            })

            st.plotly_chart(px.line(actual_vs_predicted_df, title="Predicted vs Actual"), use_container_width=True)

            window_size = 3
            rolling_mape_df = pd.DataFrame(
                {'rolling_mape': map(lambda window: MAPE(window.actual, window.predicted), actual_vs_predicted_df.rolling(window_size))},
                index=actual_vs_predicted_df.index
            )
            st.plotly_chart(px.line(rolling_mape_df[window_size:], title=f"{window_size} days rolling MAPE"), use_container_width=True)
        else:
            st.error("No overlap between prediction and actual data time windows")
