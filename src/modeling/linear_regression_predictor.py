import datetime
import pickle

import pandas as pd
import plotly.graph_objs
from plotly import express as px
from sklearn.linear_model import LinearRegression

from preprocessing.fourier_features import create_fourier_features
from preprocessing.holiday_features import create_holiday_features


class LinearRegressionPredictor:
    model_pkl_file_path = 'data/linear_regression.pkl'
    date_origin = datetime.date(year=2006, month=1, day=1)
    train_end_date = datetime.date(year=2020, month=12, day=31)
    weekly_harmonics = 5
    yearly_harmonics = 20

    def __init__(self) -> None:
        self.date_to = None
        self.date_from = None
        self.forecast_df = None
        self.model: LinearRegression = pickle.load(open(self.model_pkl_file_path, 'rb'))

    def _compute_features(self, date_from: datetime.date, date_to: datetime.date) -> pd.DataFrame:
        date_range = pd.date_range(date_from, date_to)
        dates = pd.DatetimeIndex(date_range)
        holidays_features = create_holiday_features(dates=dates)
        fourier_features = create_fourier_features(
            dates=dates,
            date_origin=self.date_origin,
            yearly_harmonics=self.yearly_harmonics,
            weekly_harmonics=self.weekly_harmonics,
        )
        # Join all features together and return
        return holidays_features.join(fourier_features)

    def predict(self, date_from: datetime.datetime, date_to: datetime.datetime) -> pd.DataFrame:
        self.date_from = date_from
        self.date_to = date_to
        regression_df = self._compute_features(date_from=date_from, date_to=date_to)
        forecast = self.model.predict(regression_df)
        self.forecast_df = pd.DataFrame({"forecast": forecast}, index=regression_df.index)
        return self.forecast_df

    def get_predictions_fig(self, show_legend: bool = False) -> plotly.graph_objs.Figure:
        return px.line(self.forecast_df) \
            .update_layout(showlegend=show_legend)
