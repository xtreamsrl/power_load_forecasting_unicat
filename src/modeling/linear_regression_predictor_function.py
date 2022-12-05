import datetime
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

from preprocessing.fourier_features import create_fourier_features
from preprocessing.holiday_features import create_holiday_features

MODEL_PICKLE_FILE_PATH = "data/linear_regression.pkl"
DATE_ORIGIN = datetime.date(2006, 1, 1)
WEEKLY_HARMONICS = 5
YEARLY_HARMONICS = 20

model: LinearRegression = pickle.load(open(MODEL_PICKLE_FILE_PATH, "rb"))


def linear_regression_predict(date_from: datetime.date, date_to: datetime.date) -> pd.DataFrame:
    # Compute features
    date_range = pd.date_range(start=date_from, end=date_to)
    dates = pd.DatetimeIndex(date_range)
    holidays_features = create_holiday_features(dates=dates)
    fourier_features = create_fourier_features(
        dates=dates,
        date_origin=DATE_ORIGIN,
        yearly_harmonics=YEARLY_HARMONICS,
        weekly_harmonics=WEEKLY_HARMONICS,
    )
    # Join all features
    regression_df = holidays_features.join(fourier_features)
    forecast = model.predict(regression_df)
    return pd.DataFrame({"forecast": forecast}, index=regression_df.index)
