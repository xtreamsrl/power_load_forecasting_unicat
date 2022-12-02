import datetime

import pandas as pd
import numpy as np


def create_fourier_features(dates: pd.DatetimeIndex, date_origin: datetime.date, weekly_harmonics: int, yearly_harmonics: int):
    """Create fourier terms as trigonometric functions of various frequencies"""
    features_df = pd.DataFrame(index=dates)

    # Turn dates into integer incremental values to be used as the x-axis
    features_df['trend'] = (dates - pd.to_datetime(date_origin)).days

    for i in range(1, weekly_harmonics + 1):
        features_df[f'sin_7_{i}'] = np.sin(2 * np.pi * i / 7 * features_df['trend'])
        features_df[f'cos_7_{i}'] = np.cos(2 * np.pi * i / 7 * features_df['trend'])
    for i in range(1, yearly_harmonics + 1):
        features_df[f'sin_365_{i}'] = np.sin(2 * np.pi * i / 365.25 * features_df['trend'])
        features_df[f'cos_365_{i}'] = np.cos(2 * np.pi * i / 365.25 * features_df['trend'])

    return features_df
