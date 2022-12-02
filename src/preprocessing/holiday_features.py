import holidays
import pandas as pd


def create_holiday_features(dates: pd.DatetimeIndex):
    """Create holiday series, valued 0 or 1 depending on the day"""
    holiday_cal = holidays.Italy()
    features_df = pd.DataFrame(index=dates)
    features_df['holiday'] = dates.to_series().apply(lambda s: s in holiday_cal).astype(int)
    return features_df
