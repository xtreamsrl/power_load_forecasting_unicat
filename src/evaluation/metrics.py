import numpy as np


def MAPE(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs(actual - predicted) / predicted)
