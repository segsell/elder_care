"""Tests for data mergeing and preparation of the estimation data set."""
# def test_merge_waves();
# def is_weakly_increasing(series):
# # Group the data by mergeid and apply the is_weakly_increasing function
# # Assert that all groups have weakly increasing work experience
# def is_weakly_increasing(series):
# # Group the data by mergeid and apply the is_weakly_increasing function
# # Assert that all groups have weakly increasing work experience
import pandas as pd
from elder_care.config import BLD


def test_data_merged():
    data = pd.read_csv(BLD / "data" / "data_merged.csv")

    assert data.shape == (26593, 291)
