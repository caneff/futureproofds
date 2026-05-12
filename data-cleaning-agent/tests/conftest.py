import pandas as pd
import pytest

from data_cleaning_agent.utils import get_dataframe_summary


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """A small DataFrame exercising every detection branch in the summary."""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, None, 35, 40],
        "country": ["US", "US", "UK", "US", "FR"],
        "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03", None, "2024-01-05"],
        "income_str": ["$50,000", "$60,000", "$70,000", "$55,000", "$65,000"],
        "is_active": ["yes", "no", "yes", "yes", "no"],
    })


@pytest.fixture
def summary(mixed_df):
    """Cached DataFrameSummary so each test reuses one computation."""
    return get_dataframe_summary(mixed_df)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Zero-row DataFrame for empty-input edge cases."""
    return pd.DataFrame({"a": pd.Series(dtype="float64")})


@pytest.fixture
def monotonic_int_df() -> pd.DataFrame:
    """Strictly increasing int column whose name does NOT end in id/uuid."""
    return pd.DataFrame({"counter": [10, 20, 30, 40]})


@pytest.fixture
def small_numeric_df() -> pd.DataFrame:
    """Single-row numeric column to exercise skew/std NaN -> 0.0 coercion."""
    return pd.DataFrame({"x": [42.0]})
