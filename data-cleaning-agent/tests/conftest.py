import pandas as pd
import pytest


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
