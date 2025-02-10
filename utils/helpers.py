import pandas as pd
import re
from typing import Any

def convert_to_snake_case(column_name: Any) -> str:
    """
    Converts a single column name to snake_case. Handles non-string column names gracefully.
    """
    if not isinstance(column_name, str):
        column_name = str(column_name)
    return re.sub(r'[^a-zA-Z0-9]+', '_', column_name).strip('_').lower()


def convert_all_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names of a DataFrame to snake_case.
    """
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    return df
