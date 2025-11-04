import pandas as pd

def sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and mixed types to safe strings for Streamlit display."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%b %d, %Y")
        elif pd.api.types.is_object_dtype(df[col]):
            if not all(isinstance(x, str) or pd.isna(x) for x in df[col]):
                df[col] = df[col].astype(str)
    return df
