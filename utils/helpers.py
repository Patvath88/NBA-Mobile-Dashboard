import pandas as pd

def sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans DataFrame for Streamlit (fixes Arrow conversion issues)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df
