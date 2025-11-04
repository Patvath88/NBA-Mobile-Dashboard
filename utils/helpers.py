import pandas as pd

def sanitize_dataframe_for_streamlit(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df
