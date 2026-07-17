import pandas as pd
import io

def load_csv(file_or_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_or_path)
        
        # Auto-detect and parse datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
                    
        return df
    except Exception as e:
        raise Exception(f"Failed to load CSV: {e}")

def get_dataframe_schema(df):
    """
    Returns a string representation of the dataframe's schema to feed to the LLM.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Also get first few rows
    head_str = df.head(3).to_string()
    
    return f"Dataframe Info:\n{info_str}\n\nSample Data:\n{head_str}"
