import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from validator import validate_and_fix

def run_secure_code(code_str: str, df: pd.DataFrame):
    """
    Executes Python code in a restricted environment.
    The code is expected to process the DataFrame `df` and optionally create a plot.
    If it creates a Plotly figure, it should assign it to a variable named `fig`.
    If it creates a Matplotlib figure, we will capture it.
    The function returns a dictionary with 'result', 'fig' (if any), and 'error' (if any).
    """
    # Define the restricted environment
    # We remove dangerous built-ins and only provide safe data manipulation tools
    safe_builtins = {
        'print': print,
        'len': len,
        'range': range,
        'list': list,
        'dict': dict,
        'set': set,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'isinstance': isinstance,
        'type': type,
        '__import__': __import__  # needed for some pandas/plotly internals, but we restrict globals
    }
    
    # Allowed modules
    restricted_globals = {
        '__builtins__': safe_builtins,
        'pd': pd,
        'np': np,
        'px': px,
        'go': go,
        'plt': plt,
    }
    
    # Local variables available to the executed code
    df_copy = df.copy()
    restricted_locals = {
        'df': df_copy,
        'fig': None,
        'result_df': None
    }
    
    output = {
        'result': None,
        'fig': None,
        'error': None,
        'printed_output': None
    }
    
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    
    try:
        # Clear any existing matplotlib figures
        plt.clf()
        plt.close('all')
        
        with redirect_stdout(f):
            # Validate and auto-fix common AI code errors before execution
            clean_code = validate_and_fix(code_str, df)
            exec(clean_code, restricted_globals, restricted_locals)
        
        output['printed_output'] = f.getvalue().strip()
        
        # Check if the code produced a figure or a modified dataframe
        if restricted_locals.get('fig') is not None:
            output['fig'] = restricted_locals['fig']
        elif len(plt.get_fignums()) > 0:
             # Matplotlib figure was created
             output['fig'] = plt.gcf()
             
        if restricted_locals.get('result_df') is not None:
            output['result'] = restricted_locals['result_df']
        elif restricted_locals['df'] is not df_copy:
            output['result'] = restricted_locals['df']
        else:
            # Fallback: find any variable that is a DataFrame or Series (other than 'df')
            for key, val in restricted_locals.items():
                if key not in ['df', 'fig', 'result_df', '__builtins__']:
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        output['result'] = val
                        break
            
            # If no new DataFrame was created, but they mutated df_copy in-place
            if output['result'] is None and not df_copy.equals(df):
                output['result'] = df_copy
            
        # Enforce that the LLM must assign both result_df and fig
        if output['result'] is None:
            output['error'] = "NameError: You did not assign the final tabular data to the 'result_df' variable. Please define 'result_df = ...' before the end of the script."
        elif output['fig'] is None:
            output['error'] = "NameError: You did not assign a visualization to the 'fig' variable. Please define 'fig = px.bar(...)' (or another Plotly chart) representing the results."

    except Exception as e:
        import traceback
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        tb_str = "".join(tb_lines)
        
        err_msg = str(e)
        err_type = type(e).__name__
        if err_type == "KeyError":
            # Extract actual missing keys from the original Pandas KeyError string
            import re
            raw_key_str = str(e)
            missing_keys = []
            m_index = re.search(r"Index\(\[([^\]]+)\]", raw_key_str)
            if m_index:
                extracted = re.findall(r"[\x27\x22]([^\x27\x22]+)[\x27\x22]", m_index.group(1))
                missing_keys.extend([c for c in extracted if c != "object"])
            if not missing_keys:
                extracted = re.findall(r"[\x27\x22]([^\x27\x22]+)[\x27\x22]", raw_key_str)
                for k in extracted:
                    k_clean = k.replace("[", "").replace("]", "").strip()
                    if k_clean and "not in index" not in k_clean and "columns" not in k_clean:
                        for sub_k in re.split(r"[,\s]+", k_clean):
                            sub_k = sub_k.strip("\x27\x22")
                            if sub_k:
                                missing_keys.append(sub_k)
            if not missing_keys:
                clean_k = raw_key_str.strip("\x27\x22[]")
                if "not in index" in clean_k:
                    clean_k = clean_k.split("not in index")[0].strip().strip("\x27\x22[]")
                if clean_k:
                    missing_keys.append(clean_k)
            
            # Format user-facing error message
            err_msg = f"KeyError: The column or index key {err_msg} was not found in the DataFrame/Series columns/index. Ensure you are accessing existing columns."
            
            # Check for DataFrame merge suffix (_x, _y) mismatch using extracted keys
            suffix_hint = ""
            for key_name in missing_keys:
                for var_name, val in restricted_locals.items():
                    if isinstance(val, pd.DataFrame):
                        cols = list(val.columns)
                        if f"{key_name}_x" in cols or f"{key_name}_y" in cols:
                            suffix_hint = f" (Hint: The column '{key_name}' was not found in DataFrame '{var_name}', but '{key_name}_x' and '{key_name}_y' were found. This usually happens when you merge two DataFrames that both contain '{key_name}' without including it in the 'on' merge keys. Either add '{key_name}' to the 'on' list in pd.merge(), or reference '{key_name}_x'/'{key_name}_y'.)"
                            break
                if suffix_hint:
                    break
            
            # Check for columns lost during groupby/aggregation using extracted keys
            groupby_hint = ""
            original_cols = list(df.columns)
            if 'dfc' in restricted_locals and isinstance(restricted_locals['dfc'], pd.DataFrame):
                original_cols.extend(list(restricted_locals['dfc'].columns))
                
            for key_name in missing_keys:
                for var_name, val in restricted_locals.items():
                    if isinstance(val, pd.DataFrame) and var_name not in ['df', 'df_copy', 'dfc']:
                        if key_name not in val.columns and key_name in original_cols:
                            groupby_hint = f" (Hint: The column '{key_name}' exists in the original DataFrame 'df'/'dfc', but was not found in '{var_name}'. This usually happens because '{var_name}' is the result of a groupby or aggregation that discarded the '{key_name}' column. You should apply your filter (e.g. for '{key_name}') on 'dfc' BEFORE the groupby, or include '{key_name}' in your groupby keys.)"
                            break
                if groupby_hint:
                    break
            
            if suffix_hint:
                err_msg += suffix_hint
            elif groupby_hint:
                err_msg += groupby_hint
            # Check if this KeyError occurred during a merge
            elif 'merge.py' in tb_str or '_get_merge_keys' in tb_str:
                err_msg += " (Hint: This KeyError occurred during a pandas merge/join operation. Ensure that all keys specified in 'on', 'left_on', or 'right_on' are actually present in BOTH DataFrames. If you subsetted one of the DataFrames right before merging (e.g., df[['col1', 'col2']]), make sure the join key is included in that subset.)"
            # Check for generic placeholders
            elif any(placeholder in err_msg for placeholder in ["category_col", "value_col", "date_col", "group_col", "region_col"]):
                err_msg += f" (Hint: You used a generic placeholder column name from the prompt examples. You must replace it with the ACTUAL column name from the dataset schema, such as: {list(df.columns)})"
        elif "Grouper and axis must be same length" in err_msg:
            err_msg = f"ValueError: {err_msg} (Hint: This error occurs when a column name in your groupby list does not exist in the DataFrame. Ensure that all groupby keys are actual columns, or if they are computed columns like 'year', that you define them first: e.g., dfc['year'] = dfc['order_date'].dt.year)"
        elif "Length mismatch" in err_msg:
            err_msg = f"ValueError: {err_msg} (Hint: You attempted to rename columns by assigning a list of names to df.columns (or result_df.columns), but the length of your list does not match the actual number of columns in the DataFrame. Ensure that the list of names matches the number of columns generated by your groupby/aggregations.)"
        elif "All arguments should have the same length" in err_msg:
            err_msg = f"ValueError: {err_msg} (Hint: In Plotly Express (px.bar, px.line, etc.), the 'color' argument must be either a column name in the DataFrame or a list/array with the EXACT same length as the DataFrame rows. If you want to color by a constant value or name, DO NOT pass a single-element list like color=['value']. Instead, omit the color parameter or add a constant column to your DataFrame first: e.g. dfc['color_group'] = 'value' and then pass color='color_group'.)"
        elif "min() iterable argument is empty" in err_msg:
            err_msg = f"ValueError: {err_msg} (Hint: This error occurs because you tried to calculate the minimum of an empty list/Series. If you filtered df to only 2024 at the beginning of the script, you cannot calculate YoY growth or shift() from 2023 because the 2023 data was discarded. Make sure you calculate yearly aggregates BEFORE filtering for 2024, similar to Example 5 in the prompt.)"
        elif "unexpected keyword argument 'abs'" in err_msg:
            err_msg = f"TypeError: {err_msg} (Hint: sort_values() does not take an 'abs' parameter. If you want to sort by absolute value, use: key=lambda x: x.abs())"
        
        output['error'] = f"{err_type}: {err_msg}" if not err_msg.startswith(err_type) else err_msg
    
    return output
