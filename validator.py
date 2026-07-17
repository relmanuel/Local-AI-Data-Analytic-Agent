import re
import pandas as pd


def validate_and_fix(code: str, df: pd.DataFrame = None) -> str:
    """
    Validates and auto-fixes common errors in AI-generated pandas/plotly code
    before it is executed in the sandbox. Returns the corrected code.
    """
    code = _remove_dummy_data(code)
    if df is not None:
        code = _fix_placeholder_columns(code, df)
        code = _fix_missing_time_cols(code, df)
    code = _fix_multi_column_groupby(code)
    code = _fix_agg_multiindex(code)
    code = _fix_diff_shift(code)
    code = _remove_show_calls(code)
    return code


def _remove_dummy_data(code: str) -> str:
    """
    Removes any block where the model tries to create its own DataFrame
    (e.g., df = pd.DataFrame({...}) or df = pd.DataFrame(data)).
    The real `df` is already available in the execution environment.
    """
    # Pattern: df = pd.DataFrame( ... ) spanning multiple lines
    # We remove from the line that does `df = pd.DataFrame(` up to the closing `)`
    # that matches the opening paren, then the optional `.copy()` chain.
    lines = code.split('\n')
    cleaned = []
    skip = False
    paren_depth = 0

    for line in lines:
        stripped = line.strip()

        # Detect start of dummy df creation
        if not skip and re.search(r'\bdf\s*=\s*pd\.DataFrame\s*\(', stripped):
            skip = True
            paren_depth = line.count('(') - line.count(')')
            # If paren_depth <= 0, it closed on the same line
            if paren_depth <= 0:
                skip = False
            continue

        # Also detect: data = { ... } ; df = pd.DataFrame(data)
        if not skip and re.search(r'\bdata\s*=\s*\{', stripped):
            skip = True
            paren_depth = line.count('{') - line.count('}')
            if paren_depth <= 0:
                skip = False
            continue

        if skip:
            paren_depth += line.count('(') - line.count(')')
            paren_depth += line.count('{') - line.count('}')
            if paren_depth <= 0:
                skip = False
            continue

        cleaned.append(line)

    # Also strip any remaining single-line df = pd.DataFrame(data) after data block was removed
    result = '\n'.join(cleaned)
    result = re.sub(r'\bdf\s*=\s*pd\.DataFrame\s*\(\s*data\s*\)', '', result)

    return result


def _fix_multi_column_groupby(code: str) -> str:
    """
    Fixes outdated pandas multi-column selection after groupby.
    e.g. .groupby('x')['col1', 'col2'] -> .groupby('x')[['col1', 'col2']]
    """
    # Match [...] containing comma-separated quoted strings (not already double-bracketed)
    # only when preceded by a closing parenthesis or bracket.
    code = re.sub(
        r'([)\]])\s*(?<!\[)\[([\'"][^\'"]+[\'"](?:,\s*[\'"][^\'"]+[\'"])+)\](?!\])',
        r'\1[[\2]]',
        code
    )
    return code


def _remove_show_calls(code: str) -> str:
    """Removes fig.show() and plt.show() calls which break the sandbox."""
    code = re.sub(r'\bfig\.show\s*\(\s*\)\s*\n?', '', code)
    code = re.sub(r'\bplt\.show\s*\(\s*\)\s*\n?', '', code)
    return code


def _fix_diff_shift(code: str) -> str:
    """
    Removes dangerous .diff().shift(-1) or .diff().shift(1) chaining.
    The .shift() applied after a grouped .diff() operates globally (not per-group),
    causing 'Grouper and axis must be same length' errors. Strip the .shift() call.
    """
    code = re.sub(r'\.diff\(\)\.shift\([^)]*\)', '.diff()', code)
    return code


def _fix_agg_multiindex(code: str) -> str:
    """
    Fixes .agg({'col': ['func']}) which creates MultiIndex columns.
    Converts list-valued agg specs to string: {'col': ['sum']} -> {'col': 'sum'}.
    e.g. .agg({'price': ['sum']}) -> .agg({'price': 'sum'})
    """
    # Match: 'col_name': ['func_name'] inside .agg({...})
    # Replace the list wrapper with just the string
    def replace_list_agg(m):
        quote1 = m.group(1)
        col = m.group(2)
        quote2 = m.group(3)
        func = m.group(4)
        return f"{quote1}{col}{quote1}: {quote2}{func}{quote2}"

    pattern = r"""(['"])([\w]+)\1\s*:\s*\[\s*(['"])([\w]+)\3\s*\]"""
    code = re.sub(pattern, replace_list_agg, code)
    return code


def _fix_placeholder_columns(code: str, df: pd.DataFrame) -> str:
    """
    Replaces generic column placeholders like category_col, value_col,
    date_col, etc. with actual columns found in df.
    """
    def find_column(candidates, dtype_filter=None):
        for cand in candidates:
            for col in df.columns:
                if col.lower() == cand.lower():
                    return col
        if dtype_filter:
            for col in df.columns:
                try:
                    if dtype_filter(df[col]):
                        return col
                except:
                    pass
        return df.columns[0] if len(df.columns) > 0 else None

    # Find actual columns matching placeholders
    actual_date = find_column(['order_date', 'date', 'time', 'timestamp', 'created_at'], pd.api.types.is_datetime64_any_dtype)
    actual_cat = find_column(['category', 'type', 'name', 'group', 'genre', 'class', 'item', 'product'], lambda x: x.dtype == 'object' or isinstance(x.dtype, pd.CategoricalDtype))
    actual_val = find_column(['price', 'value', 'amount', 'total_amount', 'sales', 'quantity', 'revenue'], pd.api.types.is_numeric_dtype)
    actual_group = find_column(['group', 'category', 'type', 'segment'], lambda x: x.dtype == 'object' or isinstance(x.dtype, pd.CategoricalDtype))
    actual_region = find_column(['region', 'location', 'country', 'city', 'state'], lambda x: x.dtype == 'object' or isinstance(x.dtype, pd.CategoricalDtype))

    # Mapping of placeholders to actual columns
    mappings = {
        'date_col': actual_date,
        'category_col': actual_cat,
        'value_col': actual_val,
        'group_col': actual_group,
        'region_col': actual_region
    }
    
    for placeholder, actual in mappings.items():
        if actual:
            # Replace quoted placeholder: 'category_col' -> 'category'
            code = re.sub(rf'["\']{placeholder}["\']', f"'{actual}'", code)
            # Replace raw placeholder: category_col -> 'category'
            code = re.sub(rf'\b{placeholder}\b', f"'{actual}'", code)
            # Clean up double quotes/nested quotes if any, e.g. ''category''
            code = re.sub(rf"''{actual}''", f"'{actual}'", code)
            code = re.sub(rf'""{actual}""', f"'{actual}'", code)
            
    return code


def _fix_missing_time_cols(code: str, df: pd.DataFrame) -> str:
    """
    If the code uses 'year', 'month', or 'quarter' but does not define them,
    injects their definitions using the identified datetime column.
    """
    # Find datetime column in df
    dt_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not dt_cols:
        for col in ['order_date', 'date', 'Date', 'Order Date']:
            if col in df.columns:
                dt_cols.append(col)
                break
    if not dt_cols:
        return code
        
    date_col = dt_cols[0]
    
    # Check if time components are used but not defined
    has_year = re.search(r'["\']\byear\b["\']', code)
    defined_year = re.search(r'["\']\byear\b["\']\s*\]\s*(?<![!=<>])=(?![=])', code) or re.search(r'\byear\s*(?<![!=<>])=(?![=])', code)
    
    has_month = re.search(r'["\']\bmonth\b["\']', code)
    defined_month = re.search(r'["\']\bmonth\b["\']\s*\]\s*(?<![!=<>])=(?![=])', code) or re.search(r'\bmonth\s*(?<![!=<>])=(?![=])', code)

    has_quarter = re.search(r'["\']\bquarter\b["\']', code)
    defined_quarter = re.search(r'["\']\bquarter\b["\']\s*\]\s*(?<![!=<>])=(?![=])', code) or re.search(r'\bquarter\s*(?<![!=<>])=(?![=])', code)

    # Determine DataFrame variable name (dfc or df)
    df_var = 'dfc' if 'dfc' in code else 'df'
    
    injections = []
    if has_year and not defined_year:
        injections.append(f"{df_var}['year'] = {df_var}['{date_col}'].dt.year")
    if has_month and not defined_month:
        injections.append(f"{df_var}['month'] = {df_var}['{date_col}'].dt.month")
    if has_quarter and not defined_quarter:
        injections.append(f"{df_var}['quarter'] = {df_var}['{date_col}'].dt.quarter")
        
    if injections:
        # Try to insert after dfc = df.copy() or dfc = df
        pattern = rf'({df_var}\s*=\s*df(?:\.copy\(\))?)'
        injection_str = "\n" + "\n".join(injections)
        if re.search(pattern, code):
            code = re.sub(pattern, r'\1' + injection_str, code, count=1)
        else:
            # Prepend to code if df/dfc assignment is not found
            code = "\n".join(injections) + "\n" + code
            
    return code



