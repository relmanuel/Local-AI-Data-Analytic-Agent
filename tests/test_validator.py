"""
TDD Test Suite for validator.py

Run with:  python -m pytest tests/test_validator.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from validator import (
    validate_and_fix,
    _remove_dummy_data,
    _fix_multi_column_groupby,
    _remove_show_calls,
)


# ─── _remove_dummy_data ───────────────────────────────────────────────────────

class TestRemoveDummyData:
    def test_removes_multiline_dataframe_creation(self):
        code = """
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})
result_df = df.groupby('a')['b'].sum().reset_index()
"""
        result = _remove_dummy_data(code)
        assert 'pd.DataFrame' not in result
        assert 'result_df' in result

    def test_removes_data_dict_then_dataframe(self):
        code = """
data = {
    'col1': [1, 2],
    'col2': [3, 4]
}
df = pd.DataFrame(data)
result_df = df['col1'].sum()
"""
        result = _remove_dummy_data(code)
        assert 'pd.DataFrame(data)' not in result
        assert 'result_df' in result

    def test_preserves_code_without_dummy_data(self):
        code = "result_df = df.groupby('category')['total_amount'].sum().reset_index()"
        result = _remove_dummy_data(code)
        assert result.strip() == code.strip()

    def test_removes_singleline_dataframe(self):
        code = "df = pd.DataFrame({'a': [1]})\nresult_df = df"
        result = _remove_dummy_data(code)
        assert 'pd.DataFrame' not in result


# ─── _fix_multi_column_groupby ────────────────────────────────────────────────

class TestFixMultiColumnGroupby:
    def test_fixes_tuple_subscript_two_columns(self):
        code = "grouped = dfc.groupby('category')['total_amount', 'profit_margin'].sum().reset_index()"
        result = _fix_multi_column_groupby(code)
        assert "[['total_amount', 'profit_margin']]" in result

    def test_does_not_double_wrap_already_correct(self):
        code = "grouped = dfc.groupby('category')[['total_amount', 'profit_margin']].sum().reset_index()"
        result = _fix_multi_column_groupby(code)
        # Should not produce triple brackets
        assert '[[[' not in result
        assert "[['total_amount', 'profit_margin']]" in result

    def test_single_column_not_changed(self):
        code = "grouped = df.groupby('category')['total_amount'].sum()"
        result = _fix_multi_column_groupby(code)
        assert "['total_amount']" in result
        assert "[[" not in result


# ─── _remove_show_calls ───────────────────────────────────────────────────────

class TestRemoveShowCalls:
    def test_removes_fig_show(self):
        code = "fig = px.bar(df, x='a', y='b')\nfig.show()"
        result = _remove_show_calls(code)
        assert 'fig.show()' not in result
        assert 'fig = px.bar' in result

    def test_removes_plt_show(self):
        code = "plt.plot([1,2,3])\nplt.show()"
        result = _remove_show_calls(code)
        assert 'plt.show()' not in result

    def test_preserves_rest_of_code(self):
        code = "fig = px.pie(df, names='x', values='y')\nfig.show()\nresult_df = df"
        result = _remove_show_calls(code)
        assert 'result_df = df' in result



# ─── _fix_diff_shift ──────────────────────────────────────────────────────────

from validator import _fix_diff_shift

class TestFixDiffShift:
    def test_removes_shift_after_diff(self):
        code = "yearly['yoy_change'] = yearly.groupby('category')['profit_margin'].diff().shift(-1)"
        result = _fix_diff_shift(code)
        assert '.shift(-1)' not in result
        assert '.diff()' in result

    def test_removes_positive_shift_after_diff(self):
        code = "s = df.groupby('a')['b'].diff().shift(1)"
        result = _fix_diff_shift(code)
        assert '.shift(1)' not in result
        assert '.diff()' in result

    def test_does_not_remove_standalone_shift(self):
        code = "s = df['col'].shift(-1)"
        result = _fix_diff_shift(code)
        assert '.shift(-1)' in result

    def test_does_not_remove_standalone_diff(self):
        code = "s = df['col'].diff()"
        result = _fix_diff_shift(code)
        assert '.diff()' in result


# ─── validate_and_fix (integration) ──────────────────────────────────────────

class TestValidateAndFix:
    def test_fixes_the_exact_failing_code(self):
        """This is the exact code pattern that caused the error in production."""
        bad_code = """
import pandas as pd

data = {
    'order_id': [1, 2, 3],
    'category': ['Home', 'Grocery', 'Electronics'],
    'total_amount': [139.47, 24.73, 166.8],
    'profit_margin': [31.17, -2.62, 13.44],
    'order_date': pd.to_datetime(['2023-12-23', '2025-04-03', '2024-10-08'])
}
df = pd.DataFrame(data)

dfc = df[df['order_date'].dt.year == 2024]
grouped = dfc.groupby('category')['total_amount', 'profit_margin'].sum().reset_index()
fig = px.bar(grouped, x='category', y='total_amount')
fig.show()
"""
        result = validate_and_fix(bad_code)
        assert 'pd.DataFrame(data)' not in result
        assert 'fig.show()' not in result
        assert "[['total_amount', 'profit_margin']]" in result

    def test_clean_code_passes_through_unchanged(self):
        clean_code = """
dfc = df.copy()
dfc = dfc[dfc['order_date'].dt.year == 2024]
dfc['month'] = dfc['order_date'].dt.month
result_df = dfc.groupby(['category', 'month'])[['total_amount']].sum().reset_index()
fig = px.bar(result_df, x='month', y='total_amount', color='category')
"""
        result = validate_and_fix(clean_code)
        assert 'result_df' in result
        assert 'fig' in result
        assert 'fig.show()' not in result


# ─── _fix_agg_multiindex ─────────────────────────────────────────────────────

from validator import _fix_agg_multiindex

class TestFixAggMultiindex:
    def test_fixes_single_list_agg(self):
        code = "grouped = dfc.groupby('category').agg({'price': ['sum']}).reset_index()"
        result = _fix_agg_multiindex(code)
        assert "{'price': 'sum'}" in result
        assert "['sum']" not in result

    def test_fixes_multiple_list_agg(self):
        code = "grouped = dfc.groupby('category').agg({'total_amount': ['sum'], 'profit_margin': ['mean']}).reset_index()"
        result = _fix_agg_multiindex(code)
        assert "'total_amount': 'sum'" in result
        assert "'profit_margin': 'mean'" in result

    def test_does_not_change_string_agg(self):
        code = "grouped = dfc.groupby('category').agg({'price': 'sum'}).reset_index()"
        result = _fix_agg_multiindex(code)
        assert "{'price': 'sum'}" in result

    def test_fixes_exact_failing_pattern(self):
        """Reproduces the exact 'Grouper and axis must be same length' error."""
        bad_code = """
dfc = df.copy()
dfc['year'] = dfc['order_date'].dt.year
grouped = dfc.groupby(['category', 'year']).agg({'price': ['sum']}).reset_index()
"""
        result = _fix_agg_multiindex(bad_code)
        assert "{'price': 'sum'}" in result
        assert "['sum']" not in result


# ─── _fix_placeholder_columns ────────────────────────────────────────────────

from validator import _fix_placeholder_columns

class TestFixPlaceholderColumns:
    def test_replaces_placeholders_with_actual_columns(self):
        import pandas as pd
        df = pd.DataFrame({
            'category': ['A', 'B'],
            'price': [10.0, 20.0],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })
        code = "result_df = dfc.groupby(['category_col', 'date_col'])[['value_col']].sum()"
        result = _fix_placeholder_columns(code, df)
        assert "'category'" in result
        assert "'order_date'" in result
        assert "'price'" in result
        assert "category_col" not in result
        assert "date_col" not in result
        assert "value_col" not in result


# ─── _fix_missing_time_cols ──────────────────────────────────────────────────

from validator import _fix_missing_time_cols

class TestFixMissingTimeCols:
    def test_injects_year_and_month_definitions(self):
        import pandas as pd
        df = pd.DataFrame({
            'category': ['A', 'B'],
            'price': [10.0, 20.0],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })
        code = """
dfc = df.copy()
dfc = dfc[dfc['order_date'].dt.year == 2024]
grouped = dfc.groupby(['category', 'year', 'month']).sum()
"""
        result = _fix_missing_time_cols(code, df)
        assert "dfc['year'] = dfc['order_date'].dt.year" in result
        assert "dfc['month'] = dfc['order_date'].dt.month" in result
        assert "dfc['quarter']" not in result

