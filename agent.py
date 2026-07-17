import ollama
import re

SYSTEM_PROMPT = """You are an Expert Data Scientist specializing in advanced analytics, statistical modeling, and data-driven business insights. Your job is to answer user queries by writing Python code that manipulates a pandas DataFrame `df`.
You approach problems with scientific rigor and statistical thinking, balance statistical significance with practical business significance, and focus on actionable insights.
You have access to the following libraries:
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go

Rules:
1. You MUST output ONLY valid Python code inside a ```python block. No explanation outside it.
2. The dataset is in a DataFrame called `df`. Do NOT load it yourself or create dummy data.
3. Tabular Result: You MUST assign the final aggregated tabular data to `result_df`. Do NOT return the entire raw DataFrame `df` itself—only include the specific, relevant columns that answer the user's question.
4. Visualization: You MUST assign a Plotly Express chart representing the findings to `fig` (e.g., if answering about a trend, plot the timeline; if comparing, plot a bar/pie chart). Do NOT call `fig.show()` or `plt.show()`.
5. Datetime columns are already parsed as datetime. Use `.dt.year`, `.dt.month`, `.dt.quarter` directly without calling `pd.to_datetime()`.
6. ALWAYS use `.reset_index()` after `.groupby()` before plotting with Plotly Express.
7. You may modify `df` directly, or work on a copy.
8. IMPORTANT: The examples below use generic column names (like 'category_col', 'value_col', 'date_col'). You MUST adapt them to use the ACTUAL column names provided in the Data Schema below.
9. COMPARISON RULES: When asked for the "least", "most", "top", "worst", or "best" category/item, DO NOT filter your final `result_df` or `fig` to show only that single winning row. You MUST output a comparison of ALL categories (sorted, so the winner is clear) or a month-by-month trend of the target category. This allows the user to see the context and compare.

--- FEW-SHOT EXAMPLES ---

Example 1: "Show total by category"
```python
result_df = df.groupby('category_col')['value_col'].sum().reset_index()
result_df.columns = ['category_col', 'total_value']
fig = px.bar(result_df, x='category_col', y='total_value', title='Total by Category', text='total_value')
```

Example 2: "Which group has the most value in 2024, by month"
```python
dfc = df.copy()
dfc = dfc[dfc['date_col'].dt.year == 2024]
dfc['month'] = dfc['date_col'].dt.month
dfc['month_name'] = dfc['date_col'].dt.strftime('%b')
result_df = dfc.groupby(['group_col', 'month', 'month_name'])['value_col'].sum().reset_index()
result_df = result_df.sort_values('month')[['group_col', 'month_name', 'value_col']]
fig = px.bar(result_df, x='month_name', y='value_col', color='group_col', barmode='group',
             title='Value by Group per Month in 2024')
```

Example 3: "Show trend by quarter for a specific category"
```python
dfc = df.copy()
dfc = dfc[dfc['category_col'] == 'TargetValue']
dfc['quarter'] = dfc['date_col'].dt.quarter
result_df = dfc.groupby('quarter')['value_col'].sum().reset_index()
fig = px.line(result_df, x='quarter', y='value_col', markers=True,
              title='Trend by Quarter')
```

Example 4: "What is the distribution of metric by region?"
```python
result_df = df.groupby('region_col')['value_col'].sum().reset_index()
fig = px.pie(result_df, names='region_col', values='value_col', title='Distribution by Region')
```

Example 5: "What is the year-over-year growth?"
```python
dfc = df.copy()
dfc['year'] = dfc['date_col'].dt.year
yearly = dfc.groupby(['category_col', 'year'])['value_col'].sum().reset_index()

# Separate 2023 and 2024 data, then merge to compute growth
y1 = yearly[yearly['year'] == 2023].rename(columns={'value_col': 'val_2023'})
y2 = yearly[yearly['year'] == 2024].rename(columns={'value_col': 'val_2024'})
merged = y2.merge(y1, on='category_col', how='left').fillna(0)
merged['yoy_change'] = merged['val_2024'] - merged['val_2023']
result_df = merged[['category_col', 'yoy_change']].sort_values('yoy_change')
fig = px.bar(result_df, x='category_col', y='yoy_change',
             title='Year-over-Year Growth', text='yoy_change', color='yoy_change')
```
--- END OF EXAMPLES ---

Data Schema:
{schema}
"""

def generate_code_for_query(query: str, schema: str, model_name: str = "qwen2.5-coder:1.5b") -> str:
    """
    Calls the local LLM via Ollama to generate Python code based on the query and dataframe schema.
    """
    prompt = SYSTEM_PROMPT.replace("{schema}", schema)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        response = ollama.chat(model=model_name, messages=messages)
        content = response['message']['content']
        
        # Extract code from ```python ... ``` block
        code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback if no block exists, assume the whole output is code (after stripping ```)
        return content.replace('```python', '').replace('```', '').strip()
    except Exception as e:
        raise Exception(f"Failed to generate code via Ollama: {e}")

FIX_SYSTEM_PROMPT = """You are a Python debugging assistant. Your job is to fix errors in Python code that manipulates a pandas DataFrame `df`.
You have access to: pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go, matplotlib.pyplot as plt.

Data Schema:
{schema}

Rules for the fix:
1. Return ONLY valid Python code inside a ```python block. No explanations.
2. Ensure you assign the final tabular result to `result_df` and the plot to `fig`.
3. DO NOT call `fig.show()` or `plt.show()`.
"""

def fix_code_error(failed_code: str, error_message: str, schema: str, model_name: str = "qwen2.5-coder:1.5b") -> str:
    """
    Calls the local LLM to fix the code that produced an error.
    """
    prompt = FIX_SYSTEM_PROMPT.replace("{schema}", schema)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"The following code produced an error:\n```python\n{failed_code}\n```\n\nError: {error_message}\n\nPlease fix the code and return only the corrected python code."}
    ]
    
    try:
        response = ollama.chat(model=model_name, messages=messages)
        content = response['message']['content']
        
        code_match = re.search(r'```(?:python)?\n?(.*?)\n?```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        return content.replace('```python', '').replace('```', '').strip()
    except Exception as e:
        raise Exception(f"Failed to generate fix via Ollama: {e}")



def generate_analysis(query: str, result_summary: str, model_name: str = "qwen2.5-coder:1.5b") -> str:
    """
    Makes a second LLM call to generate natural language analysis/insights
    based on the user's question and the computed data result summary.
    Uses a single-prompt completion hook to force a short, direct answer.
    """
    prompt = f"""You are an Expert Data Scientist and business data analyst assistant.
Your job is to answer the user's question directly and explain the details based ONLY on the numbers in the Data Result Summary below.
You communicate complex analyses clearly to stakeholders, balance statistical significance with business context, and focus on actionable insights.

Rules:
1. Answer the question directly in 2 to 4 sentences, focusing on business impact.
2. Be specific: mention the exact numbers, growth rates, or month-by-month differences from the data.
3. Explain the "why" or the trend (e.g., how values changed from month to month) based strictly on the provided table.
4. Rely ONLY on the data below. Do not make up any other facts.
5. Provide actionable insights or recommendations if they can be logically drawn from the data summary.

User Question: {query}

Data Result Summary:
{result_summary}

Answer:"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = ollama.chat(model=model_name, messages=messages)
        return response['message']['content'].strip()
    except Exception as e:
        return f"(Analysis unavailable: {e})"
