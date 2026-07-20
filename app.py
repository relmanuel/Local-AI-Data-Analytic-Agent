import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from database import load_csv, get_dataframe_schema
from sandbox import run_secure_code
from agent import generate_code_for_query, fix_code_error, generate_analysis

# Configure Streamlit page
st.set_page_config(page_title="Offline AI Data Analyst", layout="wide")

st.title("Offline AI Data Analyst")

# Sidebar setup
st.sidebar.header("Configuration")

# Dynamically retrieve the active model from Ollama
try:
    import ollama
    models_response = ollama.list()
    available_models = [m.model for m in models_response.models]
    if "qwen2.5-coder:1.5b" in available_models:
        model_choice = "qwen2.5-coder:1.5b"
    elif available_models:
        model_choice = available_models[0]
    else:
        model_choice = "qwen2.5-coder:1.5b"
except Exception:
    model_choice = "qwen2.5-coder:1.5b"

st.sidebar.info(f"Active Ollama Model: **{model_choice}**")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = load_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        schema = get_dataframe_schema(df)
        
        st.subheader("Ask a Question")
        query = st.text_area("What would you like to know about this data?", 
                             placeholder="e.g. Show me a bar chart of sales by region")
        
        if st.button("Analyze"):
            if query.strip() == "":
                st.warning("Enter your query.")
            else:
                code = ""
                with st.spinner("Analyzing..."):
                    try:
                        # 1. Generate Code (silently)
                        code = generate_code_for_query(query, schema, model_name=model_choice)
                        
                        # 2. Execute Code safely
                        output = run_secure_code(code, df)
                            
                        # If error, silently auto-fix up to 10 times
                        max_retries = 10
                        retries = 0
                        while output['error'] and retries < max_retries:
                            retries += 1
                            code = fix_code_error(code, output['error'], schema, model_name=model_choice)
                            output = run_secure_code(code, df)

                        # 3. Generate written analysis (second LLM call)
                        analysis_text = None
                        if not output['error']:
                            result_summary = ""
                            if output.get('result') is not None:
                                result_summary = output['result'].to_string(index=False)
                            elif output.get('printed_output'):
                                result_summary = output['printed_output']
                            analysis_text = generate_analysis(query, result_summary, model_name=model_choice)
                            
                    except Exception as e:
                        output = {'error': str(e), 'result': None, 'fig': None, 'printed_output': None}
                        analysis_text = None
                
                # 3. Display Results (outside spinner so UI renders)
                if output['error']:
                    st.error(f"Could not complete analysis: {output['error']}")
                    st.info("Try rephrasing your question or check if the column names are correct.")
                    with st.expander("Debug: View Last Generated Code", expanded=False):
                        st.code(code, language="python")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if output.get('result') is not None:
                            st.write("### Result Data")
                            st.dataframe(output['result'], hide_index=True)
                        elif output.get('printed_output'):
                            st.write("### Text Output")
                            st.text(output['printed_output'])
                        else:
                            st.info("No tabular data returned.")
                            
                    with col2:
                        if output['fig'] is not None:
                            st.write("### Visualization")
                            fig = output['fig']
                            if type(fig).__name__ == 'Figure' and type(fig).__module__.startswith('plotly'):
                                # Add data labels to every trace
                                fig.update_traces(textposition='auto', selector=dict(type='bar'))
                                fig.update_traces(textposition='inside', textinfo='percent+label', selector=dict(type='pie'))
                                fig.update_traces(texttemplate='%{text}', selector=dict(type='scatter', mode='markers'))
                                fig.for_each_trace(
                                    lambda t: t.update(text=t.y, texttemplate='%{text:.2s}', textposition='outside')
                                    if hasattr(t, 'y') and t.y is not None and t.type == 'bar' else None
                                )
                                # Hide y-axis tick labels
                                fig.update_layout(
                                    yaxis=dict(showticklabels=False),
                                    uniformtext_minsize=8,
                                    uniformtext_mode='hide'
                                )
                                
                                # Chronological sorting for month names
                                month_words = {
                                    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                                    'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december'
                                }
                                month_order = [
                                    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                                    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                                    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
                                    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
                                    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"
                                ]
                                
                                x_vals = []
                                y_vals = []
                                for trace in fig.data:
                                    if hasattr(trace, 'x') and trace.x is not None:
                                        try:
                                            x_vals.extend([str(v).strip().lower() for v in trace.x])
                                        except TypeError:
                                            x_vals.append(str(trace.x).strip().lower())
                                    if hasattr(trace, 'y') and trace.y is not None:
                                        try:
                                            y_vals.extend([str(v).strip().lower() for v in trace.y])
                                        except TypeError:
                                            y_vals.append(str(trace.y).strip().lower())
                                
                                if any(v in month_words for v in x_vals):
                                    fig.update_xaxes(categoryorder='array', categoryarray=month_order)
                                if any(v in month_words for v in y_vals):
                                    fig.update_yaxes(categoryorder='array', categoryarray=month_order)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.pyplot(fig)
                        else:
                            st.info("No chart returned by the AI.")

                    # 4. Display AI Analysis below results
                    if analysis_text:
                        st.divider()
                        st.write("### AI Analysis & Recommendations")
                        st.markdown(analysis_text)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
