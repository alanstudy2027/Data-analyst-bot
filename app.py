# ------- Developed by Alan Joshua ---------
# ---------------- 
# Optimized spreadsheet chatbot with better error handling,
# performance improvements, and cleaner code structure
# --------------
# Model used: Qwen3 4B

import streamlit as st
import pandas as pd
import re
import io
import sys
from mlx_lm import load, generate
from typing import Optional, Tuple, Dict, Any

# -----------------------------
# Constants
# -----------------------------
MODEL_NAME = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_ROWS_TO_DISPLAY = 1000  # For preview purposes

# -----------------------------
# Load Model with cache
# -----------------------------
@st.cache_resource
def load_model() -> Tuple[Any, Any]:  # Using Any since mlx_lm types are not exposed
    """Load and cache the MLX model and tokenizer."""
    return load(MODEL_NAME)

model, tokenizer = load_model()

# -----------------------------
# Data Cleaning Functions
# -----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the input DataFrame."""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove empty rows and duplicates
    df = df.dropna(how='all').drop_duplicates()
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': None, '': None})
    
    # Drop rows with any NA values after cleaning
    df = df.dropna()
    
    # Try to convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    
    return df

# -----------------------------
# Code Execution Functions
# -----------------------------
def execute_code(code: str, df: pd.DataFrame) -> Tuple[Optional[Any], str]:
    """
    Execute Python code in a controlled environment.
    Returns (result, output_text) tuple.
    """
    local_vars = {'pd': pd, 'df': df}
    output_capture = io.StringIO()
    
    # Redirect stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    try:
        # Split code into lines and handle the last line specially
        code_lines = [line for line in code.split('\n') if line.strip()]
        if not code_lines:
            return None, ""
            
        # Execute all but last line
        if len(code_lines) > 1:
            exec('\n'.join(code_lines[:-1]), local_vars)
        
        # Try to eval last line if it's an expression
        last_line = code_lines[-1].strip()
        try:
            result = eval(last_line, local_vars)
        except SyntaxError:
            exec(last_line, local_vars)
            result = None
    except Exception as e:
        result = None
        output_capture.write(f"Error during execution: {str(e)}")
    finally:
        # Restore stdout
        sys.stdout = original_stdout
    
    output_text = output_capture.getvalue().strip()
    return result, output_text

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("📊 Qwen CSV Analyst Bot")
    st.caption("Upload a CSV file and ask questions about your data")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your CSV file", 
        type=["csv"],
        help="Upload a CSV file to analyze"
    )
    
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis")
        return
    
    # Load and clean data
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_dataframe(df)
        
        st.success(f"✅ Cleaned data loaded: {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.head(min(len(df), MAX_ROWS_TO_DISPLAY)))
        
        # Display basic stats
        with st.expander("📊 Basic Statistics"):
            st.write(df.describe(include='all'))
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")
        return
    
    # User query section
    user_query = st.text_input(
        "Ask a question about the CSV:",
        placeholder="e.g., 'Show sales by region' or 'What are the top 5 products?'"
    )
    
    if not user_query.strip():
        return
    
    if st.button("Analyze", type="primary"):
        with st.spinner("🤖 Analyzing your data... Please wait"):
            try:
                # Construct prompt with data context
                prompt = f"""
                You are a skilled data analyst working with a Pandas DataFrame called 'df'.
                DataFrame columns: {df.columns.tolist()}
                First 3 rows as example:
                {df.head(3).to_string()}

                User Question: {user_query}

                Instructions:
                1. Provide Python code to answer the question in a ```python``` block
                2. Use only the exact column names shown above
                3. Include comments explaining the code
                4. The code should end with the result to display
                5. Assume 'df' and 'pd' are already available
                """
                
                # Generate response
                answer = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt, 
                    verbose=False,
                    max_tokens=1000
                ).strip()
                
                # Display the raw answer
                st.subheader("🔍 Model Response")
                st.markdown(answer)
                
                # Extract and execute Python code
                code_blocks = re.findall(r"```python(.*?)```", answer, re.DOTALL)
                if not code_blocks:
                    st.warning("⚠️ No executable Python code found in the response")
                    return
                
                code_to_run = code_blocks[0].strip()
                st.subheader("🛠️ Executing Code")
                st.code(code_to_run, language='python')
                
                # Execute the code
                result, output = execute_code(code_to_run, df)
                
                # Display results
                st.subheader("📊 Results")
                if output:
                    st.text(output)
                
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)
                elif not output:
                    st.info("Code executed but produced no output")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
