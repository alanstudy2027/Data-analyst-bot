# ------- Developed by Alan Joshua ---------
# ----------------
# It is a spreadsheet chatbot, where user can upload their spreadsheet and chat with it
# --------------
# model used : Qwen3 4B

import streamlit as st
import pandas as pd
import re
import io
import sys
from mlx_lm import load, generate

# -----------------------------
# Load Model Once
# -----------------------------
@st.cache_resource
def load_model():
    return load("mlx-community/Qwen3-4B-Instruct-2507-4bit")

model, tokenizer = load_model()

# -----------------------------
# CSV Upload
# -----------------------------
st.title("📊 Qwen CSV Analyst Bot")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Clean column names
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").drop_duplicates()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "": None})
    df = df.dropna()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

    st.success(f"✅ Cleaned data loaded: {len(df)} rows, {len(df.columns)} columns.")
    st.dataframe(df)

    # -----------------------------
    # User Query
    # -----------------------------
    user_query = st.text_input("Ask a question about the CSV:")

    if st.button("Ask") and user_query.strip():
        with st.spinner("🤖 Thinking... please wait"):
            prompt = f"""
            You are a skilled data analyst.
            You have a cleaned Pandas DataFrame called `df` with these columns: {df.columns.tolist()}.
            User Question: {user_query}

            Instructions:
            - If you need Python code, put it inside a ```python``` block.
            - Use only exact column names from above.
            - Avoid guessing column names.
            - Assume df and pd are already defined.
            """

            try:
                # Generate answer
                answer = generate(model, tokenizer, prompt=prompt, verbose=False).strip()

                st.markdown("**Generated Code / Answer:**")
                st.code(answer, language="python")

                # Extract Python code
                code_blocks = re.findall(r"```python(.*?)```", answer, re.DOTALL)
                if not code_blocks:
                    st.warning("⚠️ No Python code found in the model's response.")
                else:
                    code_to_run = code_blocks[0].strip()
                    code_to_run = code_to_run.replace("import pandas as pd", "")

                    # Controlled namespace
                    local_vars = {"pd": pd, "df": df}

                    # Capture stdout
                    stdout_backup = sys.stdout
                    sys.stdout = io.StringIO()

                    # Try executing
                    code_lines = code_to_run.strip().split("\n")
                    last_line = code_lines[-1].strip()

                    try:
                        exec("\n".join(code_lines[:-1]), local_vars)
                        last_value = eval(last_line, local_vars)
                    except:
                        exec(code_to_run, local_vars)
                        last_value = None

                    # Restore stdout
                    output_text = sys.stdout.getvalue().strip()
                    sys.stdout = stdout_backup

                    st.markdown("### 📄 Execution Output:")
                    if output_text:
                        st.text(output_text)
                    elif isinstance(last_value, pd.DataFrame):
                        st.dataframe(last_value)
                    elif last_value is not None:
                        st.write(last_value)
                    else:
                        st.info("No output produced.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
