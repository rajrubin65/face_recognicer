import streamlit as st
import subprocess

st.title("Streamlit with FastAPI")

# Define a function to run FastAPI app
def run_fastapi():
    subprocess.run(["python","executer.py"])
run_fastapi()
# Add a button to run FastAPI app
if st.button("Run FastAPI App"):
    st.write("Starting FastAPI app...")
    run_fastapi()
    st.write("FastAPI app is now running!")
