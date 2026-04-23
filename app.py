import streamlit as st
from src.loader import load_pdf

st.set_page_config(page_title="Pharma Assistant")

st.title("Pharma Knowledge Assistant")
st.caption("Ask questions from pharma documents using AI")

if st.button("Load Sample PDF"):
    text = load_pdf("data/sample.pdf")
    
    st.subheader("Extracted Text")
    st.text_area("PDF Content", text, height=400)