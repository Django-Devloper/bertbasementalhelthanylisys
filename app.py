import streamlit as st 
from transformers import pipeline

calssifier = pipeline('text-classification', model='djangodevloper/bert-base-sa-mental-uncased')
st.title("Sentiment Analisys")

text = st.text_area("Enter the statement")
analysis = st.button('Analyse',type='primary',use_container_width=True)

if analysis:
    if text:
        report =calssifier(text)
        for r in report :
            st.write(f'{r["label"]} : {r["score"]}')
    else:
        st.write('Kindly enter some text .')