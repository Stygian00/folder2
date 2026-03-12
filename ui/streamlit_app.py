import streamlit as st
import json
import os

def load_results():
    base_dir = os.path.dirname(__file__)
    results_path = os.path.join(base_dir, 'results.json')
    if not os.path.exists(results_path):
        return []
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    st.title('Agentic AI Email Automation Demo')
    st.write('POC: Autonomous email classification, drafting, and scheduling')
    results = load_results()
    if not results:
        st.info('No results found. Please run main.py to process emails.')
        return
    for idx, item in enumerate(results):
        st.header(f"Email #{idx+1}")
        st.subheader('Incoming Email')
        st.code(item['email'])
        st.subheader('Classification')
        st.json(item['classification'])
        st.subheader('Drafted Response')
        st.text_area('Response', item['response'], height=100)
        st.subheader('Confidence Score')
        st.progress(item['confidence'])
        st.write(f"Confidence: {item['confidence']:.2f}")
        st.subheader('Scheduling Suggestion')
        st.write(item['classification']['suggested_send_time'])
        if item['needs_review']:
            st.warning('Flagged for human review (low confidence)')
        st.markdown('---')

if __name__ == '__main__':
    main()
