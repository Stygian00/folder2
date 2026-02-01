

import os
import json
import requests

from agents.classifier_agent import classify_email
from agents.drafting_agent import draft_response
from agents.review_agent import needs_review

from rag.chromadb_setup import retrieve_context

from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(model='tinyllama')

# Ollama LLM function
def ollama_llm(prompt):
    """Call Ollama using LangChain"""
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return f"Error generating response: {str(e)}"

def process_email(email_path):
    with open(email_path, 'r') as f:
        email_text = f.read()
    # 1. Classify
    classification = classify_email(email_text, ollama_llm)
    # 2. Retrieve context (RAG)
    kb_context = retrieve_context(email_text, n_results=2)
    # 3. Draft response
    response, confidence = draft_response(email_text, kb_context, ollama_llm)
    # 4. Review
    review_flag = needs_review(confidence)
    return {
        'email': email_text,
        'classification': classification,
        'response': response,
        'confidence': confidence,
        'needs_review': review_flag
    }

def main():
    email_dir = './data/emails'
    results = []
    for fname in os.listdir(email_dir):
        if fname.endswith('.txt'):
            result = process_email(os.path.join(email_dir, fname))
            results.append(result)
    with open('./ui/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Processing complete. Results saved to ./ui/results.json')

if __name__ == '__main__':
    main()
