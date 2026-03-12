# Agentic AI Email Automation POC

## Features
- Reads mock customer emails (support, sales, complaints)
- Prioritizes by urgency/sentiment (LLM)
- Drafts responses using LLM + RAG (ChromaDB)
- Schedules send time based on priority
- Flags low-confidence responses for human review
- Streamlit UI for previewing emails, responses, confidence, and scheduling

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize ChromaDB knowledge base:
   ```bash
   python rag/chromadb_setup.py
   ```
3. Process emails:
   ```bash
   python main.py
   ```
4. Launch UI:
   ```bash
   streamlit run ui/streamlit_app.py
   ```

## Notes
- LLM and RAG calls are mocked for POC. Replace with Ollama/Gemini/real ChromaDB queries for production.
- Add more emails to `data/emails/` and knowledge docs to `rag/knowledge_base/` as needed.
- Scheduling and review logic is in agents/classifier_agent.py and agents/review_agent.py.

## Orchestration
- Agents are orchestrated in `main.py` (see comments for flow).
- Each agent is modular and can be swapped for real API calls.

---
POC by Anusha
