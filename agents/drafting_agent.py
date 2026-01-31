# Drafting Agent
# Uses LLM + RAG to draft responses

def draft_response(email_text, kb_results, llm):
    """
    Drafts a personalized response using LLM and RAG context.
    Returns: response text and confidence score.
    """
    prompt = f"""
    You are a customer support agent. Using the following knowledge base context:
    {kb_results}
    Draft a personalized, helpful, and professional response to this customer email:
    {email_text}
    Also, provide a confidence score (0-1) for how well the response addresses the email, as JSON: {{'response': ..., 'confidence': ...}}
    """
    import json as pyjson
    result = llm(prompt)
    try:
        parsed = pyjson.loads(result)
        return parsed.get('response', ''), float(parsed.get('confidence', 0.85))
    except Exception:
        return result, 0.85
