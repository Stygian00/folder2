# Email Classifier Agent
# Uses LLM to classify emails by urgency/sentiment and suggest schedule
# path: C:\Users\anusha\Downloads\assignmentfolder\folder2\agents\classifier_agent.py

def classify_email(email_text, llm):
    """
    Classifies email urgency and sentiment using LLM.
    Returns: dict with 'priority', 'sentiment', 'suggested_send_time'
    """
    prompt = f"""
Analyze the following customer email and return ONLY a single JSON object (not an array/list) with exactly these keys:
- "priority": must be "urgent", "normal", or "low"
- "sentiment": must be "positive", "neutral", or "negative"
- "suggested_send_time": "4 hours from now" for urgent, "next business day" for low, "same day" for normal

Email:
{email_text}

Respond with ONLY the JSON object, no additional text:
"""
    import json as pyjson
    import re
    
    result = llm(prompt)
    
    # Default fallback
    default = {'priority': 'normal', 'sentiment': 'neutral', 'suggested_send_time': 'same day'}
    
    try:
        # Try to extract JSON from the response (handles cases with extra text)
        json_match = re.search(r'\{[^{}]*\}', result)
        if json_match:
            result = json_match.group()
        
        parsed = pyjson.loads(result)
        
        # Handle if LLM returns a list instead of dict
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and 'priority' in item:
                    return {
                        'priority': item.get('priority', 'normal'),
                        'sentiment': item.get('sentiment', 'neutral'),
                        'suggested_send_time': item.get('suggested_send_time', 'same day')
                    }
            return default
        
        # Handle dict response - ensure all keys exist
        if isinstance(parsed, dict):
            return {
                'priority': parsed.get('priority', 'normal'),
                'sentiment': parsed.get('sentiment', 'neutral'),
                'suggested_send_time': parsed.get('suggested_send_time', 'same day')
            }
        
        return default
        
    except Exception as e:
        print(f"Classification parsing error: {e}")
        print(f"Raw LLM response: {result}")
        return default