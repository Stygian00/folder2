# Review Agent
# Flags responses for human review if confidence is low

def needs_review(confidence_score, threshold=0.8):
    """
    Returns True if the response should be flagged for human review.
    """
    return confidence_score < threshold
