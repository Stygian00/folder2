# Review Agent
# Flags responses for human review if confidence is low
import logging


logg_rewiew = logging.getLogger(__name__)
def needs_review(confidence_score, threshold=0.8):
    """
    Returns True if the response should be flagged for human review.
    """
    logg_rewiew.debug(f"confidence score : {confidence_score}")
    return confidence_score < threshold
