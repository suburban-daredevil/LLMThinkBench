import re
import json
import logging

def parse_boxed_answer(response):
    """Robust extraction and parsing of boxed answer"""
    # Find all potential boxed answers with flexible parsing
    matches = re.findall(
        r'\\boxed{([\d\-\s,\[\]\(\)]+)}', 
        response.replace('\n', ' ').replace('\\n', ' ')
    )
    
    if not matches:
        return None
    
    # Try all matches from last to first (most likely at end)
    for match in reversed(matches):
        try:
            # Clean and parse the content
            cleaned = match.strip()
            
            # Handle different formats:
            if cleaned.startswith(('[', '(')) and cleaned.endswith((']', ')')):
                parsed = json.loads(cleaned)
            else:
                # Try comma/space separated values
                parsed = [int(x) for x in re.split(r'[,\s]+', cleaned) if x.strip()]
            
            return parsed
        except Exception as e:
            logging.debug(f"Parse error in '{cleaned}': {str(e)}")
            continue
    
    return None