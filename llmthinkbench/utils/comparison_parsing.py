import re
import logging

def parse_comparison_result(response):
    """
    Extract comparison result from a boxed answer in the response.
    
    Args:
        response (str): The model's response text
        
    Returns:
        str or None: Normalized comparison result ('greater than', 'less than', 'equal to') 
                     if found, None otherwise
    """
    # Find boxed answer
    match = re.search(r'\\boxed{([^{}]+)}', response.replace('\n', ' '))
    if not match:
        return None
    
    answer = match.group(1).strip().lower()
    
    # Normalize variations of answers
    if any(term in answer for term in ['greater', '>', 'more', 'larger', 'bigger']):
        return 'greater than'
    elif any(term in answer for term in ['less', '<', 'smaller', 'lower']):
        return 'less than'
    elif any(term in answer for term in ['equal', '=', 'same']):
        return 'equal to'
    
    return None