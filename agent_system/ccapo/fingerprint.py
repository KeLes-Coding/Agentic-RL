import re

def fingerprint_alfworld(action: str) -> str:
    """
    Abstracts an ALFWorld action string into a logic fingerprint.
    
    Rules:
    1. Removes numeric IDs from object names (e.g., 'apple 1' -> 'apple').
    2. Removes numeric IDs from receptacle names (e.g., 'fridge 2' -> 'fridge').
    3. Normalizes whitespace.
    
    Examples:
    - "put apple 1 in fridge 2" -> "put apple in fridge"
    - "go to countertop 1" -> "go to countertop"
    - "open microsd 3" -> "open microsd" (if applicable, though usually generic)
    - "clean apple 1 with sinkbasin 1" -> "clean apple with sinkbasin"
    """
    if not action:
        return ""
        
    # Lowercase for consistency
    action = action.lower().strip()
    
    # Regex to remove numbers followed by optional space, or standalone numbers
    # Pattern: match a space followed by digits, likely at the end of words
    # e.g., "apple 1" -> "apple"
    # We look for words followed by digits, and remove the digits.
    
    # Strategy: Replace " <word> <number>" with " <word>"
    # Or more simply, remove all sequences of digits that appear as suffix to words
    
    # Common ALfWorld object format: "name number"
    # We want to remove the number part.
    
    # Regex explanation:
    # \s+\d+ : Matches space(s) followed by digit(s)
    # \b : Word boundary ensures we don't stripping inside words (though ALFWorld ids are usually separate)
    
    # Correct approach for ALFWorld: objects are like "apple 1", "safe 1"
    # We just want to remove the trailing numbers of object tokens.
    
    # Pass 1: Remove " 1", " 23" etc.
    abstracted = re.sub(r'\s+\d+', '', action)
    
    return abstracted.strip()
