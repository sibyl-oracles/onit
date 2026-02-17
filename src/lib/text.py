import re

def remove_tags(text: str) -> str:
    """
    Remove all tags in the format <tag> and </tag> from the text.
    
    Args:
        text: The text to process
    Returns:
        The text with all tags removed
    """
    
    if not text:
        return text
    # Regular expression to match tags like <tag> or </tag>
    tag_pattern = re.compile(r'</?[^>]+>')
    # Substitute tags with an empty string
    cleaned_text = tag_pattern.sub('', text)
    return cleaned_text

def text_between_tags(text: str, tag: str) -> tuple[bool, str]:
    """
    Extract text between <tag> and </tag> tags.
    
    Args:
        text: The text to search in
        tag: The tag name without angle brackets
        
    Returns:
        tuple: (is_full_match, extracted_text)
            is_full_match: True if text starts with <tag> and ends with </tag>
            extracted_text: The text between tags or original text if tags not found
    """
    if not text or not tag:
        return False, text
        
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    # Check if text is fully wrapped in the specified tags
    is_full_match = text.startswith(start_tag) and text.endswith(end_tag)
    
    # Find the last occurrence of the start and end tags
    start_index = text.rfind(start_tag)
    if start_index == -1:
        return False, text
        
    end_index = text.rfind(end_tag)
    if end_index == -1 or end_index <= start_index:
        return False, text
    
    # Extract the text between the tags
    extracted_text = text[start_index + len(start_tag):end_index].strip()
    return is_full_match, extracted_text