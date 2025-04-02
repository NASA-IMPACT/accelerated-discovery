import re
def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Convert any string into a valid filename by removing or replacing invalid characters.
    
    Args:
        filename (str): The original string to be sanitized.
        replacement (str): The character to replace invalid characters with (default: "_").

    Returns:
        str: A valid, sanitized filename.
    """
    # Remove leading and trailing whitespace
    filename = filename.strip()

    # Replace any invalid characters with the replacement string
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, filename)

    # Optionally, replace consecutive replacements with a single one
    sanitized = re.sub(f'{replacement}+', replacement, sanitized)

    # Limit filename length to a safe number (255 characters for most file systems)
    return sanitized[:255]


