"""
techMiner.Thesaurus
===============================

"""
import re

def find_string(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Find pattern in string.

    Args:
        pattern (string) 
        x (string) 
        ignore_case (bool)
        full_match (bool)
        use_re (bool)

    Returns:
        string or []

    """

    if use_re is False:
        pattern = re.escape(pattern)

    if full_match is True:
        pattern = "^" + pattern + "$"

    if ignore_case is True:
        result = re.findall(pattern, x, re.I)
    else:
        result = re.findall(pattern, x)

    if len(result):
        return result[0]
        
    return None


def replace_string(pattern, x, repl=None, ignore_case=True, full_match=False, use_re=False):
    """Replace pattern in string.

    Args:
        pattern (string) 
        x (string) 
        repl (string, None)
        ignore_case (bool)
        full_match (bool)
        use_re (bool)

    Returns:
        string or []

    """

    if use_re is False:
        pattern = re.escape(pattern)

    if full_match is True:
        pattern = "^" + pattern + "$"

    if ignore_case is True:
        return re.sub(pattern, repl, x, re.I)
    return re.sub(pattern, repl, x)
