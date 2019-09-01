"""
techMiner.Thesaurus
===============================

"""
import re
def find_string(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Find pattern in string.
    """

    if use_re is False:
        pattern = re.escape(pattern)

    if ignore_case is True:
        c = re.compile(pattern, re.I)
    else:
        c = re.compile(pattern)

    if full_match is True:
        result = c.fullMatch(x)
    else:
        result = c.findall(x)

    return result