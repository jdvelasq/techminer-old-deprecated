"""
techMiner.Thesaurus
===============================

This module contains functions for manipulating texts. 



Functions in this module
----------------------------------------------------------------------------------------

"""
import re

#-------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------------------
def extract_after_first(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Returns the string from the first ocurrence of the keyword to the end of string x.

    Args:
        pattern (string) : 
        x : string

    Returns:
        String

    >>> extract_after_first('aaa', '1 aaa 4 aaa 5')
    'aaa 4 aaa 5'

    >>> extract_after_first('bbb', '1 aaa 4 aaa 5')

    """        
    y = find_string(pattern, x, ignore_case, full_match, use_re)
        
    if y is not None:

        if ignore_case is True:
            c = re.compile(y, re.I)
        else:
            c = re.compile( y)

        z = c.search(x)

        if z:
            return x[z.start():]
        else:
            return None

    return None

#-------------------------------------------------------------------------------------------
def extract_after_last(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Returns the string from last ocurrence of a keyword to the end of string x.

    Args:
        x: string

    Returns:
        String

    >>> extract_after_last('aaa', '1 aaa 4 aaa 5')
    'aaa 5'

    """ 
    
    y = find_string(pattern, x, ignore_case, full_match, use_re)
        
    if y is not None:

        if ignore_case is True:
            c = re.compile(y, re.I)
        else:
            c = re.compile(y)

        z = c.findall(x)
        
        result = x
        for w in z[:-1]:
            y = c.search(result)
            result = result[y.end():]
        y = c.search(result)
        return result[y.start():]

    return None

#-------------------------------------------------------------------------------------------
def extract_nearby(pattern, x, n_words=1, ignore_case=True, full_match=False, use_re=False):
    """Extracts the words of string x in the proximity of the terms matching
    the keywords list. 

    Args:
        x (string): A string object.
        n_words (integer): number of words around term. 
            
    Returns:
        String.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['1 2 3 4 5 6', 'aaa 1 2 3 4 5', '1 aaa 2 3 4 5', '1 2 aaa 3 4 5', 
    ...          '1 2 3 aaa 4 5', '1 2 3 4 aaa 5', '1 2 3 4 5 aaa'],
    ... })
    >>> df
                   f
    0    1 2 3 4 5 6
    1  aaa 1 2 3 4 5
    2  1 aaa 2 3 4 5
    3  1 2 aaa 3 4 5
    4  1 2 3 aaa 4 5
    5  1 2 3 4 aaa 5
    6  1 2 3 4 5 aaa
    >>> df.f.map(lambda x: extract_nearby('aaa', x, n_words=2)) # doctest: +NORMALIZE_WHITESPACE
    0           None
    1        aaa 1 2
    2      1 aaa 2 3
    3    1 2 aaa 3 4
    4    2 3 aaa 4 5
    5      3 4 aaa 5
    6        4 5 aaa             
    Name: f, dtype: object
    """
    def check(pattern):
        
        if ignore_case is True:
            c = re.compile(pattern, re.I)
        else:
            c = re.compile(pattern)

        if full_match is True:
            result = c.fullMatch(x)
        else:
            result = c.findall(x)

        if len(result):
            return result[0]
        else:
            return None


    y = find_string(pattern, x, ignore_case, full_match, use_re)
        
    if y is not None:
        
        pattern = '\w\W' * n_words + y + '\W\w' * n_words
        result = check(pattern)
        
        if result is not None:
            return result
        else:
            for i in range(n_words, -1, -1):

                ## Checks at the beginning
                pattern = '^' + '\w\W' * i + y + '\W\w'*n_words
                result = check(pattern)
                if result is not None:
                    return result

            for j in range(n_words, -1, -1):
                ## Checks at the end
                pattern = '\w\W' * n_words + y + '\W\w' * j +'$'
                result = check(pattern)
                if result is not None:
                    return result

            for i in range(n_words, -1, -1):
                for j in range(n_words, -1, -1):
                    pattern = '^' + '\w\W' * i + y + '\W\w' * j +'$'
                    result = check(pattern)
                    if result is not None:
                        return result[0]

    return None

#-------------------------------------------------------------------------------------------
def extract_until_first(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Returns the string from begining of x to the first ocurrence of a keyword.

    Args:
        x: string

    Returns:
        String

    >>> extract_until_first('aaa', '1 aaa 4 aaa 5')
    '1 aaa'

    """
    y = find_string(pattern, x, ignore_case, full_match, use_re)
        
    if y is not None:

        if ignore_case is True:
            c = re.compile(y, re.I)
        else:
            c = re.compile( y)

        z = c.search(x)

        if z:
            return x[:z.end()]
        else:
            return None

    return None        

#-------------------------------------------------------------------------------------------
def extract_until_last(pattern, x, ignore_case=True, full_match=False, use_re=False):
    """Returns the string from begining of x to the last ocurrence of a keyword.

    Args:
        x: string

    Returns:
        String

    >>> extract_until_last('aaa', '1 aaa 4 aaa 5')
    '1 aaa 4 aaa'

    """
    y = find_string(pattern, x, ignore_case, full_match, use_re)
        
    if y is not None:

        if ignore_case is True:
            c = re.compile('[\w+\W+]+' + y, re.I)
        else:
            c = re.compile('[\w+\W+]+' +  y)

        z = c.search(x)

        if z:
            return x[:z.end()]
        else:
            return None

    return None 

#-------------------------------------------------------------------------------------------        
