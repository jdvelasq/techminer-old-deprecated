"""
Functions for manipulating lists of keywords
====================================================

This module contains functions that can be applied to each element of a pandas.Series 
object using the map function. 


"""
import pandas as pd
import string
import re
import geopandas
import json

def extractKeywords(x, keywordsList, ignoreCase=True, fullMatch=False):
    r"""Returns a string with the keywords in x matching the list `keywords`.

    The funcion allows the extraction of complex patterns using regular expresions (regex). 
    Detail information about regex sintax in Python can be obtained at https://docs.python.org/3/library/re.html#re-syntax.

    Args:
        x (string): A string object.
        keywordsList (string, list): string or list of strings searched. Each element of `keywordList` is interpreted as a regular expression.
        ignoreCase (bool):  ignore case?
        fullMatch (bool): the whole string matches the regular expression in keywordList.
        
    Returns:
        String.

    **Recipes**

    The following code exemplify some common cases using regular expressions.

    * Partial match.

    >>> extractKeywords( 'one two three four five', 'hre')
    'hre'


    * Word whole only. r'\b' represents white space.

    >>> extractKeywords( 'one two three four five', r"\btwo\b")
    'two'

    >>> extractKeywords( 'one two three four five', r"\bTWO\b")
    'two'


    * Case sensitive.

    >>> extractKeywords( 'one two three four five', r"\bTWO\b", ignoreCase=False)
    ''

    >>> extractKeywords( 'one TWO three four five', r"\bTWO\b", ignoreCase=False)
    'TWO'


    * A word followed by other word.

    >>> extractKeywords( 'one two three four five', r"\btwo\Wthree\b")
    'two three'


    * Multiple white spaces.

    >>> extractKeywords( 'one two    three four five', r"two\W+three")
    'two    three'

    * A list of keywords.

    >>> extractKeywords( 'one two three four five', ["xxx", "two", "yyy"])
    'two'


    * Adjacent terms but the order is unimportant.

    >>> extractKeywords( 'one two three four five', r"\Wthree\Wtwo\b|\Wthree\Wtwo\b")
    ''

    * Near words.

    Two words (`'two'`, `'four'`) separated by any other.

    >>> extractKeywords( 'one two three four five', r"\btwo\W+\w+\W+four\b")
    'two three four'


    Two words (`'two'`, `'five'`) separated by one, two or three unspecified words.

    >>> extractKeywords( 'one two three four five', r"\btwo\W+(?:\w+\W+){1,3}?five")
    'two three four five'

    * Or operator.

    >>> extractKeywords( 'one two three four five', r"three|two")
    'two'

    * And operator. One word followed by other at any word distance.

    >>> extractKeywords( 'one two three four five', r"\btwo\W+(?:\w+\W+)+?five")
    'two three four five'

    """

    if isinstance(keywordsList, str):
        keywordsList = [keywordsList]

    for keyword in keywordsList:

        if ignoreCase is True:
            c = re.compile(keyword, re.I)
        else:
            c = re.compile(keyword)

        if fullMatch is True:
            result = c.fullMatch(x)
        else:
            result = c.findall(x)

        
        if len(result):
            return result[0]
        

    return ''



def findKeywords(x, keywordsList, ignoreCase=True, fullMatch=False):
    """Search expr in string x and returns a Boolean.

    Args:
        x (string): A string object.
        keywordsList (string, list): string or list of strings searched. Each element of `keywordList` is interpreted as a regular expression.
        ignoreCase (bool):  ignore case?
        fullMatch (bool): the whole string matches the regular expression in keywordList.
        
    Returns:
        Boolean.

    """
    if extractKeywords(x, keywordsList, ignoreCase, fullMatch) == '':
        return False
    else:
        return True

def removeKeywords(x, keywordsList, ignoreCase=True, fullMatch=False):
    """Generate a new list removing the strings that match a 
    list of keywords

    Args:
        x (string): A string object.
        keywordsList (string, list): string or list of strings searched. Each element of `keywordList` is interpreted as a regular expression.
        ignoreCase (bool):  ignore case?
        fullMatch (bool): the whole string matches the regular expression in keywordList.
        
    Returns:
        String.
             
    """

    y = extractKeywords(x, keywordsList, ignoreCase, fullMatch)
    if y != '':
        result = re.sub(y, ' ', x)
    else:
        result = x
    return result

    
def extractNearbyKeywords(x, keywordsList, ignoreCase=True, fullMatch=False, n_words=1):
    """Extracts the words in the proximity of the terms matching
    a keywords list. 

     Args:
        x (string): A string object.
        keywords (list): a list of strings.
        n_words (integer): number of words around term. 
             
    Returns:
        string.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['aaa 1 2 3 4 5', '1 aaa 2 3 4 5', '1 2 aaa 3 4 5', 
    ...          '1 2 3 aaa 4 5', '1 2 3 4 aaa 5', '1 2 3 4 5 aaa'],
    ... })
    >>> df
                   f
    0  aaa 1 2 3 4 5
    1  1 aaa 2 3 4 5
    2  1 2 aaa 3 4 5
    3  1 2 3 aaa 4 5
    4  1 2 3 4 aaa 5
    5  1 2 3 4 5 aaa

    >>> df.f.map(lambda x: extractNearbyKeywords(x, keywordsList='aaa', n_words=2))
    0               
    1               
    2    1 2 aaa 3 4
    3    2 3 aaa 4 5
    4               
    5               
    Name: f, dtype: object
    """

    y = extractKeywords(x, keywordsList, ignoreCase, fullMatch)
        
    if len(y):

        if ignoreCase is True:
            c = re.compile('\w+\W+'*n_words + y + '\W+\w+'*n_words, re.I)
        else:
            c = re.compile('\w+\W+'*n_words + y + '\W+\w+'*n_words)

        if fullMatch is True:
            result = c.fullMatch(x)
        else:
            result = c.findall(x)

        if len(result):
            return result[0]
        else:
            return ''

    return ''

