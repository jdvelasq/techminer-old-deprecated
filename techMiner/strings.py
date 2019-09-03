"""
techMiner.Thesaurus
===============================

This module contains functions for manipulating texts. 



Functions in this module
----------------------------------------------------------------------------------------

"""
import re
import geopandas
import string
from nltk.stem import PorterStemmer

#-------------------------------------------------------------------------------------------
def find_string(pattern, x, ignore_case=True, full_match=False, use_re=False):
    r"""Find pattern in string.

    Args:
        pattern (string) 
        x (string) 
        ignore_case (bool)
        full_match (bool)
        use_re (bool)

    Returns:
        string or None

    >>> find_string(r'\btwo\b', 'one two three four five', use_re=True)
    'two'

    >>> find_string(r'\bTWO\b', 'one two three four five', use_re=True)
    'two'

    >>> find_string(r'\btwo\b', 'one TWO three four five', ignore_case=False, use_re=True) is None
    True

    >>> find_string(r'\btwo\Wthree\b', 'one two three four five', ignore_case=False, use_re=True)
    'two three'

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
def extract_country(x, sep=';'):
    """

    >>> import pandas as pd
    >>> x = pd.DataFrame({
    ...     'Affiliations': [
    ...         'University, Cuba; University, Venezuela',
    ...         'Univesity, United States; Univesity, Singapore',
    ...         'University;',
    ...         'University; Univesity',
    ...         'University,',
    ...         'University',
    ...         None]    
    ... })
    >>> x['Affiliations'].map(lambda x: extract_country(x))
    0             Cuba;Venezuela
    1    United States;Singapore
    2                       None
    3                       None
    4                       None
    5                       None
    6                       None
    Name: Affiliations, dtype: object
    """

    if x is None:
        return None

    ##
    ## lista generica de nombres de paises
    ##
    country_names = sorted(
        geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres')).name.tolist()
    )
        
    ## paises faltantes
    country_names.append('Singapore')            
    country_names.append('Malta') 
    country_names.append('United States') 

    ##
    ## Reemplazo de nombres de regiones administrativas
    ## por nombres de paises
    ##
    x = re.sub('Bosnia and Herzegovina', 'Bosnia and Herz.', x) 
    x = re.sub('Czech Republic', 'Czechia', x)
    x = re.sub('Russian Federation', 'Russia', x) 
    x = re.sub('Hong Kong', 'China', x)
    x = re.sub('Macau', 'China', x)
    x = re.sub('Macao', 'China', x)
    
    countries = [affiliation.split(',')[-1].strip() for affiliation in x.split(sep)]

    countries =  ';'.join(
        [country if country in country_names else '' for country in countries]
    )
    
    if countries == '' or countries == ';':
        return None
    else:
        return countries

#-------------------------------------------------------------------------------------------
def _steamming(pattern, text):
    
    text = asciify(text)
    pattern = asciify(pattern)

    text = text.strip().lower()
    pattern = pattern.strip().lower()

    porter = PorterStemmer()

    pattern = [porter.stem(w) for w in pattern.split()]
    text = [porter.stem(w) for w in text.split()]

    return [m in text for m in pattern]

#-------------------------------------------------------------------------------------------
def steamming_all(pattern, text):
    """

    >>> steamming_all('computers cars', 'car computing') 
    True
    
    >>> steamming_all('computers cars', 'car houses') 
    False

    """
    return all(_steamming(pattern, text))

#-------------------------------------------------------------------------------------------
def steamming_any(pattern, text):
    """

    >>> steamming_any('computers cars', 'car computing') 
    True

    >>> steamming_any('computers cars', 'computing house') 
    True
    
    >>> steamming_all('computers cars', 'tree houses') 
    False

    """
    return any(_steamming(pattern, text))

    


#-------------------------------------------------------------------------------------------
def asciify(text):
    """Translate non-ascii charaters to ascii equivalent.

    **Example:**

    >>> asciify('áéíóúÁÉÍÓÚñÑ')
    'aeiouaeiounn'

    """
    def translate(c):
        if c in ['\u00C0', '\u00C1', '\u00C2', '\u00C3', '\u00C4', 
                 '\u00C5', '\u00E0', '\u00E1', '\u00E2', '\u00E3',
                 '\u00E4', '\u00E5', '\u0100', '\u0101', '\u0102', 
                 '\u0103', '\u0104', '\u0105']:
            return 'a'
        if c in ['\u00C7', '\u00E7', '\u0106', '\u0107', '\u0108',
                 '\u0109', '\u010A', '\u010B', '\u010C', '\u010D']:
            return 'c'
        if c in ['\u00D0', '\u00F0', '\u010E', '\u010F', '\u0110',
                 '\u0111']:
            return 'd'
        if c in ['\u00C8', '\u00C9', '\u00CA', '\u00CB', '\u00E8',
                 '\u00E9', '\u00EA', '\u00EB', '\u0112', '\u0113',
                 '\u0114', '\u0115', '\u0116', '\u0117', '\u0118',
                 '\u0119', '\u011A', '\u011B']:
            return 'e'
        if c in ['\u011C', '\u011D', '\u011E', '\u011F', '\u0120',
                 '\u0121', '\u0122 ', '\u0123']:
            return 'g'
        if c in ['\u0124', '\u0125', '\u0126', '\u0127']:
            return 'h'
        if c in ['\u00CC', '\u00CD', '\u00CE', '\u00CF', '\u00EC', 
                 '\u00ED', '\u00EE', '\u00EF', '\u0128', '\u0129',
                 '\u012A', '\u012B', '\u012C', '\u012D', '\u012E',
                 '\u012F', '\u0130', '\u0131']:
            return 'i'
        if c in ['\u0134', '\u0135']:
            return 'j'
        if c in ['\u0136', '\u0137', '\u0138']:
            return 'k'
        if c in ['\u0139', '\u013A', '\u013B', '\u013C', '\u013D',
                 '\u013E', '\u013F', '\u0140', '\u0141', '\u0142']:
            return 'l'
        if c in ['\u00D1', '\u00F1', '\u0143', '\u0144', '\u0145',
                 '\u0146', '\u0147', '\u0148', '\u0149', '\u014A',
                 '\u014B']:
            return 'n'
        if c in ['\u00D2', '\u00D3', '\u00D4', '\u00D5', '\u00D6',
                 '\u00D8', '\u00F2', '\u00F3', '\u00F4', '\u00F5',
                 '\u00F6', '\u00F8', '\u014C', '\u014D', '\u014E',
                 '\u014F', '\u0150', '\u0151']:
            return 'o'
        if c in ['\u0154', '\u0155', '\u0156', '\u0157', '\u0158',
                 '\u0159']:
            return 'r'
        if c in ['\u015A', '\u015B', '\u015C', '\u015D', '\u015E',
                 '\u015F', '\u0160', '\u0161', '\u017F']:
            return 's'
        if c in ['\u0162', '\u0163', '\u0164', '\u0165', '\u0166',
                 '\u0167']:
            return 't'
        if c in ['\u00D9', '\u00DA', '\u00DB', '\u00DC', '\u00F9',
                 '\u00FA', '\u00FB', '\u00FC', '\u0168', '\u0169',
                 '\u016A', '\u016B', '\u016C', '\u016D', '\u016E',
                 '\u016F', '\u0170', '\u0171', '\u0172', '\u0173']:
            return 'u'
        if c in ['\u0174', '\u0175']:
            return 'w'
        if c in ['\u00DD', '\u00FD', '\u00FF', '\u0176', '\u0177',
                 '\u0178']:
            return 'y'
        if c in ['\u0179', '\u017A', '\u017B', '\u017C', '\u017D',
                 '\u017E']:
            return 'z'
        return c

    return ''.join([translate(c) for c in text])

#-------------------------------------------------------------------------------------------
def fingerprint(x):
    """Computes 'fingerprint' representation of string x.

    Args:
        x (string): string to convert.

    Returns:
        string.

    **Examples**
    
    >>> fingerprint('a A b')
    'a b'
    >>> fingerprint('b a a')
    'a b'
    >>> fingerprint(None) is None
    True
    >>> fingerprint('b c')
    'b c'
    >>> fingerprint(' c b ')
    'b c'

    
    """
    porter = PorterStemmer()

    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub('['+string.punctuation+']', '', x)
    x = asciify(x)
    x = ' '.join(porter.stem(w) for w in x.split())
    x = ' '.join({w for w in x.split()})
    x = ' '.join(sorted(x.split(' ')))
    return x

#-------------------------------------------------------------------------------------------
