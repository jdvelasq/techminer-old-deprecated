"""
Elementary functions
=========================================

This module contains functions that can be applied to each element of a pandas.Series 
object using the map function. 


"""
import pandas as pd
import string
import re
import geopandas

def extractKeywords(x, keywords, method=None):
    """Generate a new list extracting the words in a field that match a 
    list of keywords

    Args:
        x (string): A string object.
        keywords (list): a list of strings.
        method (None, 'strict', 'insensitive', 'regex'): a list of strings. 
             
    Options for `method`:

    * 'strict': Match whole words, case sensitive.

    * 'insensitive': Match whole words, case insensitive.

    * 'regex': use keywords as regular expressions. 

    Returns:
        string.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['áaa; bbb-ccc', 'bbb;ddd;eee', 'aaa bbb aaa', 'aaa ccc', 'aaa', 'eee'],
    ... })
    >>> df
                  f
    0  áaa; bbb-ccc
    1   bbb;ddd;eee
    2   aaa bbb aaa
    3       aaa ccc
    4           aaa
    5           eee

    >>> df.f.map(lambda x: extractKeywords(x, ['aaa', 'bbb']))
    0        aaa bbb
    1            bbb
    2    aaa bbb aaa
    3            aaa
    4            aaa
    5           None
    Name: f, dtype: object


    >>> df = pd.DataFrame({
    ...    'f': ['AAA; bbb-ccc', 'bbb;ddd;eee', 'aaa bbb aaa', 'aaa ccc', 'aaa', 'eee'],
    ... })
    >>> df
                  f
    0  AAA; bbb-ccc
    1   bbb;ddd;eee
    2   aaa bbb aaa
    3       aaa ccc
    4           aaa
    5           eee

    >>> df.f.map(lambda x: extractKeywords(x, ['AAA'], method='strict'))
    0     AAA
    1    None
    2    None
    3    None
    4    None
    5    None
    Name: f, dtype: object

    >>> df = pd.DataFrame({
    ...    'f': ['aa', 'aaa', 'a', 'aaaaa', 'baaab', 'baab'],
    ... })
    >>> df
           f
    0     aa
    1    aaa
    2      a
    3  aaaaa
    4  baaab
    5   baab

    >>> df.f.map(lambda x: extractKeywords(x, ['a{3,4}', 'ba{3}'], method='regex'))
    0     None
    1      aaa
    2     None
    3    aaaaa
    4    baaab
    5     None
    Name: f, dtype: object

    """

    keywords = [asciify(k) for k in keywords]
    x = asciify(x)

    if method == 'insenstive' or method is None:
        keywords = [k.lower() for k in keywords]
        x = x.lower()

    x = re.sub('['+string.punctuation+']', ' ', x)

    result = []

    ## Search using regular expressions
    if method == 'regex':
        for w in x.split(' '):
            for k in keywords:
                y = re.search(k, w) 
                if y:
                    result += [w]
                    break
    else:
        for w in x.split(' '):
            if w in keywords:
                result += [w]

    if len(result):
        return ' '.join(result) 

    return None
    
            


def removeKeywords(x, keywords, method=None):
    """Generate a new list removing the words in a field that match a 
    list of keywords

    Args:
        x (string): A string object.
        keywords (list): a list of strings.
        method (None, 'strict', 'insensitive', 'regex'): a list of strings. 
             
    Options for `method`:

    * 'strict': Match whole words, case sensitive.

    * 'insensitive': Match whole words, case insensitive.

    * 'regex': use keywords as regular expressions. 

    Returns:
        string.

    **Examples**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'f': ['AAA; bbb-ccc', 'bbb;ddd;eee', 'aaa bbb aaa', 'aaa ccc', 'aaa', 'eee'],
    ... })
    >>> df
                  f
    0  AAA; bbb-ccc
    1   bbb;ddd;eee
    2   aaa bbb aaa
    3       aaa ccc
    4           aaa
    5           eee

    >>> df.f.map(lambda x: removeKeywords(x, ['aaa', 'bbb']))
    0        ccc
    1    ddd eee
    2       None
    3        ccc
    4       None
    5        eee
    Name: f, dtype: object


    >>> df = pd.DataFrame({
    ...    'f': ['AAA; bbb-ccc', 'bbb;ddd;eee', 'aaa bbb aaa', 'aaa ccc', 'aaa', 'eee'],
    ... })
    >>> df
                  f
    0  AAA; bbb-ccc
    1   bbb;ddd;eee
    2   aaa bbb aaa
    3       aaa ccc
    4           aaa
    5           eee

    >>> df.f.map(lambda x: removeKeywords(x, ['AAA'], method='strict'))
    0        bbb ccc
    1    bbb ddd eee
    2    aaa bbb aaa
    3        aaa ccc
    4            aaa
    5            eee
    Name: f, dtype: object

    >>> df = pd.DataFrame({
    ...    'f': ['aa', 'aaa', 'a', 'aaaaa', 'baaab', 'baab'],
    ... })
    >>> df
           f
    0     aa
    1    aaa
    2      a
    3  aaaaa
    4  baaab
    5   baab

    >>> df.f.map(lambda x: removeKeywords(x, ['a{3,4}', 'ba{3}'], method='regex'))
    0      aa
    1    None
    2       a
    3    None
    4    None
    5    baab
    Name: f, dtype: object

    """
    keywords = [asciify(k) for k in keywords]
    x = asciify(x)

    if method == 'insenstive' or method is None:
        keywords = [k.lower() for k in keywords]
        x = x.lower()

    x = re.sub('['+string.punctuation+']', ' ', x)

    result = []

    ## Search using regular expressions
    if method == 'regex':
        for w in x.split(' '):
            y = [re.search(k, w) for k in keywords]
            if all(v is None for v in y):
                result += [w]
                break
    else:
        for w in x.split(' '):
            if w not in keywords:
                result += [w]

    if len(result):
        return ' '.join(result) 

    return None
    
def extractNearbyPhrases(x, keywords, n=0):
    """Extracts the words in the proximity of the terms matching
    a keywords list. 

     Args:
        x (string): A string object.
        keywords (list): a list of strings.
        n (integer): number of words around term. 
             
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

    >>> df.f.map(lambda x: extractNearbyPhrases(x, ['aaa'], n=2))
    0        aaa 1 2
    1      1 aaa 2 3
    2    1 2 aaa 3 4
    3    2 3 aaa 4 5
    4      3 4 aaa 5
    5        4 5 aaa
    Name: f, dtype: object
    """
    keywords = [asciify(k) for k in keywords]
    X = asciify(x)

    keywords = [k.lower() for k in keywords]
    X = X.lower()

    X = re.sub('['+string.punctuation+']', ' ', X) 

    result = []

    X_splited = X.split(' ')
    for idx, w in enumerate(X_splited):
        if w in keywords:
            result += x.split(' ')[max(idx-n,0):min(idx+n+1, len(X_splited))]

    if len(result):
        return ' '.join(result) 

    return None


def extractCountries(x, sep=';'):
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
    >>> x['Affiliations'].map(lambda x: extractCountries(x))
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
        
    ## paises faltantes / nombres incompletos
    country_names.append('United States')           # United States of America
    country_names.append('Singapore')               #
    country_names.append('Russian Federation')      # Russia
    country_names.append('Czech Republic')          # Czechia
    country_names.append('Bosnia and Herzegovina')  # Bosnia and Herz.
    country_names.append('Malta')                   #

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


def fingerprint(x):
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ... 'f': ['a A b', 'b a a', None, 'b c', 'c b']
    ... })

    >>> df['f'].map(lambda x: fingerprint(x))
    0     a b
    1     a b
    2    None
    3     b c
    4     b c
    Name: f, dtype: object
    
    """
    from nltk.stem import PorterStemmer
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
