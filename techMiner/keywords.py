"""
techMiner.Keywords
====================================================

This module contains functions that can be applied to each element of a pandas.Series 
object using the map function. 


Functions in this module
----------------------------

"""
import pandas as pd
import string
import re
import geopandas
import json

from techMiner.strings import find_string

class Keywords:
    """Creates a Keywords object used to find, extract or remove terms of interest from a string.

    Args:
        ignore_case (bool) :  Ignore string case.
        full_match (bool): match whole word?.
        use_re (bool): keywords as interpreted as regular expressions.

    Returns:
        Keywords : Keywords object

    

    """    
    def __init__(self, ignore_case=True, full_match=False, use_re=False):
        
        self._keywords = None
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re

    #----------------------------------------------------------------------
    def __repr__(self):
        """String representation of the object.

        >>> Keywords().add_keywords(['Big data', 'neural netoworks'])  # doctest: +NORMALIZE_WHITESPACE
        [
          "Big data",
          "neural netoworks"
        ]
        ignore_case=True, full_match=False, use_re=False  

        """
        text = json.dumps(self._keywords, indent=2, sort_keys=True)
        text += '\nignore_case={}, full_match={}, use_re={}'.format(
            self._ignore_case.__repr__(),
            self._full_match.__repr__(),
            self._use_re.__repr__()
        ) 
        return text

    #----------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()

    #----------------------------------------------------------------------
    def add_keywords(self, keywords, sep=None):
        """Adds new keywords to list of current keywords.

        Args:
            keywords (string, list of strings): new keywords to be added.
        
        Returns:
            Nothing

        >>> kyw = Keywords().add_keywords(['ann', 'deep learning'])
        >>> kyw
        [
          "ann",
          "deep learning"
        ]
        ignore_case=True, full_match=False, use_re=False

        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if isinstance(keywords, Keywords):
            keywords = keywords._keywords

        if isinstance(keywords, pd.Series):
            keywords = keywords.tolist()

        if sep is not None:
            keywords = [y for x in keywords if x is not None for y in x.split(sep)]    
        else:
            keywords = [x for x in keywords if x is not None]

        if self._keywords is None:
            self._keywords = sorted(list(set(keywords)))
        else:
            self._keywords.extend(keywords)
            self._keywords = sorted(set(self._keywords))

        return self

    #----------------------------------------------------------------------
    def common(self, x, sep=None):
        """Returns True if x is in keywords list.

        Args:
            x (string): A string object.
            
        Returns:
            Boolean.

        >>> kw = Keywords().add_keywords(['ann', 'big data', 'deep learning'])
        >>> kw.common('Big Data')
        True
        >>> kw.common('Python')
        False
        >>> kw.common('Python|R', sep='|')
        False
        >>> kw.common('Python|big data', sep='|')
        True

        """
        def _common(x):
            if self.extract(x) is None:
                return False
            else:
                return True

        if sep is None:
            return _common(x)

        return any([_common(y) for y in x.split(sep)])

    #----------------------------------------------------------------------
    def complement(self, x, sep=None):
        """Returns False if x is not in keywords list.

        Args:
            x (string): A string object.
            
        Returns:
            Boolean.

        >>> kyw = Keywords().add_keywords(['ann', 'big data', 'deep learning'])
        >>> kyw.complement('Big Data')
        False
        >>> kyw.complement('Python')
        True
        >>> kyw.complement('Python|R')
        True
        >>> kyw.complement('Python|big data')
        False

        """
        def _complement(x):
            if self.extract(x) is None:
                return True
            else:
                return False

        if sep is None:
            return _complement(x)

        return any([_complement(y) for y in x.split(sep)])        

    #----------------------------------------------------------------------
    def extract(self, x, sep='|'):
        r"""Returns a string with the keywords in string x matching the list of keywords used to fit the model.

        >>> Keywords().add_keywords([r"xxx", r"two", r"yyy"]).extract('one two three four five')
        'two'

        The funcion allows the extraction of complex patterns using regular expresions (regex). 
        Detail information about regex sintax in Python can be obtained at https://docs.python.org/3/library/re.html#re-syntax.

        Args:
            x (string): A string object.
            
        Returns:
            String.

        **Recipes**

        The following code exemplify some common cases using regular expressions.

        >>> Keywords().add_keywords('111').extract('one two three four five') is None
        True

        * Partial match.

        >>> Keywords().add_keywords('hre').extract('one two three four five')
        'hre'


        * **Word whole only**. `r'\b'` represents white space.

        >>> Keywords(use_re=True).add_keywords(r"\btwo\b").extract('one two three four five')
        'two'

        >>> Keywords(use_re=True).add_keywords(r"\bTWO\b").extract('one two three four five')
        'two'


        * **Case sensitive**.

        >>> Keywords(ignore_case=False, use_re=True).add_keywords(r"\btwo\b").extract('one two three four five')
        'two'

        >>> Keywords(ignore_case=False, use_re=True).add_keywords(r"\bTWO\b").extract('one TWO three four five')
        'TWO'


        * **A word followed by other word**.

        >>> Keywords(ignore_case=False, use_re=True).add_keywords(r"\btwo\Wthree\b").extract('one two three four five')
        'two three'


        * **Multiple white spaces**.

        >>> Keywords(ignore_case=False, use_re=True).add_keywords(r"two\W+three").extract('one two    three four five')
        'two    three'

        * **A list of keywords**.

        >>> Keywords().add_keywords([r"xxx", r"two", r"yyy"]).extract('one two three four five')
        'two'


        * **Adjacent terms but the order is unimportant**.

        >>> Keywords(use_re=True).add_keywords(r"\bthree\W+two\b|\btwo\W+three\b").extract('one two three four five')
        'two three'

        * **Near words**.

        Two words (`'two'`, `'four'`) separated by any other.

        >>> Keywords(use_re=True).add_keywords(r"\btwo\W+\w+\W+four\b").extract('one two three four five')
        'two three four'


        Two words (`'two'`, `'five'`) separated by one, two or three unspecified words.

        >>> Keywords(use_re=True).add_keywords(r"\btwo\W+(?:\w+\W+){1,3}?five").extract('one two three four five')
        'two three four five'

        * **Or operator**.

        >>> Keywords(use_re=True).add_keywords(r"123|two").extract('one two three four five')
        'two'

        * **And operator**. One word followed by other at any word distance.

        >>> Keywords(use_re=True).add_keywords(r"\btwo\W+(?:\w+\W+)+?five") \
        ...    .extract('one two three four five')
        'two three four five'

        """

        if x is None:
            return None

        result = []
        for keyword in self._keywords:

            y = find_string(
                pattern = keyword,
                x = x,
                ignore_case = self._ignore_case,
                full_match = self._full_match,
                use_re = self._use_re
            )

            if y is not None:                
                result.extend([y])
        
        if len(result):
            return sep.join(set(result))
            
        return None

    #----------------------------------------------------------------------
    def extractAfterFirst(self, x):
        """Returns the string from the first ocurrence of the keyword to the end of string x.

        Args:
            x: string

        Returns:
            String

        >>> Keywords().add_keywords('aaa').extractAfterFirst('1 aaa 4 aaa 5')
        'aaa 4 aaa 5'

        """        
        y = self.extract(x)
            
        if len(y):

            if self._ignore_case is True:
                c = re.compile(y, re.I)
            else:
                c = re.compile( y)

            z = c.search(x)

            if z:
                return x[z.start():]
            else:
                return None

        return None

    #----------------------------------------------------------------------
    def extractAfterLast(self, x):
        """Returns the string from last ocurrence of a keyword to the end of string x.

        Args:
            x: string

        Returns:
            String

        >>> Keywords().add_keywords('aaa').extractAfterLast('1 aaa 4 aaa 5')
        'aaa 5'

        """ 
        
        y = self.extract(x)
            
        if len(y):

            if self._ignore_case is True:
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

    #----------------------------------------------------------------------
    def extractNearby(self, x, n_words=1):
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
        >>> df.f.map(lambda x: Keywords().add_keywords('aaa').extractNearby(x, n_words=2)) # doctest: +NORMALIZE_WHITESPACE
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
            
            if self._ignore_case is True:
                c = re.compile(pattern, re.I)
            else:
                c = re.compile(pattern)

            if self._full_match is True:
                result = c.fullMatch(x)
            else:
                result = c.findall(x)

            if len(result):
                return result[0]
            else:
                return None


        y = self.extract(x)
            
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

    #----------------------------------------------------------------------
    def extractUntilFirst(self, x):
        """Returns the string from begining of x to the first ocurrence of a keyword.

        Args:
            x: string

        Returns:
            String

        >>> Keywords().add_keywords('aaa').extractUntilFirst('1 aaa 4 aaa 5')
        '1 aaa'

        """
        y = self.extract(x)
            
        if len(y):

            if self._ignore_case is True:
                c = re.compile(y, re.I)
            else:
                c = re.compile( y)

            z = c.search(x)

            if z:
                return x[:z.end()]
            else:
                return None

        return None        

    #----------------------------------------------------------------------
    def extractUntilLast(self, x):
        """Returns the string from begining of x to the last ocurrence of a keyword.

        Args:
            x: string

        Returns:
            String

        >>> Keywords().add_keywords('aaa').extractUntilLast('1 aaa 4 aaa 5')
        '1 aaa 4 aaa'

        """
        y = self.extract(x)
            
        if len(y):

            if self._ignore_case is True:
                c = re.compile('[\w+\W+]+' + y, re.I)
            else:
                c = re.compile('[\w+\W+]+' +  y)

            z = c.search(x)

            if z:
                return x[:z.end()]
            else:
                return None

        return None         

    #----------------------------------------------------------------------
    def remove(self, x, sep=None):
        """Returns a string removing the strings that match a 
        list of keywords from x.

        Args:
            x (string): A string object.
            
        Returns:
            String.


        >>> Keywords().add_keywords('aaa').remove('1 aaa 2') is None
        True

        >>> Keywords().add_keywords('aaa').remove('1 2')
        '1 2'

        >>> Keywords().add_keywords('aaa').remove('1 aaa 2|1 2', sep='|')
        '1 2'

        >>> Keywords().add_keywords('aaa').remove('1 aaa 2|1 aaa 2', sep='|') is None
        True

        """

        def _remove(z):
            y = self.extract(z)
            if y is not None:
                return None
            else:
                result = z
            return result    

        if sep is None:
            return _remove(x)
        
        result = [_remove(z) for z in x.split(sep) ]
        result = [z for z in result if z is not None]
        if len(result):
            return sep.join(result)
        return None

        





    


