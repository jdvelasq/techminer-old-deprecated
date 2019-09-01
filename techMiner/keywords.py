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

from techMiner.findstring import find_string

class MyKeywords:
    def __init__(self, ignore_case=True, full_match=False, use_re=False):
        """Creates a MyKeywords object used to find, extract or remove terms of interest from a field.

        Args:
            ignore_case (bool): Ignore string case.
            full_match (bool): match whole word?.

        Returns:
            `MyKeywords` object.
        """
        self._keywords = None
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re


    # def fit(self, keywords):
    #     """Fits model to keywords.

    #     Args:
    #         keywords (string, list of strings): keyword string or list of string with keywords.

    #     Returns:
    #         Nothing.
    #     """
    #     if isinstance(keywords, str):
    #         keywords = [keywords]
    #     self._keywords = list(set(keywords))
    #     return self

    # #def add_from_field(self, pattern, x):
    # #    ## pag 89
    # #    pass


    def add_keywords(self, keywords, sep=None):
        """Adds keywords to list of current keywords.

        Args:
            keywords (string, list of strings): keywords to be added.
        
        Returns:
            Nothing

        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if isinstance(keywords, MyKeywords):
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
            self._keywords = sorted(list(set(self._keywords.extend(keywords))))

        return self

    def extract(self, x, sep='|'):
        r"""Returns a string with the keywords in string x matching the list of keywords used to fit the model.

        >>> MyKeywords().add_keywords([r"xxx", r"two", r"yyy"]).extract('one two three four five')
        'two'

        The funcion allows the extraction of complex patterns using regular expresions (regex). 
        Detail information about regex sintax in Python can be obtained at https://docs.python.org/3/library/re.html#re-syntax.

        Args:
            x (string): A string object.
            
        Returns:
            String.

        **Recipes**

        The following code exemplify some common cases using regular expressions.

        * Partial match.

        >>> MyKeywords().add_keywords('hre').extract('one two three four five')
        'hre'


        * Word whole only. r'\b' represents white space.

        >>> MyKeywords(use_re=True).add_keywords(r"\btwo\b").extract('one two three four five')
        'two'

        >>> MyKeywords(use_re=True).add_keywords(r"\bTWO\b").extract('one two three four five')
        'two'


        * Case sensitive.

        >>> MyKeywords(ignore_case=False, use_re=True).add_keywords(r"\btwo\b").extract('one two three four five')
        'two'

        >>> MyKeywords(ignore_case=False, use_re=True).add_keywords(r"\bTWO\b").extract('one TWO three four five')
        'TWO'


        * A word followed by other word.

        >>> MyKeywords(ignore_case=False, use_re=True).add_keywords(r"\btwo\Wthree\b").extract('one two three four five')
        'two three'


        * Multiple white spaces.

        >>> MyKeywords(ignore_case=False, use_re=True).add_keywords(r"two\W+three").extract('one two    three four five')
        'two    three'

        * A list of keywords.

        >>> MyKeywords().add_keywords([r"xxx", r"two", r"yyy"]).extract('one two three four five')
        'two'


        * Adjacent terms but the order is unimportant.

        >>> MyKeywords(use_re=True).add_keywords(r"\bthree\W+two\b|\btwo\W+three\b").extract('one two three four five')
        'two three'

        * Near words.

        Two words (`'two'`, `'four'`) separated by any other.

        >>> MyKeywords(use_re=True).add_keywords(r"\btwo\W+\w+\W+four\b").extract('one two three four five')
        'two three four'


        Two words (`'two'`, `'five'`) separated by one, two or three unspecified words.

        >>> MyKeywords(use_re=True).add_keywords(r"\btwo\W+(?:\w+\W+){1,3}?five").extract('one two three four five')
        'two three four five'

        * Or operator.

        >>> MyKeywords(use_re=True).add_keywords(r"three|two").extract('one two three four five')
        'three|two'

        * And operator. One word followed by other at any word distance.

        >>> MyKeywords(use_re=True).add_keywords(r"\btwo\W+(?:\w+\W+)+?five") \
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

            if len(y):                
                result.extend(y)
        
        if len(result):
            return sep.join(set(result))
            
        return None



    def find(self, x):
        """Search expr in string x and returns a Boolean.

        Args:
            x (string): A string object.
            keywordsList (string, list): string or list of strings searched. Each element of `keywordList` is interpreted as a regular expression.
            ignoreCase (bool):  ignore case?
            fullMatch (bool): the whole string matches the regular expression in keywordList.
            
        Returns:
            Boolean.

        """
        if self.extract(x) is None:
            return False
        else:
            return True

    def remove(self, x):
        """Returns a string removing the strings that match a 
        list of keywords from x.

        Args:
            x (string): A string object.
            
        Returns:
            String.
                
        """

        y = self.extract(x)
        if y != '':
            result = re.sub(y, ' ', x)
        else:
            result = x
        return result

        
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

        >>> df.f.map(lambda x: MyKeywords().add_keywords('aaa').extractNearby(x, n_words=2))
        0               
        1               
        2    1 2 aaa 3 4
        3    2 3 aaa 4 5
        4               
        5               
        Name: f, dtype: object
        """

        y = self.extract(x)
            
        if len(y):

            if self._ignore_case is True:
                c = re.compile('\w+\W+'*n_words + y + '\W+\w+'*n_words, re.I)
            else:
                c = re.compile('\w+\W+'*n_words + y + '\W+\w+'*n_words)

            if self._full_match is True:
                result = c.fullMatch(x)
            else:
                result = c.findall(x)

            if len(result):
                return result[0]
            else:
                return ''

        return ''

