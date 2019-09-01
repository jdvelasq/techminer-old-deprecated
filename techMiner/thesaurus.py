"""
techMiner.Thesaurus
===============================

"""
import pandas as pd
import json
from techMiner.transform import fingerprint

class Thesaurus:
    def __init__(self, name_strategy='mostfrequent', search_strategy='fingerprint'):
        self._thesaurus = None
        self._name_strategy = name_strategy
        self._search_strategy = search_strategy


    def fit(self, x, sep=None):
        """Create a dictionary for list cleanup.

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...    'f': ['a b c a b a', 
        ...          'a b c a b',
        ...          'a b c a b', 
        ...          'A C b', 
        ...          'a b', 
        ...          'a, b, c, a', 
        ...          'a B'],
        ... })
        >>> df
                     f
        0  a b c a b a
        1    a b c a b
        2    a b c a b
        3        A C b
        4          a b
        5   a, b, c, a
        6          a B

        >>> print(Thesaurus().fit(df.f).to_json()) # doctest: +NORMALIZE_WHITESPACE
        {
          "a b": [
            "a B",
            "a b"
          ],
          "a b c a b": [
            "A C b",
            "a b c a b",
            "a b c a b a",
            "a, b, c, a"
          ]
        }

        >>> print(Thesaurus(name_strategy='shortest').fit(df.f).to_json()) # doctest: +NORMALIZE_WHITESPACE
        {
          "A C b": [
            "A C b",
            "a b c a b",
            "a b c a b a",
            "a, b, c, a"
          ],
          "a b": [
            "a B",
            "a b"
          ]
        }

        >>> print(Thesaurus(name_strategy='longest').fit(df.f).to_json()) # doctest: +NORMALIZE_WHITESPACE
        {
          "a B": [
            "a B",
            "a b"
          ],
          "a b c a b a": [
            "A C b",
            "a b c a b",
            "a b c a b a",
            "a, b, c, a"
          ]
        }

        >>> df = pd.DataFrame({
        ...    'f': ['a b, c a, b a', 
        ...          'A b, c A, b',
        ...          'a b, C A, B', 
        ...          'A C, b', 
        ...          None,
        ...          'a b', 
        ...          'a, b, c, a', 
        ...          'a B'],
        ... })
        >>> df
                       f
        0  a b, c a, b a
        1    A b, c A, b
        2    a b, C A, B
        3         A C, b
        4           None
        5            a b
        6     a, b, c, a
        7            a B

        >>> import json
        >>> print(Thesaurus(name_strategy='longest').fit(df.f, sep=',').to_json()) # doctest: +NORMALIZE_WHITESPACE
        {
        "A C": [
            "A C",
            "C A",
            "c A",
            "c a"
        ],
        "a B": [
            "A b",
            "a B",
            "a b",
            "b a"
        ],
        "b": [
            "B",
            "b"
        ]
        }



        """

        if isinstance(x, dict):
            self._thesaurus = x
            return self

        x = x.dropna()

        if sep is not None:
            x = pd.Series([z.strip() for y in x for z in y.split(sep)]) 

        y = x.map(lambda w: fingerprint(w))
        y = y.sort_values()
        counts = y.value_counts()
        counts = counts[counts > 1]

        result = {}
        for z in counts.index.tolist():

            w = x[y == z]

            if self._name_strategy is None or self._name_strategy == 'mostfrequent':
                m = w.value_counts().sort_values()
                m = m[m == m[-1]].sort_index()
                groupName = m.index[-1]
                
            if self._name_strategy == 'longest' or self._name_strategy == 'shortest':
                m = pd.Series([len(a) for a in w], index = w.index).sort_values()
                if self._name_strategy == 'longest':
                    groupName = w[m.index[-1]]
                else:
                    groupName = w[m.index[0]]

            z = w.sort_values().unique().tolist()
            if len(z) > 1:
                result[groupName] = z

        self._thesaurus = result

        return self
    
        

    def transform(self, x, sep=None):
        """

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...    'f': [0, 1, 2, 3, None, 4, 5, 6, 7, 8, 9],
        ... })
        >>> df
              f
        0   0.0
        1   1.0
        2   2.0
        3   3.0
        4   NaN
        5   4.0
        6   5.0
        7   6.0
        8   7.0
        9   8.0
        10  9.0
        
        >>> d = {'a':[0, 1, 2],
        ...      'b':[4, 5, 6],
        ...      'c':[7, 8, 9]}
        >>> 

        >>> df.f.map(lambda x: Thesaurus().fit(d).transform(x))
        0       a
        1       a
        2       a
        3       3
        4     NaN
        5       b
        6       b
        7       b
        8       c
        9       c
        10      c
        Name: f, dtype: object

        >>> df = pd.DataFrame({
        ...    'f': ['a b, A B', 'A b, A B', None, 'b c', 'b, B A', 'b, a, c', 'A, B'],
        ... })
        >>> df
                  f
        0  a b, A B
        1  A b, A B
        2      None
        3       b c
        4    b, B A
        5   b, a, c
        6      A, B
        >>> d = {'0':['a b', 'A B', 'B A'],
        ...      '1':['b c'],
        ...      '2':['a', 'b']}
        >>> df.f.map(lambda x: Thesaurus().fit(d).transform(x, sep=','))
        0      0,0
        1    A b,0
        2     None
        3        1
        4      2,0
        5    2,2,c
        6      A,B
        Name: f, dtype: object


        """

        if x is None:
            return None

        if sep is None:
            for k in self._thesaurus.keys():
                if x in self._thesaurus[k]:
                    return k
            return x
        else:
            result = []
            for z in x.split(sep):
                z = z.strip()
                found = False
                for k in self._thesaurus.keys():
                    if z in self._thesaurus[k]:
                        result += [k]
                        found = True
                        break
                if found is False:    
                    result += [z]
            return sep.join(result)


    def to_json(self):
        """Returns a json representation of the Thesaurus.
        """
        return json.dumps(self._thesaurus, indent=2, sort_keys=True)


    def merge_keys(self, key, popkey):
        """Adds the strings associated to popkey to key and delete popkey.
        """
        if isinstance(popkey, list):
            for k in popkey:
                self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[k]        
                self._thesaurus.pop(k)
        else:
            self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[popkey]
            self._thesaurus.pop(popkey)

    def pop_key(self, key):
        """Deletes key from thesaurus.
        """
        self._thesaurus.pop(key)
        
    def change_key(self, current_key, new_key):
        self._thesaurus[new_key] = self._thesaurus[current_key]
        self._thesaurus.popkey(current_key)
