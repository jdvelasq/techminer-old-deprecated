import re
import pandas as pd
import unicodedata
from techminer.common import nlp_resource, load_stopwords
from techminer.result import Result
from techminer.dataframe import RecordsDataFrame

def clean_text(rdf, columns = None, min_len = 4, language = 'english'):
    
    if not isinstance(rdf, techminer.dataframe.RecordsDataFrame):
        raise TypeError('A RecordsDataFrame must be provided')
    
    if columns == None:
        raise NameError('No columns provided to perform the text preprocessing\n')
        
    nlp = nlp_resource(language=language)
    stopwords_list = load_stopwords(language=language)
        
    if isinstance(columns, str):
        rows = []
        for text in rdf[columns]:
            text = unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8')\
            .lower().replace(r'\\n','').replace(r'-','')
            text = re.sub("[^a-zA-Z]", " ", str(text))
            text = ' '.join(
                    [word.lemma_ for word in nlp(text)
                    if len(word.text) >= min_len 
                    and word.lemma_.isalpha() 
                    and word.text != ' '
                    and word.lemma_ != '-PRON-'
                    and word.lemma not in stopwords_list]
                    )
            rows.append(text)
        return RecordsDataFrame(pd.DataFrame(rows, columns=[columns + '_cleaned']))
    
    if isinstance(columns, list):
        new_cols = {}
        for col in columns:
            rows = []
            for text in rdf[col]:
                text = unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8')\
                .lower().replace(r'\\n','').replace(r'-','')
                text = re.sub("[^a-zA-Z]", " ", str(text))
                text = ' '.join(
                        [word.lemma_ for word in nlp(text)
                        if len(word.text) >= min_len 
                        and word.lemma_.isalpha() 
                        and word.text != ' '
                        and word.lemma_ != '-PRON-'
                        and word.lemma not in stopwords_list]
                        )
                rows.append(text)
            new_cols[col + '_cleaned'] = rows
        return RecordsDataFrame(pd.DataFrame(new_cols))