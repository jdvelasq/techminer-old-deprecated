"""
TechMiner.docs_normalizer
author: Alejandro Cadavid Romero
==================================================================================================

"""
import spacy
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from techminer import RecordsDataFrame
import pandas as pd


class DocNormalizer(BaseEstimator, TransformerMixin):
    """Base transformer class for normalize documents.
    Steps performes:
        * lowercase  
        * strip special characters  
        * strip whitespaces  
        * lemmatization  
        * remove stopwords  
        
    Parameters
    ----------
    
    words_keep: list, optional, default: None 
    list of words to keep from removing stop words from documents.
    """

    def __init__(self, words_keep = None):
        self.__pipeline = None
        self.stopwords_ = None
        self.words_keep_ = words_keep

    def __load_spacy_resources(self):
        """ Loads the spacy nlp pipeline resources and 
        stopwords to assign them as attributes for the object
        """
        try:
            nlp = spacy.load('en_core_web_sm')
            stopwords_list = spacy.lang.en.stop_words.STOP_WORDS
        except OSError:
            print('spacy english language model not found try python -m spacy download en_core_web_sm')
        self.__pipeline = nlp
        self.stopwords_ = stopwords_list
        return self

    def fit(self, X, y=None):
        """ Fit method:  
            * loads spacy resources  
            * If words_keep is not None, then words_keep are removed from stopwords list
        """
        self.__load_spacy_resources()
        if self.words_keep_ is not None:
            try:
                assert type(self.words_keep_) == list
            except:
                TypeError("If words_keep is not None, a list must be passed")
            stopwords = list(self.stopwords_)
            words_keep = [word.lower() for word in self.words_keep_]
            for word_1, word_2 in zip(self.words_keep_, words_keep):
                if word_2 in stopwords:
                    print(f'Word == {word_1} ==  removed from stopwords list')
            stopwords = [word for word in stopwords if word not in words_keep]
            V_stopwords = len(stopwords)
            self.stopwords_ = stopwords
        else:
            self.stopwords_ = list(self.stopwords_)
            V_stopwords = len(self.stopwords_)       

        print(f"Loaded {V_stopwords} stopwords")
        return self

    def transform(self, docs):
        """
        Transform method to clean the text of a column of a RecordsDataFrame object
        """

        try:
            getattr(self, "stopwords_")
        except AttributeError:
            raise RuntimeError("You must fit the normalizer before any transformation")

        try:
            assert type(docs) == pd.Series
        except:
            raise TypeError("DocNormalizer only can clean one column of text type from a RecordsDataFrame")
        
        print(f'Normalizing documents')
        
        docs_ = docs.copy()
        new_docs_ = []
        stopwords_list = list(self.stopwords_)

        for doc in docs_:
            doc = unicodedata.normalize('NFKD', str(doc))\
                .encode('ascii', errors='ignore').decode('utf-8')\
                    .lower().replace(r'\\n', '').replace(r'-', '').rstrip()
            doc = re.sub('[^a-zA-Z]', ' ', str(doc))
            doc = self.__pipeline(doc)
            text = ' '.join([word.lemma_ for word in doc
                             if word.lemma_.isalpha()
                             and word.text != ' '
                             and word.lemma_ != '-PRON-'
                             and word.lemma not in stopwords_list])
            new_docs_.append(text)
        return new_docs_