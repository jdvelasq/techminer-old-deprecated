"""
TechMiner.docs_transformer
author: Alejandro Cadavid Romero
==================================================================================================

"""
import spacy
import unicodedata
import re
from sklearn.base import TransformerMixin


class DocsTransformer(TransformerMixin):

    def __init__(self):
        self.__pipeline = None
        self.__stopwords = None
        self.__language = 'English'

    def _load_spacy_resources(self):
        try:
            nlp = spacy.load('en_core_web_sm')
            stopwords_list = spacy.lang.es.stop_words.STOP_WORDS
        except OSError as es:
            print('spacy english language model not found try python -m spacy download en_core_web_sm')
        self.__pipeline = nlp
        self.__stopwords = stopwords_list
        return self

    @property
    def nlp(self):
        return self.__pipeline

    @property
    def stopwords(self):
        return self.__stopwords

    def fit(self, X, y=None):
        self._load_spacy_resources()
        return self

    def transform(self,
                  docs,
                  column: str):
        """
        Method to clean the text in the documents stored in a RecordsDataFrame
        """

        docs_ = docs[[column]].copy()
        new_docs_ = []
        stopwords_list = list(self.__stopwords)

        for doc in docs_:
            doc = unicodedata.normalize('NFKD', str(doc)) \
                .encode('ascii', errors='ignore').decode('utf-8') \
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
