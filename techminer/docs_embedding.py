"""
TechMiner.docs_embeddings
author: Alejandro Cadavid Romero
==================================================================================================

"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DocEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Base transformer class to get average word embeddings from GloVe.


    Parameters
    ----------
     
    dimension: int, default: 50 
    Number of word embeddings features, only receives 50 or 100

    
    """
    
    def __init__(self, dimension:int = 50):
        self.dimension_ = dimension
        self.__available_dimensions = [50,100]
    
    def __load_embeddings(self):
        print("Loading word vectors...")
        word2vec = {}
        embedding = []
        idx2word = []
        if self.dimension_ not in self.__available_dimensions:
            raise ValueError("dimension argument only acept 50 or 100 feature vector")
        
        with open(f"../data/embeddings/glove/glove.6B.{self.dimension_}d.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)

        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.vocabulary_size_, self.feature_size_ = self.embedding.shape
        return self

    def fit(self, data):
        """ Fit method that loads the word embeddings from the dimensions specified in the constructor
        """
        self.__load_embeddings()
        return self

    def transform(self, data):
        """ Transform method that get words embeddings
        """
        try:
            getattr(self, "vocabulary_size_")
        except AttributeError:
            print("You must fit the normalizer before any transformation")

        try:
            assert type(data) == pd.Series
        except:
            raise TypeError("DocNormalizer only can clean one column of text type from a RecordsDataFrame")            


        X = np.zeros((len(data), self.feature_size_))
        n = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
        return X