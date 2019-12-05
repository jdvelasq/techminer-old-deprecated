"""
TechMiner.document_clustering
author: Alejandro Cadavid Romero
==================================================================================================

"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from techminer.dataframe import RecordsDataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot


class DocumentClustering(BaseEstimator, TransformerMixin):
    """Document clustering with Latent Semantic Analysis.
    Steps performes:
        * Vectorization using Bag of Words model or Term Frequency - Inverse Document Frequency  
        * Dimensionality reduction using Singular Value Decomposition (LSA)  
        * Clustering using K-Means  
        
     Parameters
    ----------

    vectorize: bool, default: True
    If the raw text or clean text is going to be passed, 
    a vectorization of the words must be performed

    min_count: float or int, default: 1
    The minimum number of occurrences that a word must 
    appear in the corpus to add it in the document-term matrix.
    If int, is the frequency threshold.
    If float, must be in the range of [0.0, 1.0] and represent the
    proportion of documents that the term must appear in.

    max_count: float or int, default: 1.0
    The maximum number of occurrences that a word can 
    appear in the corpus to add it in the document-term matrix.
    If int, is the frequency threshold.
    If float, must be in the range of [0.0, 1.0] and represent the
    proportion of documents that the term can appear in.

    use_tfidf: bool, default = True
    If the inverse document frequency must be computer and applied to the 
    document-term matrix

    reduce_dimensions: bool, default = True
    """


    def __init__(self
                , vectorize = True
                , min_count = 1
                , max_count = 1.0
                , use_tfidf = True
                , reduce_dimensions = True
                , n_components = 10
                , n_clusters = 2
                , random_state = None):
                 
        # Public attributes
        self.random_state = random_state

        ## Attributes for vectorization of text
        self.vectorize = vectorize
        self.use_tfidf = use_tfidf
        self.min_count = min_count
        self.max_count = max_count
        self.vocabulary_ = None

        ## Attributes for dimensionality reduction
        self.reduce_dimensions = reduce_dimensions
        if reduce_dimensions is False:
            if n_components is not None:
                self.n_components = None
        else:
            self.n_components = n_components

        self.explained_var_components_ = None
        self.explained_var_ = None
        self.components_ = None

        ## Attributes for clustering
        self.n_clusters = n_clusters
        self.cluster_labels_ = None
        self.within_sum_squares_ = None
        self.silhouette_score_ = None

        # Private attributes

        ## Attributes for vectorization of text
        self.__vectorizer = None
        self.__dfm = None 
        
        self.__cluster_model = None
        
        self.__fitted = False
               


    def __vectorize_text(self 
                        , docs: pd.Series):

        try:
            assert type(docs) == pd.Series
        except:
            raise TypeError("DocNormalizer only can clean one column of text type from a RecordsDataFrame")

        docs_ = docs.copy()
        if self.use_tfidf:
            bag_of_words = CountVectorizer(min_df=self.min_count, max_df=self.max_count)
            tfidf = TfidfTransformer()
            vectorizer = make_pipeline(bag_of_words, tfidf)
            dfm = vectorizer.fit_transform(docs_)
            self.__vectorizer = vectorizer
            self.vocabulary_ = vectorizer.steps[0][1].vocabulary_
            self.__dfm = dfm
            return self
        else:
            vectorizer = CountVectorizer(min_df=self.min_count, max_df=self.max_count)
            dfm = vectorizer.fit_transform(docs_)
            self.__vectorizer = vectorizer
            self.vocabulary_ = vectorizer.vocabulary_
            self.__dfm = dfm
            return self

    def __reduce_dimensions(self):

        # Latent Semantic Analysis step
        random_state = check_random_state(self.random_state)
        U, Sigma, VT = randomized_svd(self.__dfm, n_components=self.n_components, n_iter=100, random_state=random_state)
        
        # term - concept matrix
        self.components_ = VT
        X_reduced = U * Sigma

        # Calculate explained variance per each component of X reduced
        explained_var_components_ = np.var(X_reduced, axis = 0)
        _, total_variance = mean_variance_axis(self.__dfm, axis = 0)
        total_variance = total_variance.sum()
        self.explained_var_components_ = explained_var_components_
        self.explained_var_ = (explained_var_components_ / total_variance).sum()

        # Normalization of X reduced
        X_normalized = normalize(X_reduced)

        self.__dfm = X_normalized
        return self


    def __cluster_kmeans(self):
        # K-means step
        random_state = check_random_state(self.random_state)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        kmeans.fit(self.__dfm)
        cluster_labels = kmeans.predict(self.__dfm)
        self.cluster_labels_ = cluster_labels
        self.within_sum_squares_ = kmeans.inertia_
        self.silhouette_score_ = silhouette_score(self.__dfm, cluster_labels, metric = 'cosine')
        self.__cluster_model = kmeans
        return self


    def fit(self, X, y=None):
        if self.vectorize:
            self.__vectorize_text(X)
        else:
            X = check_array(X, accept_sparse=['csr', 'csc'],
                            ensure_min_features=2)
            self.__dfm = X
        if self.reduce_dimensions:
            self.__reduce_dimensions()
        
        self.__cluster_kmeans()
        self.__fitted = True
        return self


    def predict(self, X):
        if self.__fitted is True:
            # Vectorization step
            if self.vectorize:
                X_vect = self.__vectorizer.transform(X)
            else:
                X_vect = check_array(X, accept_sparse=['csr', 'csc'],
                                    ensure_min_features=2)
            # Dimension reduction step
            if self.reduce_dimensions:
                X_vect = safe_sparse_dot(X_vect, self.components_.T)
                X_vect = normalize(X_vect)
            # Clustering step
            labels = self.__cluster_model.predict(X_vect)
            return labels
        else:
            raise RuntimeError("You must fit the estimator before any prediction")
    

    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.cluster_labels_

    @property
    def dfm(self):
        return self.__dfm
