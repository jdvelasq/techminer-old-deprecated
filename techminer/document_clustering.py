"""
TechMiner.document_clustering
author: Alejandro Cadavid Romero
==================================================================================================

"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.utils.validation import check_array
from scipy.cluster.hierarchy import dendrogram, linkage
from techminer.dataframe import RecordsDataFrame


class DocumentClustering(TransformerMixin, BaseEstimator):
    # Class attributes
    __reduce_techniques = ['PCA', 'SVD']

    def __init__(self,
                 method: str
                 , reduction_technique: str = None
                 , num_dimensions: int = None
                 , n_clusters: int = 2
                 , vectorize: bool = True
                 , use_tfidf: bool = True):
        self._method = method
        self._reduction_technique = reduction_technique
        self._n_clusters = n_clusters
        self._vectorize = vectorize
        self._use_tfidf = use_tfidf
        self._vectorizer = None
        self._dfm = None
        self._transformer = None
        self._vocabulary = None
        self._cluster_labels = None
        self._fitted = False

        if reduction_technique is not None:
            if num_dimensions is None:
                raise ValueError(f'Number of dimensions was not specified')
            else:
                self._num_dimensions = num_dimensions
        else:
            self._num_dimensions = None

    def vectorize_text(self,
                       text,
                       column: str,
                       min_df: int = 1,
                       max_df: float = 1.0):

        text_ = text[[column]].copy()
        if self._use_tfidf:
            bag_of_words = CountVectorizer(text_, min_df=min_df, max_df=max_df)
            tfidf = TfidfTransformer()
            vectorizer = make_pipeline(bag_of_words, tfidf)
            dfm = vectorizer.fit_transform(text_)
            self._vectorizer = vectorizer
            self._vocabulary = vectorizer.steps[0][1].vocabulary_
            self._dfm = dfm
            return self
        else:
            vectorizer = CountVectorizer(text_, min_df=min_df, max_df=max_df)
            dfm = vectorizer.fit_transform(text_)
            self._vectorizer = vectorizer
            self._vocabulary = vectorizer.vocabulary_
            self._dfm = dfm
            return self

    def _reduce_dimensions(self):
        if self._reduction_technique not in DocumentClustering.__reduce_techniques:
            raise ValueError(f'reduction technique {self._reduction_technique} is not valid')
        elif self._reduction_technique == 'PCA':
            reducer = PCA(n_components=self._num_dimensions)
        else:
            reducer = TruncatedSVD(n_components=self._num_dimensions)
        normalizer = Normalizer(copy=False)
        normalizer_after = Normalizer(copy=False)
        transformer = make_pipeline(normalizer, reducer, normalizer_after)
        self._dfm = transformer.fit_transform(self._dfm)
        self._transformer = transformer
        return self

    @property
    def dfm(self):
        return self._dfm

    @property
    def vocabulary(self):
        return self._vocabulary

    def _cluster_kmeans(self):
        kmeans = KMeans(n_clusters=self._n_clusters)
        kmeans.fit(self._dfm)
        self._cluster_labels = kmeans.predict(self._dfm)
        self._cluster_method = kmeans
        return self

    def _agglomerative_cluster(self):
        pass

    def fit(self, X, y=None):
        if self._vectorize:
            self.vectorize_text(X)
        else:
            X = check_array(X, accept_sparse=['csr', 'csc'],
                            ensure_min_features=2)
            self._dfm = X
        if self._reduction_technique is not None:
            self._reduce_dimensions()

        self._cluster_kmeans()
        self._fitted = True
        return self

    def predict(self, X):
        if self._fitted is True:
            if self._vectorize:
                X = self._vectorizer.transform(X)
            if self._reduction_technique is not None:
                X = self._transformer.transform(X)
            labels = self._cluster_method.predict(X)
            return labels

    @property
    def cluster_labels(self):
        return self._cluster_labels
