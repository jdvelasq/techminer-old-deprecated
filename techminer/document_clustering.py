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

    Attributes
    ----------
    vocabulary_: dict 
        vocabulary identified in the corpus of documents
    explained_var_components_: array of shape (n_components,) 
        Information contained in each of the components
    explained_var_: float
        Total information contained in the n_components
    components_: array of shape (n_components, vocabulary) 
        Each row contains the coefficients found in the spectral decomposition, which can be used as a linear combination of words
    cluster_labels_: list 
        Labels of the clusters found with the k-means algorithm
    within_sum_squares_: float
        Sum of sum of squares of the difference between the centroids and the observations belonging to that centroid 
    silhouette_score_: float
        Measurement of how similar a document is to its own cluster compared to other clusters.

    Examples
    --------
    >>> import pandas as pd
    >>> from techminer import DocNormalizer
    >>> from techminer import DocumentClustering
    >>> from techminer import RecordsDataFrame
    >>> rdf = RecordsDataFrame(
    ...     pd.read_json('./data/cleaned.json', orient='records', lines=True)
    ... )
    >>> doc_normalizer = DocNormalizer()
    >>> doc_normalizer.fit(rdf['Abstract'])
    >>> rdf['Abstract_cleaned'] = doc_normalizer.transform(rdf['Abstract'])
    >>> document_clustering = DocumentClustering(vectorize=True, 
                                         min_count=5,
                                         max_count=1.0, 
                                         use_tfidf=True, 
                                         reduce_dimensions=True, 
                                         n_components=100,
                                         n_clusters=6, 
                                         random_state=42)
    >>> document_clustering.fit(rdf.loc[:,'Abstract_cleaned'])
    >>> document_clustering.dfm[0]
    array([ 7.13428736e-01, -4.76060165e-02,  5.29940552e-02,  8.73025970e-02,
       -7.45694045e-02,  1.42941227e-01, -5.09853411e-02, -1.78203898e-01,
        8.52714009e-02, -1.00128667e-01, -2.31763222e-02,  5.40155273e-03,
       -5.04605826e-02, -9.81179696e-02,  1.94404247e-02, -2.14461019e-03,
        7.85441508e-02, -1.02662513e-01,  5.81985038e-03, -3.76739637e-02,
        3.45761128e-02, -2.22396057e-02, -2.93425749e-02,  5.89096797e-03,
        1.89613453e-02, -8.45293248e-02, -8.78250107e-02,  9.49944220e-02,
       -7.06022545e-02, -1.99971986e-02, -2.80765440e-02,  1.77894762e-02,
       -9.15141236e-02,  9.32812885e-02, -3.69038879e-02,  4.83552399e-02,
        3.76390947e-02, -2.07317645e-03, -7.05211776e-02, -6.68652865e-02,
       -5.91281772e-02,  5.88780287e-02, -6.93434412e-05,  2.83905118e-03,
        2.17129828e-02, -1.75907532e-01,  4.74205535e-02,  8.86449161e-03,
       -1.13561354e-02, -1.50931675e-02,  8.09334218e-02,  2.58106542e-02,
        5.44487107e-02,  7.73061652e-02, -4.50092168e-03,  3.92250566e-02,
       -4.51770619e-02,  6.14419294e-02, -2.04833581e-01,  6.95145643e-04,
       -8.94358270e-02,  7.87068150e-02, -9.79536397e-02, -6.77876865e-03,
        3.34612847e-02, -1.44266785e-02, -1.00398079e-01,  1.02428170e-01,
       -6.21642074e-02, -3.52210453e-02, -7.78938220e-02,  2.55518971e-02,
        1.02093170e-01,  1.66019773e-01,  1.03667849e-01, -5.40555791e-02,
       -8.24466984e-02, -3.80644714e-02,  5.81702300e-02,  2.07751936e-02,
        9.29880597e-02,  7.86332188e-02, -8.38177667e-03,  4.37609253e-02,
       -9.47405749e-02, -1.17851382e-01,  4.72879828e-02,  3.04537631e-02,
        3.60489849e-02, -5.33173177e-02, -3.91417663e-02, -7.40816356e-02,
       -6.40443625e-03, -6.43947848e-02,  5.78075056e-02,  7.62550092e-02,
       -6.46091877e-02, -4.01021078e-02,  5.87939975e-02,  3.22097256e-03])
    >>> document_clustering.explained_var_
    0.902145389871721
    >>> document_clustering.silhouette_score_
    0.05536640934412707
    >>> rdf['Cluster_labels'] = document_clustering.cluster_labels_
    >>> rdf['Cluster_labels'].value_counts()
    1    97
    0    25
    5    10
    3     6
    4     3
    2     3
    Name: Cluster_labels, dtype: int64
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
