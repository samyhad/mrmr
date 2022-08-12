from sklearn.base import BaseEstimator, TransformerMixin
from mrmr import mrmr_classif

class mrmr_customized(BaseEstimator, TransformerMixin):
    """
    Do a bunch of fancy, specialized text normalization steps,
    like tokenization, part-of-speech tagging, lemmatization
    and stopwords removal.
    """
    ...

    def __init__(self, k):
        self.k = k

    def mrmr_(self, X, y):
        #selected_features = mrmr_classif(X=X, y=y, K=self.k)
        #self.sf = selected_features
        # do the special stuff here
        return mrmr_classif(X=X, y=y, K=self.k)

    def fit(self, X, y=None):
        return mrmr_classif(X=X, y=y, K=self.k)

    """def transform(self, documents):
        for document in documents:
            yield self.normalize(document)"""