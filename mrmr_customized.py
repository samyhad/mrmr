from sklearn.base import BaseEstimator, TransformerMixin
from mrmr import mrmr_classif
import pandas as pd

class mrmr_customized(BaseEstimator, TransformerMixin):

    def __init__(self, k):
        self.k = k

    def mrmr_(self, X, y):
        #selected_features = mrmr_classif(X=X, y=y, K=self.k)
        #self.sf = selected_features
        # do the special stuff here
        #return mrmr_classif(X=X, y=y, K=self.k)

        df_y = pd.DataFrame(y)
        df_X = pd.DataFrame(X)
        sc = mrmr_classif(X=df_X, y=df_y, K=self.k)
        df_selected_features = df_X[sc]
        return df_selected_features

    def fit(self, X, y=None):
        df_y = pd.DataFrame(y)
        df_X = pd.DataFrame(X)
        sc = mrmr_classif(X=df_X, y=df_y, K=self.k)
        df_selected_features = df_X[sc]
        return df_selected_features

    def transform(self, X):
        pass