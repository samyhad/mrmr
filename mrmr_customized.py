from sklearn.base import BaseEstimator, TransformerMixin
from mrmr import mrmr_classif
import pandas as pd

class mrmr_customized(BaseEstimator, TransformerMixin):

    def __init__(self, k):
        self.k = k
        self.df_selected_features = []

    def mrmr(self, X, y):
        df_y = pd.DataFrame(y)
        df_X = pd.DataFrame(X)
        sc = mrmr_classif(X=df_X, y=df_y, K=self.k)
        self.df_selected_features = df_X[sc]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        return self.mrmr(X, y)