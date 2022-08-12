# create some pandas data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


import pandas as pd

X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

import mrmr_customized
#from . import mrmr_customized

mrmr_filter = mrmr_customized.mrmr_customized(k=10)
clf = LinearSVC()
mrmr_pipe = make_pipeline(mrmr_filter, clf)
mrmr_pipe.fit(X_train, y_train)
y_pred = mrmr_pipe.predict(X_test)
#print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)