import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.metrics import recall_score, precision_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

problem_title = 'Graphs of Wikipedia: Influential Thinkers'

_target_column_name = 'link'
_ignore_column_names = ['index']
_prediction_label_names = [0,1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
#workflow = rw.workflows.FeatureExtractorClassifier()

class Wikipedia(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier', '	nodes_info_new.csv']):
        super(Wikipedia, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = Wikipedia()

#--------------------------------------------
# Scoring
#--------------------------------------------

class Precision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, true, pred):
        return precision_score(true, pred)

class Recall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, true, pred):
        return recall_score(true, pred)


score_types = [Precision(), Recall()]


#--------------------------------------------
# Cross validation
#--------------------------------------------


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
