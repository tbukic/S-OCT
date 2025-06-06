"""OCT (our reimplementation) comprehensive experiments."""
from datetime import datetime
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, cross_validate

from scripts.consts import proj_paths
from s_oct.oct_comprehensive import OCT
from scripts.datasets import (
    load_balance_scale, load_congressional_voting_records, load_soybean_small,
    load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons
)

categorical_datasets = [load_balance_scale, load_congressional_voting_records, load_soybean_small]
sklearn_datasets = [load_iris, load_wine, load_breast_cancer]
numerical_datasets = sklearn_datasets + [load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons]
datasets = categorical_datasets + numerical_datasets

time_limit = 600

for dataset in datasets:
    dataset_name = dataset.__name__[5:]
    for max_depth in [2, 3, 4]:
        print(f"***** {datetime.now().time()} "
              f"| {dataset_name} "
              f"| max_depth={max_depth} *****")
        if dataset in sklearn_datasets:
            X, y = dataset(return_X_y=True, as_frame=True)
        else:
            X, y = dataset()
        tree = OCT(max_depth=max_depth, hyperplanes=False, time_limit=time_limit)
        if dataset in categorical_datasets:
            pipeline = Pipeline([
                ('pre', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')),
                ('tree', tree)
            ])
        else:
            pipeline = Pipeline([
                ('pre', MinMaxScaler(feature_range=(0.00001, 0.99999))),
                ('tree', tree)
            ])
        if max_depth == 2:
            max_splits_settings = [2,3]
        elif max_depth == 3:
            max_splits_settings = [3,5,7]
        else:
            max_splits_settings = [5,10,15]
        clf = GridSearchCV(pipeline, cv=2, param_grid=dict(tree__max_splits=max_splits_settings))
        X, y = shuffle(X, y, random_state=0) # cross_validate does not shuffle, some datasets like balance scale require shuffling
        cv_results = cross_validate(clf, X, y, cv=3, return_train_score=True, return_estimator=True, error_score='raise')
        train_scores = cv_results['train_score']
        test_scores = cv_results['test_score']
        estimators = cv_results['estimator']
        train_times = [e.best_estimator_.named_steps['tree'].fit_time_ for e in estimators]
        max_splits_values = [e.best_estimator_.named_steps['tree'].max_splits for e in estimators]
        line = ["OCT-reimplementation", dataset_name, max_depth,
                *train_scores, *test_scores, *train_times,
                *max_splits_values]
        line = [str(x) for x in line]
        with open(proj_paths.results.comprehensive, 'a') as f:
            f.write(', '.join(line) + '\n')
