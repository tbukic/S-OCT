"""FlowOCT MIP comparison experiments."""
from datetime import datetime
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import OneHotEncoder

from scripts.consts import proj_paths
from scripts.datasets import (
    QuantileBucketizer, 
    load_balance_scale, load_congressional_voting_records, load_soybean_small,
    load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons
)
from s_oct.flow_oct import FlowOCT

categorical_datasets = [load_balance_scale, load_congressional_voting_records, load_soybean_small]
sklearn_datasets = [load_iris, load_wine, load_breast_cancer]
numerical_datasets = sklearn_datasets + [load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons]
datasets = categorical_datasets + numerical_datasets

time_limit = 600

for benders in [False, True]:
    for dataset in datasets:
        dataset_name = dataset.__name__[5:]
        for max_depth in [2, 3, 4]:
            if dataset in sklearn_datasets:
                X, y = dataset(return_X_y=True, as_frame=True)
            else:
                X, y = dataset()
            if dataset in categorical_datasets:
                pre = OneHotEncoder(drop='if_binary', sparse_output=False)
            else:
                pre = QuantileBucketizer()
            X = pre.fit_transform(X)
            print(f"***** {datetime.now().time()} "
                  f"| {dataset_name} "
                  f"| max_depth={max_depth} *****")
            tree = FlowOCT(max_depth=max_depth, benders=benders, warm_start=False, time_limit=time_limit)
            tree.fit(X, y)
            train_time = tree.fit_time_
            ub = tree.model_.ObjBound
            lb = tree.model_.ObjVal
            mip_gap = tree.model_.MIPGap
            model_name = "FlowOCT-Benders" if benders else "FlowOCT"
            line = [model_name, dataset_name, max_depth,
                    train_time, ub, lb, mip_gap, "#N/A", "#N/A"]
            line = [str(x) for x in line]
            with open(proj_paths.results.mip_comparison, 'a') as f:
                f.write(', '.join(line) + '\n')
