import os
import sys
from pathlib import Path
from typing import Tuple

root_dir = Path(os.getcwd()).parent.parent.parent.parent
sys.path.append(root_dir.as_posix())

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.tampering.evaluate import evaluate


class TamperingClassificator:
    def __init__(self, model_name: str, model_parameters=None):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.test_split_size = 0.3

    def set_data(self, X, y, ids):
        self.X, self.y = X, y
        self.ids = ids

    def build_model(self):
        params = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": 1,
        }
        if self.model_parameters is not None:
            params.update(self.model_parameters)
        if self.model_name == "simple_threshold":
            return DecisionTreeClassifier(**params)
        else:
            raise ValueError(f"Model name ({self.model_name}) unknown!")

    def split_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.test_split_size > 0:
            (X_train, X_test, y_train, y_test, ids_train, ids_test,) = train_test_split(
                self.X,
                self.y,
                self.ids,
                test_size=self.test_split_size,
                shuffle=True,
                stratify=self.y,
            )
            y_train = y_train.astype(np.int)
            y_test = y_test.astype(np.int)
        else:
            (X_train, X_test, y_train, y_test, ids_train, ids_test,) = (
                self.X,
                None,
                self.y,
                None,
                self.ids,
                None,
            )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            ids_train,
            ids_test,
        )

    def validate_model(self, kfold=5):
        X, _, y, _, _, _ = self.split_data()
        kfold = StratifiedKFold(kfold)

        models = []
        train_metrics, val_metrics = [], []
        for ith_fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, y_train, X_val, y_val = (
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
            )
            model = self.build_model()
            model.fit(X_train, y_train)
            train_metrics.append(evaluate(model, X_train, y_train))
            val_metrics.append(evaluate(model, X_val, y_val))
            models.append(model)
        train_metrics_summary = pd.DataFrame(train_metrics).mean()
        val_metrics_summary = pd.DataFrame(val_metrics).mean()

        return train_metrics_summary, val_metrics_summary, models

    def train(self):
        X, X_test, y, y_test, _, _ = self.split_data()
        model = self.build_model()
        model.fit(X, y)
        train_metrics = evaluate(model, X, y)
        if X_test is not None:
            test_metrics = evaluate(model, X_test, y_test)
        else:
            test_metrics = None

        return model, train_metrics, test_metrics
