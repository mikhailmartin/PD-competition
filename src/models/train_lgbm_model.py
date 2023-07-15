import os
from typing import Dict

import joblib
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import lightgbm
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from src import constants

TRAIN_DATA_PATH = os.path.join('..', '..', 'data', 'PD-data-train.csv')
MODEL_PATH = os.path.join('..', '..', 'models', 'lgbm_pipeline.pkl')
STUDY_PATH = os.path.join('..', '..', 'studies', 'lgbm_study.pkl')
RESULTS_PATH = os.path.join('..', '..', 'results.csv')
N_SPLITS = 10
N_TRIALS = 100

train_data = pd.read_csv(TRAIN_DATA_PATH, sep=';', index_col='record_id')

y = train_data['default_12m']
X = train_data.drop(columns='default_12m')


class AsCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = np.array(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in X.columns:
            X[feature] = pd.Categorical(X[feature])

        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns


def objective(trial):
    """Целевая функция для оптимизации гиперпараметров."""
    model = get_model(trial, mode='fit')
    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        scoring='roc_auc',
        cv=StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=constants.RANDOM_STATE),
    )

    return scores.mean()


def get_model(trial: optuna.trial.Trial, mode: str) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ('drop', 'drop', ['ul_systematizing_flg']),
            ('as_category', AsCategory(), ['ul_staff_range']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    lgbm = lightgbm.LGBMClassifier(**get_init_hyperparams(trial, mode))

    model = Pipeline([
        ('preprocessing', preprocessing),
        ('lgbm', lgbm),
    ])

    return model


def get_init_hyperparams(trial: optuna.trial.Trial, mode: str) -> Dict:
    if mode == 'fit':
        # Core Parameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', .05, 1., log=True)
        num_leaves = trial.suggest_int('num_leaves', 2, 6)
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'rf'])  # 'dart'

        # Learning Control Parameters
        max_depth = trial.suggest_int('max_depth', 1, 3)
        min_child_samples = trial.suggest_int('min_child_samples', 20, 50)
        reg_alpha = trial.suggest_float('reg_alpha', .5, 2)
        reg_lambda = trial.suggest_float('reg_lambda', .5, 2)
        subsample = trial.suggest_float('subsample', .05, 1.)
        subsample_freq = trial.suggest_int('subsample_freq', 1, 5)

        min_split_gain = trial.suggest_float('min_split_gain', .01, 1.)
    elif mode == 'refit':
        # Core Parameters
        n_estimators = trial.params['n_estimators']
        learning_rate = trial.params['learning_rate']
        num_leaves = trial.params['num_leaves']
        boosting_type = trial.params['boosting_type']

        # Learning Control Parameters
        max_depth = trial.params['max_depth']
        min_child_samples = trial.params['min_child_samples']
        reg_alpha = trial.params['reg_alpha']
        reg_lambda = trial.params['reg_lambda']
        subsample = trial.params['subsample']
        subsample_freq = trial.params['subsample_freq']

        min_split_gain = trial.params['min_split_gain']
    else:
        assert False, 'ABOBA'

    hyperparams = dict(
        # Core Parameters
        objective='binary',
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        n_jobs=-1,
        random_state=constants.RANDOM_STATE,

        # Learning Control Parameters
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,

        # Objective Parameters
        is_unbalance=True,
    )

    if boosting_type == 'gbdt':
        hyperparams['learning_rate'] = learning_rate
        hyperparams['reg_alpha'] = reg_alpha
        hyperparams['reg_lambda'] = reg_lambda
    elif boosting_type == 'rf':
        hyperparams['subsample'] = subsample
        hyperparams['subsample_freq'] = subsample_freq

    return hyperparams


if __name__ == '__main__':
    study = optuna.create_study(
        sampler=TPESampler(seed=constants.RANDOM_STATE),
        direction='maximize',
        study_name='lgbm',
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    model = get_model(study.best_trial, mode='refit')
    model.fit(X, y)

    print('Лучшие гиперпараметры:')
    default_hyperparams = [
        'n_estimators', 'num_leaves', 'max_depth', 'min_child_samples', 'boosting_type', 'min_split_gain',
    ]
    gbdt_hyperparams = ['learning_rate', 'reg_alpha', 'reg_lambda']
    rf_hyperparams = ['subsample', 'subsample_freq']
    if study.best_trial.params['boosting_type'] == 'gbdt':
        hyperparams = default_hyperparams + gbdt_hyperparams
    elif study.best_trial.params['boosting_type'] == 'rf':
        hyperparams = default_hyperparams + rf_hyperparams
    else:
        assert False, 'ABOBA'
    for hyperparam in hyperparams:
        print(f'* {hyperparam}: {study.best_trial.params[hyperparam]}')
    print(f'Лучший mean AUC-ROC: {study.best_trial.value}')

    results = pd.read_csv(RESULTS_PATH, index_col=0)
    best_aucroc = study.best_trial.value
    if results.loc['meanCV AUC-ROC', 'lgbm'] < best_aucroc:
        results.loc['meanCV AUC-ROC', 'lgbm'] = best_aucroc
        results.to_csv(RESULTS_PATH)

        joblib.dump(study, STUDY_PATH)
        joblib.dump(model, MODEL_PATH)
