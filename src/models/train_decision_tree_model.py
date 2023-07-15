import os
from typing import Dict

import joblib
import optuna
from optuna.samplers import TPESampler
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

from src import constants

TRAIN_DATA_PATH = os.path.join('..', '..', 'data', 'PD-data-train.csv')
MODEL_PATH = os.path.join('..', '..', 'models', 'decision_tree_pipeline.pkl')
STUDY_PATH = os.path.join('..', '..', 'studies', 'decision_tree_study.pkl')
RESULTS_PATH = os.path.join('..', '..', 'results.csv')
N_SPLITS = 10
N_TRIALS = 100

train_data = pd.read_csv(TRAIN_DATA_PATH, sep=';', index_col='record_id')

y = train_data['default_12m']
X = train_data.drop(columns='default_12m')


def objective(trial: optuna.trial.Trial):
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
            ('OHE', OneHotEncoder(sparse_output=False), ['ul_staff_range']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    decision_tree = DecisionTreeClassifier(**get_init_hyperparams(trial, mode))

    model = Pipeline([
        ('preprocessing', preprocessing),
        ('tree', decision_tree),
    ])

    return model


def get_init_hyperparams(trial: optuna.trial.Trial, mode: str) -> Dict:
    if mode == 'fit':
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = trial.suggest_int('max_depth', 1, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 100)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 1e-5, .1, step=1e-5)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 20)
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 1e-7, 1e-3, step=1e-7)
    elif mode == 'refit':
        criterion = trial.params['criterion']
        max_depth = trial.params['max_depth']
        min_samples_split = trial.params['min_samples_split']
        min_samples_leaf = trial.params['min_samples_leaf']
        min_weight_fraction_leaf = trial.params['min_weight_fraction_leaf']
        max_leaf_nodes = trial.params['max_leaf_nodes']
        min_impurity_decrease = trial.params['min_impurity_decrease']
    else:
        assert False, 'ABOBA'

    hyperparams = dict(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
        max_leaf_nodes=max_leaf_nodes,
        random_state=constants.RANDOM_STATE,
        class_weight='balanced',
    )

    return hyperparams


if __name__ == '__main__':
    study = optuna.create_study(
        sampler=TPESampler(seed=constants.RANDOM_STATE),
        direction='maximize',
        study_name='decision_tree',
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    model = get_model(study.best_trial, mode='refit')
    model.fit(X, y)

    print('Лучшие гиперпараметры:')
    for hyperparam, value in study.best_trial.params.items():
        print(f'* {hyperparam}: {value}')
    print(f'Лучший mean AUC-ROC: {study.best_trial.value}')

    results = pd.read_csv(RESULTS_PATH, index_col=0)
    best_aucroc = study.best_trial.value
    if results.loc['meanCV AUC-ROC', 'decision_tree'] < best_aucroc:
        results.loc['meanCV AUC-ROC', 'decision_tree'] = best_aucroc
        results.to_csv(RESULTS_PATH)

        joblib.dump(study, STUDY_PATH)
        joblib.dump(model, MODEL_PATH)
