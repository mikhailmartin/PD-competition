import click
import joblib
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def main(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path, sep=';', index_col='record_id')
    X_train = train_data.drop(columns='default_12m')
    y_train = train_data['default_12m']

    decision_tree_pipe = get_model()
    decision_tree_pipe.fit(X_train, y_train)

    joblib.dump(decision_tree_pipe, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ('drop', 'drop', ['ul_systematizing_flg']),
            ('OHE', OneHotEncoder(sparse_output=False), ['ul_staff_range']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    decision_tree = DecisionTreeClassifier(
        criterion='log_loss',
        max_depth=11,
        min_samples_split=88,
        min_samples_leaf=20,
        min_weight_fraction_leaf=0.0112,
        min_impurity_decrease=0.0001874,
        max_leaf_nodes=13,
        random_state=42,
        class_weight='balanced',
    )

    model = Pipeline([
        ('preprocessing', preprocessing),
        ('decision_tree', decision_tree),
    ])

    return model


if __name__ == '__main__':
    main()
