import click
import joblib
import pandas as pd
import lightgbm

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import my_ds_tools


FEATURES_TO_DROP = [
    'ul_systematizing_flg',  # константаная фича
    'ul_strategic_flg',  # квазиконстантная фича (3 отличных примера)
    'head_actual_age',  # полная корреляция с 'adr_actual_age'
    'cap_actual_age',  # полная корреляция с 'adr_actual_age'
]
FEATURES_TO_ABS = [
    'ar_total_expenses',
    'ar_sale_cost',
]


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def main(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path, sep=';', index_col='record_id')
    X_train = train_data.drop(columns='default_12m')
    y_train = train_data['default_12m']

    lgbm_pipe = get_model()
    lgbm_pipe.fit(X_train, y_train)

    joblib.dump(lgbm_pipe, output_model_path)


def get_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ('drop', 'drop', FEATURES_TO_DROP),
            ('as_category', my_ds_tools.custom_column_transformers.AsCategory({'ul_staff_range': ['(100-500]', '> 500', '[1-100]']}), ['ul_staff_range']),
            ('abs', my_ds_tools.custom_column_transformers.ABS(), FEATURES_TO_ABS),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    lgbm = lightgbm.LGBMClassifier(
        # Core Parameters
        objective='binary',
        boosting_type='gbdt',
        n_estimators=266,
        learning_rate=.05,
        reg_alpha=1.8891431798868852,
        reg_lambda=1.9698467479376796,
        num_leaves=6,
        n_jobs=-1,
        random_state=42,

        # Learning Control Parameters
        max_depth=3,
        min_child_samples=47,
        min_split_gain=0.07238534804289289,

        # Objective Parameters
        is_unbalance=True,

        verbose=-1,
    )

    model = Pipeline([
        ('preprocessing', preprocessing),
        ('LGBM', lgbm),
    ])

    return model


if __name__ == '__main__':
    main()
