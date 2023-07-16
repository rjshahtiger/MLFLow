"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    outlier_transformer = Outlier(method='mean')
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    features_transformer = ColumnTransformer([
        
        ## categorical columns
        #('tgt_enc', TargetEncoder(return_df=False),
        #list(set(cat_columns))),
        
        # NOTE: if the same column gets repeated, then they are weighed in the final output
        # If we want a sequence of operations, then we use a pipeline but that doesen't YET support
        # get_feature_names. 
        ('tgt_enc_sim_impt', OneHotEncoder(), list(set(cat_columns))),
            
        ## numeric columns
        ('med_enc', SimpleImputer(strategy='median'), num_columns),
        
    ])

    train_X = get_dataframe(
    features_transformer.fit_transform(train_X, train_y), 
    get_feature_names_from_column_transformer(features_transformer))

    out = eda.get_correlation_table(train_X)
    out[out["Abs Corr Coef"] > 0.6]

    curated_columns = list(
    set(train_X.columns.to_list()) 
    - set(['households', 'total_bedrooms','population','latitude','ocean_proximity_<1H OCEAN']))
    train_X = train_X[curated_columns]


    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    save_pipeline(
        features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )