from mlblocks.mlpipeline import MLPipeline


class RandomForestClassifier(MLPipeline):
    """Random forest classifier pipeline."""

    BLOCKS = ['random_forest_classifier']


class RandomForestRegressor(MLPipeline):
    """Random forest classifier pipeline."""

    BLOCKS = ['random_forest_regressor']
