from mlblocks.mlpipeline import MLPipeline


class TraditionalImagePipeline(MLPipeline):
    """Traditional image pipeline using HOG features."""

    BLOCKS = ['HOG', 'random_forest_classifier']
