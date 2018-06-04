from mlblocks.mlpipeline import MLPipeline


class TraditionalTextPipeline(MLPipeline):
    """Traditional text pipeline."""

    BLOCKS = ['count_vectorizer', 'to_array', 'tfidf_transformer', 'multinomial_nb']
