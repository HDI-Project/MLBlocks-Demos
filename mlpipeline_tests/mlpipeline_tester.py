#!/usr/bin/env python
# -*- coding: utf-8 -*-

class PipelineTester(object):

    @staticmethod
    def print_hyperparams(pipeline):
        print("\n{} fixed hyperparams:".format(pipeline.__class__.__name__))
        for hyperparam, value in pipeline.get_fixed_hyperparams().items():
            print('{}: {}'.format(hyperparam, value))

        print("\n{} tunable hyperparams:".format(pipeline.__class__.__name__))
        for hyperparam in pipeline.get_tunable_hyperparams():
            print(hyperparam)

    def get_data(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, observed, expected, *args, **kwargs):
        raise NotImplementedError

    def build_pipeline(self, *args, **kwargs):
        return self.pipeline_class()

    def score_pipeline(self, fit_params=None, predict_params=None, *args, **kwargs):
        pipeline = self.build_pipeline(*args, **kwargs)

        self.print_hyperparams(pipeline)

        X_train, X_test, y_train, y_test = self.get_data(*args, **kwargs)

        print("\nFitting pipeline...")
        pipeline.fit(X_train, y_train, fit_params)

        print("\nScoring pipeline...")
        y_pred = pipeline.predict(X_test, predict_params)

        return self.score(y_pred, y_test, *args, **kwargs)
