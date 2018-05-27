#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for TraditionalTextPipeline on the Newsgroups Dataset."""

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlpipeline_tests import PipelineTester
from mlpipelines.text.traditional_text import TraditionalTextPipeline


class TraditionalTextPipelineTester(PipelineTester):

    pipeline_class = TraditionalTextPipeline

    def __init__(self):
        self.newsgroups = fetch_20newsgroups()

    def get_data(self):
        return train_test_split(
            self.newsgroups.data,
            self.newsgroups.target,
            train_size=9051,
            test_size=2263
        )

    def score(self, observed, expected, *args, **kwargs):
        return f1_score(observed, expected, average='micro')


if __name__ == '__main__':
    tester = TraditionalTextPipelineTester()

    score = tester.score_pipeline()

    print("TraditionalTextPipeline Score: {}".format(score))
