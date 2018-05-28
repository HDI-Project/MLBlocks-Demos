#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import imp
import importlib
import json
import os
from datetime import datetime

from examples.audio import audio
from examples.image import simple_cnn_classifier
from examples.image import traditional_image_pipeline
from examples.multitable import multitable
from examples.tabular import random_forest_classifier
from examples.tabular import random_forest_regressor
from examples.text import lstm_text_classifier
from examples.text import traditional_text_pipeline


def import_module(module_name):
    if os.path.isfile(module_name):
        return imp.load_source("example", module_name)
    else:
        return importlib.import_module(module_name)


def parse_arg(value):
    try:
        return json.loads(value)

    except json.JSONDecodeError:
        return value


def parse_kwargs(kwargs_list):
    kwargs = dict()
    for kwarg in kwargs_list:
        key, arg = tuple(s.strip() for s in kwarg.split('=', 1))
        kwargs[key] = parse_arg(arg)

    return kwargs



def store_results(output, example, kwargs, extra_kwargs, score, exception):
    now = datetime.utcnow()
    example_name = example.replace('/', '.')
    if example.endswith('.py'):
        example_name = example_name[:-3]

    filename = '{}_{}.json'.format(now.strftime('%Y%m%d%H%M%S'), example_name)

    results = {
        'example': example,
        'extra_kwargs': extra_kwargs,
        'kwargs': kwargs,
        'datetime': now.isoformat(),
        'score': score,
        'exception': str(exception)
    }

    filepath = os.path.join(args.output, filename)
    print("Storing results in {}".format(filepath))
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


EXAMPLES = {
    'audio': {'train_size': 740, 'test_size': 185},
    'simple_cnn_classifier': {'train_size': 56000, 'test_size': 14000, 'epochs': 12},
    'traditional_image_pipeline': {'train_size': 56000, 'test_size': 14000},
    'multitable': {},
    'random_forest_classifier': {'train_size': None, 'test_size': 0.3},
    'random_forest_regressor': {'train_size': None, 'test_size': 0.3},
    'lstm_text_classifier': {'train_size': 9051, 'test_size': 2263, 'epochs': 1},
    'traditional_text_pipeline': {'train_size': 9051, 'test_size': 2263},
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run MLBlocks pipelines.')

    parser.add_argument('--output', '-o', help="Path were results will be stored")
    parser.add_argument('--times', '-t', type=int, default=1,
                        help="Number of times to run the example")
    parser.add_argument('example', help="Available examples: {}".format(list(EXAMPLES.keys())))

    try:
        args, unknown = parser.parse_known_args()
    except SystemExit as se:
        if se.code:
            parser.print_help()
        raise

    extra_kwargs = parse_kwargs(unknown)

    example = args.example
    if example in EXAMPLES:
        kwargs = EXAMPLES[example]
        kwargs.update(extra_kwargs)
        example = locals()[example]

    elif example:
        example = import_module(example)
        kwargs = extra_kwargs

    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    for i in range(args.times):
        try:
            score = example.run(**kwargs)
            exception = None
        except Exception as e:
            score = None
            exception = e

        if args.output:
            store_results(args.output, args.example, kwargs, extra_kwargs, score, exception)
