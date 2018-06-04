#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for LstmTextClassifier on the Newsgroups Dataset."""

import keras
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlpipelines.text.lstm_text import LstmTextClassifier


def run(train_size=90, test_size=22, epochs=1, num_classes=20, pad_length=1000):

    print("============================================")
    print("Testing Text LSTM")
    print("============================================")

    newsgroups = fetch_20newsgroups()
    X, X_test, y, y_test = train_test_split(
        newsgroups.data,
        newsgroups.target,
        train_size=train_size,
        test_size=test_size)

    lstm_text = LstmTextClassifier(num_classes=num_classes, pad_length=pad_length)

    # Check that the hyperparameters are correct.
    for hyperparam in lstm_text.get_fixed_hyperparams():
        print(
            str(hyperparam) + ":",
            lstm_text.get_fixed_hyperparams()[hyperparam])
    for hyperparam in lstm_text.get_tunable_hyperparams():
        print(hyperparam)

    # Check that the blocks are correct.
    expected_blocks = {
        'tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs'
    }
    blocks = set(lstm_text.blocks.keys())
    assert expected_blocks == blocks

    y_cat = keras.utils.np_utils.to_categorical(y)

    # Check that we can score properly.
    print("\nFitting pipeline...")
    fit_params = {('lstm_text', 'epochs'): epochs}
    lstm_text.fit(X, y_cat, fit_params=fit_params)
    print("\nFit pipeline.")

    print("\nScoring pipeline...")
    predicted_y_labels = lstm_text.predict(X_test)
    score = f1_score(predicted_y_labels, y_test, average='micro')
    print("\nf1 micro score: %f" % score)

    return score

if __name__ == '__main__':
    run()
