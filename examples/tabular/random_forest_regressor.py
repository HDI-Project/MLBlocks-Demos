#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for RandomForestRegressor on the Boston Dataset."""


from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.tabular.random_forest import RandomForestRegressor


def run(train_size=405, test_size=101):

    print("============================================")
    print("Testing Random Forest Regressor")
    print("============================================")

    housing = load_boston()
    X, X_test, y, y_test = train_test_split(
        housing.data, housing.target, train_size=train_size, test_size=test_size)

    rf_regressor = RandomForestRegressor()

    # Check that the hyperparameters are correct.
    for hyperparam in rf_regressor.get_tunable_hyperparams():
        print(hyperparam)

    # Check that the steps are correct.
    expected_steps = {'rf_regressor'}
    steps = set(rf_regressor.steps_dict.keys())
    assert expected_steps == steps

    # Check that we can score properly.
    print("\nFitting pipeline...")
    rf_regressor.fit(X, y)
    print("\nFit pipeline.")

    print("\nScoring pipeline...")
    predicted_y_val = rf_regressor.predict(X_test)
    score = r2_score(y_test, predicted_y_val)
    print("\nr2 score: %f" % score)

    return score

if __name__ == '__main__':
    run()
