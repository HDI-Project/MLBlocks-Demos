"""Usage example for Multitable pipeline on a Retail Dataset."""

import featuretools as ft
import pandas as pd
from sklearn.metrics import f1_score

from mlblocks.ml_pipeline.ml_pipeline import MLPipeline
from mlblocks.ml_pipeline.ml_hyperparam import MLHyperparam

def make_entity_set(orders_table, order_products_table):
    entity_set = ft.EntitySet("instacart")

    entity_set.entity_from_dataframe(
        entity_id="order_products",
        dataframe=order_products_table,
        index="order_product_id",
        variable_types={
            "aisle_id": ft.variable_types.Categorical,
            "reordered": ft.variable_types.Boolean
        },
        time_index="order_time"
    )

    entity_set.entity_from_dataframe(
        entity_id="orders",
        dataframe=orders_table,
        index="order_id",
        time_index="order_time"
    )

    entity_set.add_relationship(
        ft.Relationship(
            entity_set["orders"]["order_id"],
            entity_set["order_products"]["order_id"]
        )
    )

    entity_set.normalize_entity(
        base_entity_id="orders", new_entity_id="users", index="user_id")
    entity_set.add_last_time_indexes()

    entity_set["order_products"]["department"].interesting_values = [
        'produce', 'dairy eggs', 'snacks', 'beverages', 'frozen', 'pantry',
        'bakery', 'canned goods', 'deli', 'dry goods pasta'
    ]
    entity_set["order_products"]["product_name"].interesting_values = [
        'Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach',
        'Organic Strawberries', 'Organic Hass Avocado', 'Organic Avocado',
        'Large Lemon', 'Limes', 'Strawberries', 'Organic Whole Milk'
    ]
    return entity_set


def run():

    print("""
    ============================================
    Testing Multi Table Pipeline
    ============================================
    """)

    orders = pd.read_csv("data/Retail/orders.csv")
    order_products = pd.read_csv("data/Retail/order_products.csv")
    label_times = pd.read_csv("data/Retail/label_times.csv")

    X = label_times.sample(frac=0.8)
    X_test = label_times.drop(X.index)
    y = X["label"]
    y_test = X_test["label"]

    entity_set = make_entity_set(orders, order_products)

    multitable = MLPipeline.from_ml_json(['dfs', 'random_forest_classifier'])

    updated_hyperparam = MLHyperparam('max_depth', 'int', [1, 10])
    updated_hyperparam.step_name = 'dfs'
    # multitable.update_tunable_hyperparams([updated_hyperparam])

    # Check that the hyperparameters are correct.
    for hyperparam in multitable.get_tunable_hyperparams():
        print(hyperparam)

    # Check that the steps are correct.
    expected_steps = {'dfs', 'rf_classifier'}
    steps = set(multitable.steps_dict.keys())
    assert expected_steps == steps

    # Check that we can score properly.
    produce_params = {
        ('dfs', 'entityset'): entity_set,
        ('dfs', 'cutoff_time_in_index'): True
    }
    print("\nFitting pipeline...")
    fit_params = {
        ('dfs', 'entityset'): entity_set,
        ('dfs', 'target_entity'): "users",
        ('dfs', 'training_window'): ft.Timedelta("60 days")
    }
    multitable.fit(X_train, y_train, fit_params=fit_params, produce_params=produce_params)
    print("\nFit pipeline.")

    print("\nScoring pipeline...")
    predicted_y_val = multitable.predict(X_test, predict_params=produce_params)
    score = f1_score(predicted_y_val, y_test, average='micro')
    print("\nf1 micro score: %f" % score)

    return score

if __name__ == '__main__':
    run()
