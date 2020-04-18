import numpy as np
import pandas as pd
import random

from dtreefunctions import decision_tree_algorithm,decision_tree_predictions,train_test_split


def bootstrapping(train_df, n_bootstrap):
    bootstrap_examples = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_examples]
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    randomforest = []
    totaltrees  = range(n_trees)
    for i in totaltrees:
        df_bootstraped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstraped, max_depth=dt_max_depth, num_features=n_features)
        randomforest.append(tree)

    return randomforest

def random_forest_predictions(test_df, forest):
    df_pred = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_pred[column_name] = predictions

    df_pred = pd.DataFrame(df_pred)
    random_forest_predictions = df_pred.mode(axis=1)[0] #Voting method implemented to find the best accuracy

    return random_forest_predictions
