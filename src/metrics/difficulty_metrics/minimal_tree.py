import pandas as pd
from sklearn import tree
import numpy as np


MAXIMUM_TREE_DEPTH = 20
NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION = 12


def calculate_ralation_between_dree_depth_and_accuaracy(
    synthetic_x: pd.DataFrame,
    synthetic_y: list[int | float],
    real_x: pd.DataFrame,
    real_y: int,
):
    accuracy_for_tree_depth = {}
    if len(np.unique(synthetic_y)) > NUMBER_OF_UNIQUE_ELEMENTS_FOR_CLASIFICATION:
        tree_type = tree.DecisionTreeRegressor
    else:
        tree_type = tree.DecisionTreeClassifier
    for tree_depth in range(1, MAXIMUM_TREE_DEPTH + 1):
        decision_tree = tree_type(tree_depth=tree_depth)
        decision_tree.fit(synthetic_x, synthetic_y)
        accuracy_for_tree_depth[tree_depth] = decision_tree.score(real_x, real_y)

    return accuracy_for_tree_depth
