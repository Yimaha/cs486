# version 1.0
from typing import List

import dt_global
import dt_core


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """
    fold_count = len(folds)
    # fold is folds, value_list is a list of value for maximum depth we will be working with
    solution_train = []
    solution_test = []
    # note that , it is simply too slow to evalute the result by building tree again and again
    # thus, we stop at max depth during validaton instead
    trees = []
    for i in range(fold_count):
        training = []
        for j in range(fold_count):
            if j != i:
                training += folds[j]
        trees.append(dt_core.learn_dt(examples=training, features=dt_global.feature_names[:-1]))
    print("tree is successfully generated")
    for max_depth in value_list:
        train_accuracies = 0.0
        test_accuracies = 0.0
        for i in range(fold_count):
            train_accuracies += dt_core.get_prediction_accuracy(trees[i], training, max_depth)
            test_accuracies += dt_core.get_prediction_accuracy(trees[i], folds[i], max_depth)
        solution_train += [train_accuracies/fold_count]
        solution_test += [test_accuracies/fold_count]
    return solution_train, solution_test


def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """
    fold_count = len(folds)
    # fold is folds, value_list is a list of value for maximum depth we will be working with
    value_count = len(value_list)
    solution_train = [0]*value_count
    solution_test = [0]*value_count
    trees = []
    sorted_list = sorted([(m, i) for i, m in enumerate(value_list)])
    for i in range(fold_count):
        training = []
        for j in range(fold_count):
            if j != i:
                training += folds[j]
        trees.append(dt_core.learn_dt(examples=training, features=dt_global.feature_names[:-1]))
    
    for min_info_gain, index in sorted_list:
        train_accuracies = 0.0
        test_accuracies = 0.0
        for i in range(fold_count):
            dt_core.post_prune(trees[i], min_info_gain)
            train_accuracies += dt_core.get_prediction_accuracy(trees[i], training)
            test_accuracies += dt_core.get_prediction_accuracy(trees[i], folds[i])
        solution_train[index] = train_accuracies/fold_count
        solution_test[index] = test_accuracies/fold_count
    return solution_train, solution_test
