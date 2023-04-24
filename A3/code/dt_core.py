# version 1.0
from cProfile import label
import heapq
import math
from typing import List
from anytree import Node
import heapq

import dt_global
import dt_provided
import node_id


def compute_length(distribution):
    sum = 0
    for v in distribution.values():
        sum += v
    return sum


def compute_h_after(before_distribution, after_distribution, l_examples):
    l_before = compute_length(before_distribution)
    l_after = compute_length(after_distribution)
    probabilities_before = get_probabilities(before_distribution, l_before)
    probabilities_after = get_probabilities(after_distribution, l_after)
    before = (l_before / l_examples) * I(probabilities_before)
    after = (l_after / l_examples) * I(probabilities_after)
    h_after = before + after
    return h_after


def get_feature_index(feature: str) -> int:
    return dt_global.feature_names.index(feature)


def get_sample_distribution(examples):
    sample_distribution = {}
    for e in examples:
        sample_class = e[dt_global.label_index]
        sample_distribution[sample_class] = sample_distribution.get(sample_class, 0) + 1
    return sample_distribution


def get_majority(examples):
    sample_distribution = {}
    majority = -math.inf
    majority_class = None
    for e in examples:
        sample_class = e[dt_global.label_index]
        sample_distribution[sample_class] = sample_distribution.get(sample_class, 0) + 1
        if sample_distribution[sample_class] > majority:
            majority = sample_distribution[sample_class]
            majority_class = sample_class
        elif sample_distribution[sample_class] == majority and majority_class > sample_class:
            majority_class = sample_class
    return majority_class


def get_probabilities(sample_distribution: dict, l: int):
    return [v / l for v in sample_distribution.values()]


def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values
    :rtype: List[float]
    """
    # step 1, sort the example, assuming feature is a string indicating the target
    # we assume that feature is simply a string that can be converted to a index for now
    feature_index = get_feature_index(feature)
    feature_list = sorted([(e[feature_index], e[dt_global.label_index]) for e in examples])
    l = len(feature_list)
    potential_split = []
    i = 0
    # check if the previous value of the split point contains either 0/1 for class
    prev_contains_class = set()
    prev_value = None
    while i < l:
        if dt_provided.less_than(feature_list[i-1][0], feature_list[i][0]):   # note that since we sorted greater is impossible
            # we experienced a value change, keep increasing i until we filled the after_contains_class
            after_contains_class = set()
            after_contains_class.add(feature_list[i][1])
            after_value = feature_list[i][0]
            i += 1
            while i < l and not dt_provided.less_than(feature_list[i-1][0], feature_list[i][0]):
                after_contains_class.add(feature_list[i][1])
                i += 1
            # we either exhausted the list or we finished collecting, now we check if there is change in class
            if prev_contains_class.symmetric_difference(after_contains_class):
                # if so, we have a new midpoint that can be used as a split point
                potential_split += [(prev_value + after_value) / 2]
            # update the sequence for next parse
            prev_contains_class = after_contains_class
            prev_value = after_value
        elif i == 0 or dt_provided.less_than_or_equal_to(feature_list[i - 1][0], feature_list[i][0]):
            prev_contains_class.add(feature_list[i][1])
            prev_value = feature_list[i][0]
            i += 1
    return potential_split


def I(probabilities):
    total = 0
    for p in probabilities:
        if p == 0:
            continue
        total += p * math.log2(p)
    return -total


def choose_feature_split(examples: List, features: List[str]) -> (str, float, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None, -1, and -inf.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value.
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :return: the best feature, the best split value, the max expected information gain
    :rtype: str, float, float
    """

    l_examples = len(examples)
    sample_distribution = get_sample_distribution(examples)
    # in order the return the maximum expected information gain, we need to compute h_before too
    h_before = I(get_probabilities(sample_distribution, l_examples))

    global_min_h_after = math.inf
    global_min_split = -1
    global_min_label = None

    for feature in features:
        feature_index = get_feature_index(feature)
        if feature_index == dt_global.label_index:
            continue
        potential_splits = get_splits(examples, feature)
        min_split_value = -1
        min_split_h = math.inf
        if not potential_splits:
            # there is no valid split point
            continue

            # note the split is in sorted order, meaning that you can iterate through examples and check
            # only when you hit a split point, since we are doing binary classification we need to
            # first loop through the array, make sure we have the sampele distribution
        split_index = 0
        before_distribution = {}
        after_distribution = sample_distribution.copy()

        sorted_features = sorted([(example[feature_index], example[dt_global.label_index]) for example in examples])
        for example in sorted_features:
            if dt_provided.less_than_or_equal_to(potential_splits[split_index], example[0]):
                l_before = compute_length(before_distribution)
                l_after = compute_length(after_distribution)
                probabilities_before = get_probabilities(before_distribution, l_before)
                probabilities_after = get_probabilities(after_distribution, l_after)
                before = (l_before / l_examples) * I(probabilities_before)
                after = (l_after / l_examples) * I(probabilities_after)
                h_after = before + after
                if dt_provided.less_than(h_after, min_split_h):
                    min_split_h = h_after
                    min_split_value = potential_splits[split_index]
                split_index += 1
                if split_index >= len(potential_splits):
                    break
            sample_class = example[1]
            after_distribution[sample_class] -= 1
            before_distribution[sample_class] = before_distribution.get(sample_class, 0) + 1
        if dt_provided.less_than(min_split_h, global_min_h_after):
            global_min_h_after = min_split_h
            global_min_split = min_split_value
            global_min_label = dt_global.feature_names[feature_index]

        # note the order is the h_after, feature tag, then by the split value
    return global_min_label, global_min_split, h_before - global_min_h_after


def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """
    less = []
    more = []
    feature_index = get_feature_index(feature)

    for e in examples:
        if dt_provided.less_than_or_equal_to(e[feature_index], split):
            less += [e]
        else:
            more += [e]

    return less, more


def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """
    # the identification of leaf node is that either maximum depth is reached, or
    # the split is good enough s.t. it classifies everything to 0
    # we either reach max depth, or no feature is left
    if cur_node.depth > node_id.max_depth:
            node_id.max_depth = cur_node.depth    
    if cur_node.depth == max_depth or len(features) == 0:  # note that class is still a feature, so termination is 1 instead of 0
        cur_node.decision = get_majority(examples)
    else:
        best_feature, best_split_val, max_expected_information_gain = choose_feature_split(examples, features)
        if best_feature == None:
            # no valid split point, which can only be resulted from having
            # all examples in the same class, this means we stop as a leaf
            # note that this will be non empty, cause the empty case is captured by parent
            cur_node.decision = examples[0][dt_global.label_index]
        else:
            before_split, after_split = split_examples(examples, best_feature, best_split_val)
            cur_node.feature = best_feature
            cur_node.split = best_split_val
            cur_node.decision = get_majority(examples)
            cur_node.information_gain = max_expected_information_gain
            left_child = Node(name=node_id.node_id, parent=cur_node)
            node_id.node_id += 1

            right_child = Node(name=node_id.node_id, parent=cur_node)
            node_id.node_id += 1

            if len(before_split) == 0:
                left_child.decision = cur_node.decision
                if left_child.depth > node_id.max_depth:
                    node_id.max_depth = left_child.depth
            else:
                split_node(left_child, examples=before_split, features=features, max_depth=max_depth)
            if len(after_split) == 0:
                right_child.decision = cur_node.decision
                if right_child.depth > node_id.max_depth:
                    node_id.max_depth = right_child.depth
            else:
                split_node(right_child, examples=after_split, features=features, max_depth=max_depth)


def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """
    root_node = Node(name=node_id.node_id)
    node_id.node_id += 1
    split_node(root_node, examples, features, max_depth)
    print(node_id.max_depth)
    return root_node


def predict(cur_node: Node, example, max_depth=math.inf) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth,
    return the majority decision at this node.

    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the decision for the given example
    :rtype: int
    """
    # leaf node reached or max depth reached
    if cur_node.is_leaf or cur_node.depth == max_depth:
        return cur_node.decision
    # we are still going
    feature_index = get_feature_index(cur_node.feature)

    if dt_provided.less_than_or_equal_to(example[feature_index], cur_node.split):
        # take left child
        return predict(cur_node.children[0], example, max_depth)
    else:
        return predict(cur_node.children[1], example, max_depth)


def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf) -> float:
    """
    Given a tree with cur_node as the root, some examples,
    and optionally the max depth,
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples.
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """

    total_case = len(examples)
    successful = 0
    for e in examples:
        if predict(cur_node, e, max_depth) == e[dt_global.label_index]:
            successful += 1

    return successful / total_case


def post_prune(cur_node: Node, min_info_gain: float):
    """
    Given a tree with cur_node as the root, and the minimum information gain,
    post prunes the tree using the minimum information gain criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants.
    Go through all the leaf parents.
    If the information gain at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the information gain at every leaf parent is greater than
    or equal to the pre-defined value of the minimum information gain.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_info_gain: the minimum information gain
    :type min_info_gain: float
    """
    # if it is not a leaf node, keep propagate, note the in the case it is just a leaf node, return normally
    if not cur_node.is_leaf:
        post_prune(cur_node.children[0], min_info_gain)
        post_prune(cur_node.children[1], min_info_gain)

        # then check if you have turned into leaf parents
        if cur_node.children[0].is_leaf and cur_node.children[1].is_leaf:
            # if you have turned in to leaf node
            if dt_provided.less_than(cur_node.information_gain, min_info_gain):
                # if you are not informative enough, turn you into leaf
                for n in cur_node.children:
                    n.parent = None
