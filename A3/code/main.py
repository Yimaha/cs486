import dt_core
import dt_provided
import dt_global
import dt_cv
from anytree import RenderTree
import numpy as np

# # test for dataset reading
examples = dt_provided.read_data("/u2/t28cai/cs486/A3/code/data.csv")
# examples = dt_provided.read_data("/home/justin/Work/cs486/A3/code/data.csv")
# examples = dt_provided.read_data("/home/justin/Work/cs486/A3/code/data_sepen.csv")

print(dt_global.num_label_values)
print(dt_global.feature_names)
print(dt_global.label_index)
# print([(example[2], example[8]) for example in examples])
# # test for get_splits
# examples = [[95.4375, 35.65651019, 0.301534629, 2.228429554, 2.156354515, 14.92634541, 9.785341561, 117.9905312, 0], [71.5859375, 33.70518491, 2.816973782, 12.34781475, 30.44481605, 70.39489282, 2.103614391, 2.836092635, 1], [124.875, 53.51138897, 0.011036886, -0.48655312, 1.60451505, 10.95864588, 13.2794533, 254.232943, 1], [88.6484375, 36.47721674, 0.677769335, 2.474718785,
# 0.81270903, 10.76163316, 17.32727334, 340.8326899, 0], [85.8984375, 37.11191685, 0.501840466, 2.597700459, 1.591973244, 17.27828695, 12.11384386, 153.0795951, 0], [132.046875, 52.21378221, -0.089728695, -0.439797531, 2.837792642, 17.720907, 8.826011873, 91.67276321, 1], [142.84375, 49.5986135, -0.067810145, 0.081202847, 1.234949833, 11.84660102, 13.58646917, 226.6614441, 0]]

# examples = [[23.5, 0], [2332.5, 0], [23.535, 1], [22323.5, 1], [26.532, 0], [363.5, 1], [363.5, 0]]
# print(examples)
# print(dt_core.get_splits(examples, "0"))

# test for choose_feature_split

# print(dt_core.choose_feature_split(examples, dt_global.feature_names[:-1]))


# print(dt_core.get_prediction_accuracy(root, examples[:-100]))
print(dt_cv.cv_post_prune(dt_provided.preprocess(examples), list(np.arange(0, 1.1, 0.1))))


# print(dt_core.choose_feature_split(examples, dt_global.feature_names))
