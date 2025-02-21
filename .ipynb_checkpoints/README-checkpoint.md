# Tree Models

In this project you can find a couple of implementation for tree-like models such as a decision tree classifier and a regressor. The models are build from scratch and can be tested in the ipynb included.

Link to project: https://github.com/JPontS/tree-models


## How It's Made:
Tech used: python

The implementations are done in basic python functions and just require NumPy for a couple extra functionalities. Both algorithms work on the same greedy fashion. The algorithm starts from a root node and tries to separate the given samples using all possible thresholds for all features. Whenever a threshold is tried, the information gain of that given split is computed and the best split is selected according to that metric. Once a split is chosen, the data is stored in the two new nodes and the process is repeated recursively. The algorithm is stopped once the information gain is not different to zero or the maximum depth is reached or the minimum samples per split conditions are no longer fulfilled. Once a leaf is reached, the value of the leave is computed, i.e. the dominating class for classification and the mean for regression.

The classes are implemented emulating the sklearn estimators for convenience, with the usual fit and predict methods.


## Lessons Learned:
This project, it's been a great way to:
* Refine my theoretical knowledge of tree models.
* Better understand the problems that may arise when implementing a recursive algorithm.
* Review some of the information theory concepts such as entropy and information gain

## Examples:

A couple of examples of how to use the module are found in the following ipynbs:

Decision Tree Classifier --> https://github.com/JPontS/tree-models/DecisionTreeClassifier.ipynb

Decision Tree Regressor --> https://github.com/JPontS/tree-models/DecisionTreeRegressor.ipynb
