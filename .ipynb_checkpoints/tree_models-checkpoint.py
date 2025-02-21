import numpy as np

from sklearn.exceptions import NotFittedError

class Node():
    def __init__(self, feature_index=None, threshold=None, left_node=None, right_node=None, information_gain=None, value=None):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.information_gain = information_gain

        self.value = value
        
class DecisionTreeClassifier():

    def __init__(self, min_samples_split, max_depth, n_cuts=10, criterion='gini'):
        
        self.root = None
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_cuts = n_cuts
        self.criterion = criterion

    def build_tree(self, dataset, y_array, current_depth=0):

        num_samples, num_features = dataset.shape
        
        if num_samples>=self.min_samples_split and current_depth<=self.max_depth:

            best_split = self.get_best_split(dataset, y_array)

            if best_split['information_gain'] > 0:

                #recur left branch
                left_node = self.build_tree(best_split['dataset_left'], best_split['y_left'], current_depth+1)
                
                #recur right branch
                right_node = self.build_tree(best_split['dataset_right'], best_split['y_right'], current_depth+1) 

                return Node(best_split['feature_index'], best_split['threshold'],
                               left_node, right_node, best_split['information_gain'])

        return Node(value=self.compute_leaf_value(y_array))

    def compute_leaf_value(self, y_array):
        list_y = list(y_array)
        return max(list_y, key=list_y.count)

    def get_best_split(self, dataset, y_array):
    
        best_info_split = {}
        max_information_gain = -float("inf")
        
        for feature_index in range(dataset.shape[1]):
    
            # We run through all the features and all possible thresholds
            feature_values = dataset[:,feature_index] 
            possible_thresnold = np.linspace(np.min(feature_values), np.max(feature_values), self.n_cuts)
            
            for cut in possible_thresnold:
    
                # Make cut and get features and target datasets for both offspring nodes
                dataset_left, dataset_right, y_left, y_right = self.make_split(dataset, y_array, feature_index, cut)
    
                # Check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                   
                    # Compute information gain
                    current_information_gain = self.compute_information_gain(y_array, y_left, y_right)
                    if current_information_gain > max_information_gain:
                        best_info_split['feature_index'] = feature_index
                        best_info_split['threshold'] = cut
                        best_info_split['dataset_left'] = dataset_left
                        best_info_split['dataset_right'] = dataset_right
                        best_info_split['y_left'] = y_left
                        best_info_split['y_right'] = y_right
                        best_info_split['information_gain'] = current_information_gain
                        max_information_gain = current_information_gain
    
        return  best_info_split

    def compute_entropy(self, y):   
        entropy = 0
        for label in np.unique(y):
            prob = len(y[y==label]) / len(y)
            entropy += -prob*np.log2(prob)
      
        return entropy

    def compute_gini(self, y):        
        gini = 0
        for label in np.unique(y):
            prob = len(y[y==label]) / len(y)
            gini += prob**2
        return 1.0 - gini
    
    def make_split(self, dataset, y, feature_index, threshold):
        condition = dataset[:,feature_index] <= threshold
        return dataset[condition,:], dataset[~condition,:], y[condition],  y[~condition]

    def compute_information_gain(self, y_parent, y_left, y_right):
        w_left = len(y_left)/len(y_parent)
        w_right = len(y_right)/len(y_parent)
        if self.criterion == 'gini':
            gini_parent = self.compute_gini(y_parent)
            return gini_parent - (w_left*self.compute_gini(y_left) + w_right*self.compute_gini(y_right))
        else:
            entropy_parent = self.compute_entropy(y_parent)
            return entropy_parent - (w_left*self.compute_entropy(y_left) + w_right*self.compute_entropy(y_right))

    def fit(self, X, y):
        self.root = self.build_tree(X, y, current_depth=0)
        
    def predict(self, X):
        if self.root is None:
            raise NotFittedError("This decision tree instance is not fitted yet. Call 'fit' \
                with appropriate arguments before using this estimator.")
        
        predictions = [self.make_prediction(row, self.root) for row in X]
        return predictions
        
    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feat_value = x[tree.feature_index]
        if feat_value <= tree.threshold:
            return self.make_prediction(x, tree.left_node)
        else:
            return self.make_prediction(x, tree.right_node)

    def print_tree(self, node=None, depth=0):
        if self.root is None:
            raise NotFittedError("This decision tree instance is not fitted yet. Call 'fit' \
                with appropriate arguments before using this estimator.")

        if not node:
            node = self.root

        if node.value is not None:
            print(f"{'  '*depth} Leaf value {node.value}")
        else:
            print(f"{'  '*depth} Level {depth} ---> Feature index {node.feature_index} with threshold {node.threshold} and IG {node.information_gain}.")
            print(f"{'  '*(depth+1)} left")
            self.print_tree(node.left_node, depth+1)
            print(f"{'  '*(depth+1)} right")
            self.print_tree(node.right_node, depth+1)


class DecisionTreeRegressor(DecisionTreeClassifier):

    def compute_leaf_value(self, y_array):
        return np.mean(y_array)
        return max(list_y, key=list_y.count)

    