import gzip
import numpy as np
from sklearn.tree import _tree
import json
from joblib import load
import random
from sklearn.metrics import r2_score
import time

def extract_tree_structure(decision_tree, feature_names, precision=4):
    """
    Extracts and returns the structure of a decision tree in a recursive dictionary format.
    
    This function traverses a decision tree and extracts information about each node, including
    the feature used for splitting (if any), the threshold for splitting, and the values at the leaf nodes.
    The structure is returned as a nested dictionary, where decision nodes contain information about
    the feature name ('n'), threshold ('t'), and recursive structures for the left ('l') and right ('r') branches.
    Leaf nodes contain the values ('v'). All numeric values are rounded to the specified precision.

    Parameters:
    - decision_tree (DecisionTreeClassifier or DecisionTreeRegressor): A trained decision tree.
    - feature_names (list of str): A list containing the names of the features used in the decision tree.
    - precision (int, optional): The number of decimal places to round numeric values to. Default is 4.

    Returns:
    - dict: A nested dictionary representing the structure of the decision tree.
    """
    tree_ = decision_tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    thresholds = np.around(tree_.threshold, decimals=precision)
    values = np.around(tree_.value, decimals=precision)

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = thresholds[node]
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return {'n': name, 't': np.around(threshold, decimals=precision), 'l': left, 'r': right}
        else:
            temp = np.around(values[node], decimals=precision)
            return {'v': temp.tolist()}

    return recurse(0, 1)

def load_model(path, zip=True):
    """
    Load a model structure from a specified file. The function can handle both compressed (.gz) and 
    uncompressed (.json) files based on the 'zip' parameter.

    Parameters:
    - path (str): The file path to the model structure. This can be a path to either a compressed (.gz)
                  file or an uncompressed (.json) file.
    - zip (bool, optional): A flag indicating whether the file is compressed. If True, the function
                            expects a compressed (.gz) file and will decompress it. If False, it expects 
                            an uncompressed (.json) file. The default value is True.

    Returns:
    - dict: The model structure loaded from the file. The structure of the returned dictionary depends
            on the format of the model saved in the file.
    """

    if zip:
        with gzip.open(path, 'rt', encoding='UTF-8') as zipfile:
            decompressed_data = zipfile.read()
        return json.loads(decompressed_data)
    else:
        with open(path, 'r') as jsonfile:
            model_structure = json.load(jsonfile)
        return model_structure

def predict_from_tree(tree, features, feature_names):
    """
    Recursively traverse a single decision tree to make a prediction based on the provided features.

    This function navigates through a decision tree, going left or right at each node depending on the 
    value of the corresponding feature, until it reaches a leaf node. At the leaf node, it returns the 
    prediction stored in that node. The structure of the tree is expected to be in a specific format, 
    where each decision node includes a feature name ('n'), a threshold value ('t'), and links to the 
    left ('l') and right ('r') child nodes. Leaf nodes contain a value ('v') which represents the prediction.

    Parameters:
    - tree (dict): The decision tree structure, represented as a nested dictionary. The dictionary
                   format should follow the specific structure described above.
    - features (list or numpy.ndarray): The input features for making the prediction. The order of 
                                        features in this list should match the order in 'feature_names'.
    - feature_names (list of str): A list of the names of the features. This list should correspond to 
                                   the features used to train the decision tree and is used to match 
                                   feature values to the features in the decision tree.

    Returns:
    - numpy.ndarray: The prediction for the input features. The prediction is extracted from the leaf
                     node reached by traversing the decision tree.
    """
    if "v" in tree:
        # Leaf node
        return np.array(tree["v"])
    else:
        # Decision node
        feature_index = feature_names.index(tree["n"])
        if features[feature_index] <= tree["t"]:
            return predict_from_tree(tree["l"], features, feature_names)
        else:
            return predict_from_tree(tree["r"], features, feature_names)


def predict_with_model(trees, features, feature_names):
    """
    Aggregates predictions from an ensemble of decision trees, typically representing a Random Forest 
    model, to make a final prediction for the given input features.

    This function iterates over each decision tree in the ensemble, uses the 'predict_from_tree' function 
    to obtain predictions based on the input features, and then aggregates these predictions to produce 
    a final prediction. For regression problems, the aggregation is typically done by averaging the predictions 
    from all trees. For classification problems, the aggregation could involve voting, but this specific 
    implementation focuses on averaging, suitable for regression.

    Parameters:
    - trees (list of dicts): The ensemble of decision trees, where each tree is represented as a nested 
                             dictionary structure as expected by the 'predict_from_tree' function.
    - features (list or numpy.ndarray): The input features for making the prediction, where each element 
                                        corresponds to a feature value.
    - feature_names (list of str): The names of the features corresponding to the order of features in the 
                                   'features' list or array. This is used to match the input features with 
                                   the features used in the decision trees.

    Returns:
    - numpy.ndarray: The final aggregated prediction from the ensemble of decision trees.
    """
    # Accumulate predictions from all trees
    predictions = [predict_from_tree(tree, features, feature_names) for tree in trees]
    # Aggregate predictions (e.g., by averaging for a regression problem)
    return np.mean(predictions, axis=0)


def compare_predictions(joblib_model_path, json_model_path):
    """
    Compares predictions made by two models loaded from given paths, one in Joblib format and the other in JSON format,
    on a randomly generated dataset. It calculates and prints the R2 score for the predictions from both models.

    Parameters:
    - joblib_model_path (str): The file path to the Joblib model file.
    - json_model_path (str): The file path to the JSON model file.

    The function assumes both models accept the same number of features and can make predictions on a similar dataset.
    It generates a dataset with 1000 instances, each instance having a number of features equal to `n_features_in_` of the Joblib model.
    It prints the R2 score to evaluate how similar the models' predictions are, considering one model's predictions as the true values and the other's as predicted values.
    """
    
    # Load models from the provided paths
    joblib_model = load(joblib_model_path)
    json_model = load_model(json_model_path)

    # Determine the number of input features based on the joblib model
    num_features = joblib_model.n_features_in_
    len_test = 1000
    print()
    print(f"Comparing both models for {len_test} sets for input features!")
    # Generate a random dataset of 1000 samples, each with `num_features` features
    features = np.random.rand(len_test, num_features)

    # Generate predictions for each model based on the generated dataset
    # Using np.squeeze to ensure the prediction arrays are in the correct shape for comparison

    prediction_joblib = np.squeeze(joblib_model.predict(features))
    prediction_json = np.squeeze(np.array([predict_with_model(json_model, feature, list(range(num_features))) for feature in features]))

    # Compare predictions dimension-wise if they are multi-dimensional, else compare directly
    if prediction_json.ndim > 1:
        # Iterate through each dimension (output feature) and calculate the R2 score
        for outputs in range(prediction_json.shape[-1]):
            pred_json = prediction_joblib[:, outputs]
            pred_joblib = prediction_json[:, outputs]
            score = r2_score(pred_json, pred_joblib)
            print(f"R2 score for output feature ({outputs}) is {round(score, 2)}")
    else:
        # Calculate and print the R2 score for single-dimensional predictions
        score = r2_score(prediction_joblib, prediction_json)
        print(f"R2 score for output feature is {round(score, 2)}")
