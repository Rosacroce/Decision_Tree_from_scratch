# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:49:29 2020

Set of functions to create classification and regression trees.

The whole script is based on:
    https://github.com/SebastianMantey

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

import random
import os


# TRAIN-TEST-SPLIT
def train_test_split(df, test_size):
    """
    DESCRIPTION:
        Performs train-test-split.
        Test size can be entered a percentage of the total entries or as an integer

    """

    if isinstance(test_size, float):                                # if test_size is entered as a percentage (float)...
        test_size = round(test_size * len(df))                      # ...make it an integer

    indices = df.index.tolist()                                      # which is here extrated a transformed to a list
    test_indices = random.sample(population=indices, k= test_size)   # train and test values are randomly picked based on their row index,

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    """
    DESCRIPTION:
        Check if is a certain partition of the data separates/isolates a whole class of the target variable.
        The term 'pure' is intended in that sense  i.e. the partition contains only one class.

    INPUT:
        - data (numpy array)  -> value of the dataframe i.e. no index or column headers

    OUTPUT:
        - True/False (boolean)

    """

    label_column = data[:, -1]                  # get label column, which HAS to be the leftmost
    unique_classes = np.unique(label_column)    # get number of unique classes

    if len(unique_classes) == 1:
        return True
    else:
        return False


def create_leaf(data, ml_task):
    """
    DESCRIPTION:
        Depending on the specified algorithm i.e. classification or regression different tasks are performed.
        If regression is chosen the mean of the label values is returned.

        If classification is chosen:
        Returns the label for the class that occurs more often in the input dataset.
        If there are two classes that appear equally often one of them is picked randomly.
        Ideally the function runs only if a pure group is found (using function 'check_purity').
        However it can be used also if data are not pure and in that case it returns the most recurring values

    INPUT:
        - data    (np array)    -> value of the dataframe i.e. no index or column headers
        - ml_task (str)         -> can either be 'regression' or 'classification'

    OUTPUT:
        - classification (str)   -> label of the classified class

    EXAMPLE:
        create_leaf(data, ml_task='regression')
        create_leaf(data, ml_task='classification')

    """

    label_column = data[:, -1]                  # get label column, which HAS to be the leftmost

    if ml_task == 'regression':
        leaf = np.mean(label_column)

    else: # classification
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)     # unique values and respective occurrence

        index = counts_unique_classes.argmax()      # find the class that appears most often
        leaf = unique_classes[index]                # class that appears more often in the data

    return leaf


def get_potential_splits(data):
    """
    DESCRIPTION:
        Returns a dictionary where the keys are the indices of the columns i.e. 0, 1, 2, 3, etc.
        The dictionary values are lists with potential splits for the feature corresping to the considered key i.e. 0, 1, 2, etc.
        For each split obtained the purity can then be assessed.

    INPUT:
        - data           (np array)  -> value of the dataframe i.e. no index or column headers

    OUTPUT:
        - potential_splits (dict)    -> keys are the featurs and values are the values where the splits can be made

    PRECONDITION:
        - The input np array 'data' cannot be a single column but it must have > 1 columns

    EXAMPLE:
        potential_splits = get_potential_splits(df.values)

    """

    potential_splits = {}
    _ , n_columns = data.shape                       # getting the number of columns to iterate on
    for column_index in range(n_columns - 1):       # skip the label column
        # potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)           # the unique values are used to set the splitting points

        potential_splits[column_index] = unique_values

    return potential_splits


def split_data(data, split_column, split_value):
    """
    DESCRIPTION:
        Split data in 2 parts at a specified value.

    INPUT:
        - data          (numpy array)  -> Dataframe values or numpy matrix. It does not work if it is a pandas df with header!
        - split_column  (int)          -> Column to consider for the split
        - split_value   (float)        -> Value along the column to split at

    OUTPUT:
        - data_below    (np array)     -> data in the left section of the splitted range
        - data_above    (np array)     -> data in the right section of the splitted range

    EXAMPLE:
        data_below, data_above = split_data(df.values, split_column=1, split_value='male')

    """

    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]           # check what type of values are in this column. See function 'determine_type_of_feature(df)'

    if type_of_feature == 'continuous':
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]

    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def calculate_mse(data):
    """
    DESCRIPTION:
        Calculate mean square error of the label column. It will be used in case of regression.
    """

    actual_values = data[:,-1]

    if len(actual_values) == 0: # empty data
        mse = 0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction)**2)

    return mse


def calculate_entropy(data):
    """
    DESCRIPTION:
        Calculate entropy of one of the 2 partition of the target resulting from a certain split.
        The calculation is done according to the formula sum(p_i*(-log2p_i)),
        where index 'i' refers to the unique classes of the target variable in the considered partition.

    INPUT:
        - data (np array) -> Dataframe values or numpy matrix. It does not work if it is a pandas df with header!

    OUTPUT:
        - entropy (float) -> Entropy of the target variable in the considered partition

    """

    label_column = data[:, -1]
    _ , counts =  np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()                       # array of probabilities, used to calculate entropy
    entropy = sum(probabilities * -np.log2(probabilities))      # entropy of the target variable associated to a certain split

    return entropy


def calculate_overall_metric(data_below, data_above, metric_function):
    """
    DESCRIPTION:
        Calculate total entropy resulting from a split as the sum of the entropies of the 2 partitions
        weighed by the respective probabilities (or percentages).
        The calculation is done using formula sum(p_j*Entropy_j),
        where the index j refers to each of the 2 partitions resulting from the split.

    INPUT:
        - data_below (np array)     -> data in the left section of the splitted range
        - data_above (np array)     -> data in the right section of the splitted range
        - metric_function (str)     -> can either be 'calculate_entropy' for classification or 'calculate_mse' for regression

    """

    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below) / n_data_points      # probability / weight for the partition on the left of the split
    p_data_above = len(data_above) / n_data_points      # probability / weight for the partition on the right of the split

    overall_metric = (p_data_below * metric_function(data_below) + p_data_above * metric_function(data_above))

    return overall_metric


def determine_best_split(data, potential_splits, ml_task):
    """
    DESCRIPTION:
        Determines the split that results in the lowest entropy (if classification) or lowest mse (regression)

    INPUT:
        - data          (numpy array)   -> Dataframe values or numpy matrix. It does not work if it is a pandas df with header!
        - potential_splits (dict)       -> keys are the featurs and values are the values where the splits can be made
        - ml_task           (str)       -> can either be 'regression' or 'classification'
    """

    first_iteration = True       # makes sure the cycle 'if first_iteration or (current_overall_metric <= overall_metric):' is executed at least once

    for column_index in potential_splits:           	                                # loop through the keys (features) of the dictionary
        for value in potential_splits[column_index]:                                    # loop through the possible splits for each feature
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)              # derive the 2 partitions

            if ml_task == 'regression':
                current_overall_metric = calculate_overall_metric(data_below, data_above, calculate_mse) # calculate overall entropy resulting from the current partition

            else:   # classification
                current_overall_metric = calculate_overall_metric(data_below, data_above, calculate_entropy) # calculate overall entropy resulting from the current partition

            if first_iteration or (current_overall_metric <= best_overall_metric):
                first_iteration = False

                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def determine_type_of_feature(df):
    """
    DESCRIPTION:
        Determine is a column feature is a contiunous value or a categorical.
        To evaluate this the number of different values in a column is considered: it is assumed that
        a categorical feature should not have 'too many' different values. A continuous feature has typically many values.

    INPUT:
        - df (pd dataframe)         -> dataset containing column names

    OUTPUT:
        - feature_types (list(str)) -> list of column types

    """
    feature_types = []                  # list containing the type of a feature (categorical or continuous)
    n_unique_values_threeshold = 15     # max number of different unique values for a feature to be considered categorical

    for column in df.columns:
        unique_values = df[column].unique()     # number of unique values for that feature
        example_value = unique_values[0]        # take first element to check its type

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threeshold):  # strings or features with few unique values
            feature_types.append('categorical')
        else:
            feature_types.append('continuous')

    return feature_types


def decision_tree_algorithm(df, ml_task, counter=0, min_samples=2, max_depth=5):
    """
    DESCRIPTION:
        Create the decision tree using a recursive function starting with a base case.
        The function allows to decide if using a pandas dataframe or a numpy array.
        The parameter that regulates this is 'counter'. Pandas dataframes are slower to process than numpy arrays.

    INPUT:
        - df (pd dataframe / np array)    -> data, including features and target.
        - counter (float)                 -> indicates if df is a pd dataframe or a numpy array
        - min_samples (int)               -> minimum number of samples
        - ml_task (str)                   -> either 'regression' or 'classification'

    OUTPUT:
        - classification (str)            -> label of the classified class
        - sub_tree (dict)                 -> tree containing splits (questions) and answers

    """

    # DATA PREPARATION
    if counter == 0:                    # this means that the input df is a pandas dataframe
        global COLUMN_HEADERS           # has to be global otherwise in the recursive steps it would be ignored
        global FEATURE_TYPES            # list of types of features
        COLUMN_HEADERS = df.columns     # get the name of the columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values                # and it has to be transformed in a numpy array
    else:
        data = df

    # BUILD DECISION TREE ALGORITHM USING A RECURSIVE FUNCTION
    if (check_purity(data)) or (len(data) < min_samples) or (counter==max_depth):       # The data contains only 1 class i.e. pure or there are very few samples
        leaf = create_leaf(data, ml_task)                    # get the name of the class that makes up the whole target
        return leaf

    # RECURSIVE PART
    else:                                       # the partition is not pure, there are > 1 classes
        counter += 1

        # HELPER FUNCTIONS
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)
        data_below, data_above = split_data(data, split_column, split_value)

        # CHECK FOR EMPTY DATA
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)                    # get the name of the class that makes up the whole target
            return leaf

        # INSTANTIATE SUB-TREE
        feature_name = COLUMN_HEADERS[split_column]

        # CHECK THE TYPE OF THE FEATURE - CONTINUOUS OR CATAEGORICAL?
        type_of_feature = FEATURE_TYPES[split_column]           # check what type of values are in this column. See function 'determine_type_of_feature(df)'

        if type_of_feature == 'continuous':
            question = f'{feature_name} <= {split_value}'
        else:
            question = f'{feature_name} = {split_value}'

        sub_tree = {question: []}

        # FIND ANSWERS (RECURSIVE PART)
        yes_answer = decision_tree_algorithm(data_below, ml_task, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, ml_task, counter, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer

        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def predict_example(example, tree):
    """
    DESCRIPTION:
        Make prediction for one specific case using a specified tree.

    INPUT:
        - example (np array) -> data point i.e. a signle row of the dataset
        - tree      (dic)    -> decision tree obtained using the function 'decision_tree_algorithm'
    """

    # sub_tree = {question: [yes_answer, no_answer]}                # this is shown for convenience: reminds that at each subtree there is a yes and no answer

    question = list(tree.keys())[0]                                 # take the questions from the tree, which is a dictionary
    feature_name, comparison_operator, value = question.split()     # split the string to derive its constituents

    # ASKING QUESTION
    if comparison_operator == '<=':
        if example[feature_name] <= float(value):
            answer = tree[question][0]                                  # 'yes' answer at that split of the decision tree. Note it can be a dict or a str.
        else:
            answer = tree[question][1]                                  # 'no' answer at that split of the decision tree. Note it can be a dict or a str.

    else:                                                               # categorical feature
        if str(example[feature_name]) == value:
            answer = tree[question][0]                                  # 'yes' answer at that split of the decision tree. Note it can be a dict or a str.
        else:
            answer = tree[question][1]

    # BASE CASE
    if not isinstance(answer, dict):                                # not a dictionary, the end of a branch was reached and a classification (str) was given
        return answer                                               # classified item

    # RECURSIVE PART
    else:                                                           # answer 'no', another dictionary that has to be processed by the function classify_example
        residual_tree = answer                                      # remind the answer is a dict
        return predict_example(example, residual_tree)             # recursion

def calculate_r_squared(df, tree):
    """
    DESCRIPTION:
        Calculate r^2 to check the accuracy of the prediction.
    """

    labels = df['label']
    mean = labels.mean()
    predictions = df.apply(predict_example, args=(tree,), axis=1)

    ss_res = sum((labels - predictions)**2)
    ss_tot = sum((labels - mean)**2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared

def calculate_accuracy(df, tree):

    df['classification'] = df.apply(predict_example, axis=1, args=(tree,))
    df['classification_correct'] = df['classification'] == df['label']

    accuracy = df['classification_correct'].mean()

    return accuracy

def hyperparameter_tuning(train_df, val_df, max_depth_range, min_samples_range):
    """
    DESCRIPTION:
        Performs parameter optimisation for decision tree regression.

    INPUT:
        - train_df (pd dataframe)           -> train dataframe
        - val_df (pd dataframe)             -> validation dataframe
        - max_depth_range (list(int))       -> range of the maximum depth of the tree to be investigated. The order has to be [min_value, max_value, step]
        - min_samples_range (list(int))     -> range of the minimum number of samples of the tree to be investigated. The order has to be [min_value, max_value, step]

    OUTPUT:
        - opt_max_depth (int)               -> optimal value (the one that maximises r^2) of the parameter 'max_depth'
        - opt_min_samples (int)             -> optimal value (the one that maximises r^2) of the parameter 'min_samples'
        - opt_r_squared_train (float)       -> maximised train r^2 obtained using 'opt_max_depth' and 'opt_max_depth'
        - opt_r_squared_val (float)         -> maximised validation r^2 obtained using 'opt_max_depth' and 'opt_max_depth'

    PRECONDITION:
        - it is supposed the function 'decision_tree_algorithm' is used to do the grid search
        - the function optimises the parameters 'max_depth' and 'min_samples'
        - no check is performed whether the input lists 'max_depth_range' and 'min_samples_range' are correct [min_value, max_value, step] (see INPUT)

    EXAMPLE:
        max_depth_range = [2,11]
        min_samples_range = [5,10,5]
        opt_max_depth, opt_min_samples, opt_r_squared_train, opt_r_squared_val = hyperparameter_tuning(train_df, val_df, [2,11], [5,10,5])

    """

    grid_search = {'max_depth':[], 'min_samples':[], 'r_squared_train':[], 'r_squared_val':[]}

    # DERIVE LIMITS FOR THE PARAMET 'max_depth' BASED ON THE INPUT LIST
    min_max_depth_range = max_depth_range[0]              # minimum
    max_max_depth_range = max_depth_range[1]              # maximum

    if len(max_depth_range) == 3:               # if also the step is imput in 3rd position of the list
        step_depth = max_depth_range[2]         # step
    else:                                       # otherwise (step not input -> list of length 2) use a step=1 by default
        step_depth = 1

    # DERIVE LIMITS FOR THE PARAMET 'min_samples' BASED ON THE INPUT LIST
    min_min_samples_range = min_samples_range[0]              # minimum
    max_min_samples_range = min_samples_range[1]              # maximum

    if len(min_samples_range) == 3:               # if also the step is imput in 3rd position of the list
        step_samples = min_samples_range[2]         # step
    else:
        step_samples = 1

    for max_depth in range(min_max_depth_range, max_max_depth_range, step_depth):
        for min_samples in range(min_min_samples_range, max_min_samples_range, step_samples):
            tree = decision_tree_algorithm(train_df, ml_task='regression', max_depth=max_depth, min_samples=min_samples)

            r_squared_train = calculate_r_squared(train_df, tree)
            r_squared_val = calculate_r_squared(val_df, tree)

            grid_search['max_depth'].append(max_depth)
            grid_search['min_samples'].append(min_samples)
            grid_search['r_squared_train'].append(r_squared_train)
            grid_search['r_squared_val'].append(r_squared_val)

    grid_search = pd.DataFrame(grid_search)
    grid_search.sort_values('r_squared_val', ascending=False)

    max_r_squared_val = grid_search['r_squared_val'].max()
    best_params = grid_search[grid_search['r_squared_val']==max_r_squared_val].values[0]

    opt_max_depth = int(best_params[0])
    opt_min_samples = int(best_params[1])
    opt_r_squared_train = best_params[2]
    opt_r_squared_val = best_params[3]

    return opt_max_depth, opt_min_samples, opt_r_squared_train, opt_r_squared_val
