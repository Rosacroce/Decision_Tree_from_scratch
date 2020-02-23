# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:44:28 2020

Importing decision_tree_functions to perform classification o regression tasks based on the user's specificaion.

@author: aless
"""

import numpy as np
import pandas as pd
import os
import random

import matplotlib.pyplot as plt

import decision_tree_functions as dtf

######################################
#     REGRESSION PLOT FUNCTIONS      #
######################################


def create_plot(df, tree, title):
    """
    DESCRIPTION:
        Scatter plot of actual versus predicted values

    """
    predictions = df.apply(dtf.predict_example, args=(tree,), axis=1)
    actual = df['label']

    plt.scatter(actual, predictions)
    plt.grid()

def plot_target_vs_feature(df, tree, feature, label_name):
    """
    DESCRIPTION:
        Creates a plot of the target variable versus a specified feature.
        The target is plotted both the actul and the predicted

    PRECONDITION:
        The target vaiable in the input dataframe is named 'label'

    """
    predictions = df.apply(dtf.predict_example, args=(tree,), axis=1)
    actual = df['label']

    fig, ax = plt.subplots()

    ax.scatter(df[feature], actual, label='actual')
    ax.scatter(df[feature], predictions, label='predicted')
    plt.grid()
    plt.xlabel(feature)
    plt.ylabel(label_name)
    plt.legend()

##########################
#     INPUT SECTION      #
##########################

data_path = r'C:\Users\aless\Desktop\AI\CORSI\GustoMSC\JB117'
data_file = 'JB117_wd_20_30_40m.pkl'

# 'sig_roll', 'sig_pitch', 'wd',
features = ['sig_z_all_legs', 'sig_roll', 'sig_pitch', 'wd', 'UF_MPM']        # list of features to keep. Some features may not be necessary
target_name = 'UF_MPM'                 # name of the taget variable. It will be changed to 'label' (required by the imported functions)

is_ml_task_classification = True    # True if 'classification', False if 'regression'

test_size = 0.3                     # can be either an integer or a percentage
# random.seed(0)

###########################
#    DATA PREPARATION     #
###########################

df_input = pd.read_pickle(os.path.join(data_path, data_file))
# df_corr = df.corr()

# df = pd.read_csv(os.path.join(data_path, data_file))
# df = df.drop('Id', axis=1)
random.seed(0)
df = df_input[features]
df = df.rename(columns={target_name: 'label'})          # decision_tree_functions.py requires the target name to be 'label'

train_df, test_df = dtf.train_test_split(df, test_size)

###########################
# DECISION TREE ALGORITHM #
###########################


# #sub_tree = {question: [yes_answer, no_answer]}
if is_ml_task_classification:                   # Classification
    tree = dtf.decision_tree_algorithm(train_df, ml_task="classification", max_depth=3)
else:
    train_df, test_df = dtf.train_test_split(df, test_size)
    tree = dtf.decision_tree_algorithm(train_df, ml_task="regression", max_depth=3)
    r_squared = dtf.calculate_r_squared(test_df, tree)
    print('r_squared: ',  r_squared)
    #create_plot(test_df, tree, title="Test Data")
    plot_target_vs_feature(test_df, tree, 'sig_z_all_legs', 'Unity Check [-]')
    plot_target_vs_feature(test_df, tree, 'sig_roll', 'Unity Check [-]')
    plot_target_vs_feature(test_df, tree, 'sig_pitch', 'Unity Check [-]')

    plot_target_vs_feature(test_df, tree, 'wd', 'Unity Check [-]')

    y_pred = test_df.apply(dtf.predict_example, args=(tree,), axis=1)
    y_test = test_df['label']

    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





    # plot_df = pd.DataFrame({"actual": actual, "predictions": predictions})

    # plot_df.plot(figsize=(18, 5), title=title)




