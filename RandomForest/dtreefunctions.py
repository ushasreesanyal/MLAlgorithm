# coding: utf-8

import numpy as np
import random
import pandas as pd

# Creating a basic version of Train Test Split which is similar to sklearn.model_selection.train_test_split
# We pass a similar arguments in this and extract a random set of data for the test set, and create the train set by removing the test data
def train_test_split(data, test_size):

    size_of_data =  len(data)
    if isinstance(test_size, float):
        test_size = round(test_size * size_of_data)
    indices = data.index
    indices = indices.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    train_data.head()
    return train_data, test_data

def train_test_split1(data, test_size, validation_size):

    size_of_data =  len(data)
    if isinstance(test_size, float):
        test_size = round(test_size * size_of_data)
    if isinstance(validation_size, float):
        validation_size = round(validation_size * size_of_data)
    indices = data.index
    indices = indices.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_data = data.loc[test_indices]
    data.drop(test_indices,inplace=True)
    indices = data.index
    indices = indices.tolist()
    validate_indices = random.sample(population=indices, k=validation_size)
    validate_data = data.loc[validate_indices]
    train_data = data.drop(validate_indices)
    return train_data, test_data, validate_data


# Based on the type of feature, we can choose how to split the tree and calculate entropy
def create_feature_type_list(data):

    feature_array = []
    unique_values_treshold = 20
    for feature in data.columns:
        if feature != "label":
            unique_values = data[feature].unique()
            total_unique_values = len(unique_values)
            random_feature_value = unique_values[0]   #Choosing any random value in the dataset to distinguish if it is a string type value
            if (isinstance(random_feature_value, str)) or (total_unique_values <= unique_values_treshold): #Checking if the random feature value extracted is of type string
                feature_array.append("Category")
            else:
                feature_array.append("Continuous")
    return feature_array


# To check if there are more than one class in the label (to make sure it is unique - only one class)
def check_purity(data):
    label_column = data[:, -1]
    total_unique_classes = len(np.unique(label_column))
    if total_unique_classes == 1:
        return True
    else:
        return False


# To find a list of how many splits for the column value datas are possible
def get_all_possible_splits(data, num_features=999):

    all_possible_splits = {}
    _, no_of_columns = data.shape
    all_column_indices = list(range(no_of_columns - 1))    # excluding the last column which is the label
    if num_features and num_features <= len(all_column_indices):
        all_column_indices = random.sample(population=all_column_indices, k=num_features) # extract k random features for RF

    for column_index in all_column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)
        all_possible_splits[column_index] = unique_values
    return all_possible_splits


# To split data based on the specific split value and index into positive and negative
def split_data(data, split_column_index, split_value):

    split_column_values = data[:, split_column_index]

    type_of_feature = FEATURE_LIST[split_column_index]
    if type_of_feature == "Category":                             # feature is categorical
        data_negative = data[split_column_values == split_value]
        data_positive = data[split_column_values != split_value]
    else:                                                        # feature is continuous
        data_negative = data[split_column_values <= split_value]
        data_positive = data[split_column_values >  split_value]

    return data_negative, data_positive


# Calculating Entropy for each part of split data
def calculate_entropy(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probability = counts / np.sum(counts)
    entropy = np.sum(probability * -np.log2(probability)) # Using Entropy formula

    #entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])


    return entropy

#Calculate overall entropy of the feature to be splitted on
def calculate_overall_entropy(data_negative, data_positive):

    num_possible_combination = len(data_negative) + len(data_positive)
    probability_data_negative = len(data_negative) / num_possible_combination
    probability_data_positive = len(data_positive) / num_possible_combination

    overall_entropy =  (probability_data_negative * calculate_entropy(data_negative)
                      + probability_data_positive * calculate_entropy(data_positive))

    return overall_entropy


def determine_best_split(data, all_possible_splits):

    overall_entropy = 9999          #Setting a high entropy
    for column_index in all_possible_splits:
        for value in all_possible_splits[column_index]:
            data_negative, data_positive = split_data(data, split_column_index=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_negative, data_positive)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value



# Classify the data; if base condition has met
def classify_data(data):

    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


# Main DT Algorithm to construct the tree
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, num_features=None):

    # data preparations
    if counter == 0:
        global COLUMN_HEADER_NAME, FEATURE_LIST
        COLUMN_HEADER_NAME = df.columns
        FEATURE_LIST = create_feature_type_list(df)
        data = df.values
    else:
        data = df


    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification


    # recursive part
    else:
        counter += 1

        # helper functions
        all_possible_splits = get_all_possible_splits(data, num_features)
        split_column_index, split_value = determine_best_split(data, all_possible_splits)
        data_negative, data_positive = split_data(data, split_column_index, split_value)

        # check for empty data
        if len(data_negative) == 0 or len(data_positive) == 0:
            classification = classify_data(data)
            return classification

        # determine question
        feature_name = COLUMN_HEADER_NAME[split_column_index]
        type_of_feature = FEATURE_LIST[split_column_index]
        if type_of_feature == "Continuous":
            question = "{} <= {}".format(feature_name, split_value)

        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_negative, counter, min_samples, max_depth, num_features)
        no_answer = decision_tree_algorithm(data_positive, counter, min_samples, max_depth, num_features)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


# 3. Make predictions
# 3.1 One example
def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
                answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


# 3.2 All examples of the test data
def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions
